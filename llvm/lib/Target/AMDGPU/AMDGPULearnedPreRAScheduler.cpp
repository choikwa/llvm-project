//===- AMDGPULearnedPreRAScheduler.cpp - Neural pre-RA search -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedPreRAScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {
constexpr unsigned Depths[] = {1, 2, 4, 8, 16, 32};
constexpr unsigned Beam = 64;

/// Bit-for-bit port of NumPy 2.1.2 SeedSequence(uint64_t) + PCG64 and the
/// uint32 Lemire path used by Generator.integers for these small ranges.
class NumPyPCG64 {
  using UInt128 = __uint128_t;
  UInt128 State = 0;
  UInt128 Increment = 0;
  bool HasUInt32 = false;
  uint32_t BufferedUInt32 = 0;

  static constexpr UInt128 Multiplier =
      (UInt128(2549297995355413924ULL) << 64) | UInt128(4865540595714422341ULL);

  static uint32_t hashMix(uint32_t Value, uint32_t &HashConstant) {
    Value ^= HashConstant;
    HashConstant *= uint32_t(0x931e8875);
    Value *= HashConstant;
    Value ^= Value >> 16;
    return Value;
  }
  static uint32_t mix(uint32_t X, uint32_t Y) {
    uint32_t Result = uint32_t(0xca01f9dd) * X - uint32_t(0x4973f715) * Y;
    return Result ^ (Result >> 16);
  }
  void step() { State = State * Multiplier + Increment; }

public:
  explicit NumPyPCG64(uint64_t Seed) {
    SmallVector<uint32_t, 2> Entropy;
    Entropy.push_back(uint32_t(Seed));
    if (Seed >> 32)
      Entropy.push_back(uint32_t(Seed >> 32));
    std::array<uint32_t, 4> Pool{};
    uint32_t HashConstant = 0x43b0d7e5;
    for (unsigned I = 0; I != Pool.size(); ++I)
      Pool[I] = hashMix(I < Entropy.size() ? Entropy[I] : 0, HashConstant);
    for (unsigned Source = 0; Source != Pool.size(); ++Source)
      for (unsigned Destination = 0; Destination != Pool.size(); ++Destination)
        if (Source != Destination)
          Pool[Destination] =
              mix(Pool[Destination], hashMix(Pool[Source], HashConstant));
    for (unsigned Source = Pool.size(); Source < Entropy.size(); ++Source)
      for (unsigned Destination = 0; Destination != Pool.size(); ++Destination)
        Pool[Destination] =
            mix(Pool[Destination], hashMix(Entropy[Source], HashConstant));

    std::array<uint32_t, 8> Generated{};
    HashConstant = 0x8b51f9dd;
    for (unsigned I = 0; I != Generated.size(); ++I) {
      uint32_t Value = Pool[I % Pool.size()] ^ HashConstant;
      HashConstant *= uint32_t(0x58f38ded);
      Value *= HashConstant;
      Generated[I] = Value ^ (Value >> 16);
    }
    std::array<uint64_t, 4> Words{};
    for (unsigned I = 0; I != Words.size(); ++I)
      Words[I] =
          uint64_t(Generated[2 * I]) | (uint64_t(Generated[2 * I + 1]) << 32);
    const UInt128 InitialState = (UInt128(Words[0]) << 64) | Words[1];
    const UInt128 InitialSequence = (UInt128(Words[2]) << 64) | Words[3];
    Increment = (InitialSequence << 1) | 1;
    step();
    State += InitialState;
    step();
  }

  uint64_t next64() {
    step();
    const uint64_t High = uint64_t(State >> 64);
    const uint64_t Low = uint64_t(State);
    const unsigned Rotation = unsigned(High >> 58);
    const uint64_t Value = High ^ Low;
    return (Value >> Rotation) | (Value << ((-Rotation) & 63));
  }
  uint32_t next32() {
    if (HasUInt32) {
      HasUInt32 = false;
      return BufferedUInt32;
    }
    uint64_t Next = next64();
    HasUInt32 = true;
    BufferedUInt32 = uint32_t(Next >> 32);
    return uint32_t(Next);
  }
  unsigned bounded(unsigned Bound) {
    assert(Bound > 0);
    if (Bound == 1)
      return 0;
    const uint32_t InclusiveRange = Bound - 1;
    const uint32_t ExclusiveRange = Bound;
    uint64_t Product = uint64_t(next32()) * ExclusiveRange;
    uint32_t Leftover = uint32_t(Product);
    if (Leftover < ExclusiveRange) {
      const uint32_t Threshold = (UINT32_MAX - InclusiveRange) % ExclusiveRange;
      while (Leftover < Threshold) {
        Product = uint64_t(next32()) * ExclusiveRange;
        Leftover = uint32_t(Product);
      }
    }
    return unsigned(Product >> 32);
  }
};

struct State {
  SmallVector<unsigned> Order;
  SmallVector<float, 22> Features;
  SmallVector<float, 55> ActionFeatures;
  double Novelty = 0.0;
  double Endpoint = 0.0;
  double Support = 0.0;
  double PathScore = 0.0;
  double Immediate = 0.0;
  double ActionScore = 0.0;
  uint64_t Hash = 0;
  unsigned LineageDepth = 0;
  unsigned RelocationDepth = 0;
};

uint64_t scheduleHash(ArrayRef<unsigned> Order) {
  return uint64_t(hash_combine_range(Order.begin(), Order.end()));
}

SmallVector<double> rank01(ArrayRef<double> Values) {
  SmallVector<unsigned> Indices(Values.size());
  std::iota(Indices.begin(), Indices.end(), 0);
  llvm::stable_sort(
      Indices, [&](unsigned A, unsigned B) { return Values[A] < Values[B]; });
  SmallVector<double> Result(Values.size(), 0.0);
  if (Values.size() > 1)
    for (unsigned Rank = 0; Rank != Indices.size(); ++Rank)
      Result[Indices[Rank]] = double(Rank) / (Values.size() - 1);
  return Result;
}

SmallVector<double> retentionScores(ArrayRef<State> States) {
  SmallVector<double> Path, Endpoint, Novelty, Support;
  for (const State &S : States) {
    Path.push_back(S.PathScore);
    Endpoint.push_back(S.Endpoint);
    Novelty.push_back(S.Novelty);
    Support.push_back(S.Support);
  }
  auto PRank = rank01(Path), ERank = rank01(Endpoint);
  auto NRank = rank01(Novelty), SRank = rank01(Support);
  SmallVector<double> Scores(States.size());
  for (unsigned I = 0; I != States.size(); ++I)
    Scores[I] =
        0.65 * PRank[I] + 0.20 * ERank[I] + 0.15 * NRank[I] - 0.15 * SRank[I];
  return Scores;
}

SmallVector<unsigned> descendingScoreOrder(ArrayRef<double> Scores) {
  SmallVector<unsigned> Indices(Scores.size());
  std::iota(Indices.begin(), Indices.end(), 0);
  llvm::stable_sort(
      Indices, [&](unsigned A, unsigned B) { return Scores[A] < Scores[B]; });
  std::reverse(Indices.begin(), Indices.end());
  return Indices;
}

bool containsOrder(const DenseMap<uint64_t, SmallVector<unsigned, 1>> &ByHash,
                   ArrayRef<State> States, uint64_t Hash,
                   ArrayRef<unsigned> Order) {
  auto It = ByHash.find(Hash);
  if (It == ByHash.end())
    return false;
  for (unsigned Index : It->second)
    if (States[Index].Order == Order)
      return true;
  return false;
}

bool relocate(const LearnedPreRARegionFeatures &Region,
              ArrayRef<unsigned> Parent, NumPyPCG64 &RNG, unsigned Depth,
              SmallVectorImpl<unsigned> &Output, Relocation &Descriptor) {
  Output.assign(Parent.begin(), Parent.end());
  bool HasFirst = false;
  unsigned TotalDistance = 0;
  SmallVector<unsigned> Position;
  for (unsigned Step = 0; Step != std::max(1u, Depth); ++Step) {
    Region.positions(Output, Position);
    const unsigned Node = RNG.bounded(Region.size());
    const unsigned Old = Position[Node];
    int Lower = -1;
    for (unsigned Pred : Region.getPredecessors(Node))
      Lower = std::max(Lower, int(Position[Pred]));
    unsigned Upper = Region.size();
    for (unsigned Succ : Region.getSuccessors(Node))
      Upper = std::min(Upper, Position[Succ]);
    const unsigned Lo = unsigned(Lower + 1);
    const unsigned Hi = Upper - 1;
    const unsigned ChoiceCount = Hi >= Lo ? Hi - Lo : 0;
    if (!ChoiceCount)
      continue;
    unsigned Choice = RNG.bounded(ChoiceCount);
    unsigned New = Lo + Choice;
    if (New >= Old)
      ++New;
    Output.erase(Output.begin() + Old);
    Output.insert(Output.begin() + New, Node);
    TotalDistance += Old > New ? Old - New : New - Old;
    if (!HasFirst) {
      Descriptor = {Node, Old, New, 0, Region.getCategory(Node), Depth};
      HasFirst = true;
    }
  }
  if (!HasFirst)
    Descriptor = {0, 0, 0, 0, Region.getCategory(0), Depth};
  Descriptor.Distance = TotalDistance;
  return HasFirst && Output != Parent;
}
} // namespace

Expected<LearnedPreRASearchResult> llvm::AMDGPU::runLearnedPreRASearch(
    const LearnedPreRARegionFeatures &Region, ArrayRef<unsigned> FounderOrder,
    PreRASchedSearchAdvisor &Advisor, unsigned Budget, uint64_t Seed) {
  Budget = std::max(1u, Budget);
  if (!Region.isLegal(FounderOrder))
    return createStringError(inconvertibleErrorCode(),
                             "invalid learned-search founder order");
  NumPyPCG64 RNG(Seed);
  SmallVector<unsigned> FounderPosition(Region.size());
  for (auto [Position, Node] : enumerate(FounderOrder))
    FounderPosition[Node] = Position;
  State Founder;
  Founder.Order.assign(FounderOrder.begin(), FounderOrder.end());
  Region.stateFeatures(Founder.Order, Founder.Features);
  auto FounderPrediction = Advisor.evaluateState(Founder.Features);
  if (!FounderPrediction)
    return FounderPrediction.takeError();
  Founder.Endpoint = FounderPrediction->Endpoint;
  Founder.Support = FounderPrediction->Support;
  Founder.Hash = scheduleHash(Founder.Order);
  SmallVector<State, 0> Population(1, Founder);
  SmallVector<State, 0> VisitedStates(1, Founder);
  DenseMap<uint64_t, SmallVector<unsigned, 1>> VisitedByHash;
  VisitedByHash[Founder.Hash].push_back(0);
  LearnedPreRASearchResult Result;

  while (Result.Generated < Budget) {
    // Generate at most one full candidate batch at a time for beam=64:
    // min(max(512, beam*16), remaining) == min(1024, remaining).
    const unsigned Count = std::min(1024u, Budget - Result.Generated);
    SmallVector<State, 0> Children;
    SmallVector<ActionPrediction> Predictions;
    SmallVector<unsigned> ParentIndices;
    DenseMap<uint64_t, SmallVector<unsigned, 1>> LocalByHash;
    for (unsigned J = 0; J != Count; ++J) {
      const unsigned ParentIndex = J % Population.size();
      const State &Parent = Population[ParentIndex];
      const unsigned Depth = Depths[(Result.Generated + J) % std::size(Depths)];
      State Child;
      Relocation Action;
      bool Changed = false;
      uint64_t ChildHash = 0;
      // The reference retries exactly twelve complete relocation walks.  If
      // all attempts collide or are unchanged, its final attempt is still
      // evaluated and admitted to the candidate batch.
      for (unsigned Retry = 0; Retry != 12; ++Retry) {
        Changed =
            relocate(Region, Parent.Order, RNG, Depth, Child.Order, Action);
        ChildHash = scheduleHash(Child.Order);
        if (Changed &&
            !containsOrder(LocalByHash, Children, ChildHash, Child.Order))
          break;
      }
      Child.Hash = ChildHash;
      LocalByHash[Child.Hash].push_back(Children.size());
      Region.stateFeatures(Child.Order, Child.Features);
      SmallVector<float, 55> ActionFeatures;
      Region.actionFeatures(Parent.Order, Child.Order, Action, Parent.Features,
                            Child.Features, ActionFeatures);
      Child.ActionFeatures = ActionFeatures;
      auto StatePred = Advisor.evaluateState(Child.Features);
      if (!StatePred)
        return StatePred.takeError();
      auto ActionPred = Advisor.evaluateAction(ActionFeatures);
      if (!ActionPred)
        return ActionPred.takeError();
      Child.Endpoint = StatePred->Endpoint;
      Child.Support = std::max(StatePred->Support, ActionPred->Support);
      Child.Immediate = ActionPred->Immediate;
      Child.LineageDepth = Parent.LineageDepth + 1;
      Child.RelocationDepth = Depth;
      SmallVector<unsigned> Position;
      Region.positions(Child.Order, Position);
      uint64_t Displacement = 0;
      for (unsigned Node = 0; Node != Region.size(); ++Node)
        Displacement += Position[Node] > FounderPosition[Node]
                            ? Position[Node] - FounderPosition[Node]
                            : FounderPosition[Node] - Position[Node];
      Child.Novelty =
          double(Displacement) /
          std::max<uint64_t>(1, uint64_t(Region.size()) * Region.size());
      Children.push_back(std::move(Child));
      Predictions.push_back(*ActionPred);
      ParentIndices.push_back(ParentIndex);
    }
    Result.Generated += Count;
    ++Result.Batches;
    if (Children.empty())
      break;

    SmallVector<double> Immediate;
    std::array<SmallVector<double>, 3> Consequence;
    for (const ActionPrediction &P : Predictions) {
      Immediate.push_back(P.Immediate);
      for (unsigned I = 0; I != 3; ++I)
        Consequence[I].push_back(P.Consequence[I]);
    }
    auto ImmediateRank = rank01(Immediate);
    std::array<SmallVector<double>, 3> ConsequenceRank = {
        rank01(Consequence[0]), rank01(Consequence[1]), rank01(Consequence[2])};
    for (unsigned I = 0; I != Children.size(); ++I) {
      const ActionPrediction &P = Predictions[I];
      const double ConsequenceMean =
          (ConsequenceRank[0][I] + ConsequenceRank[1][I] +
           ConsequenceRank[2][I]) /
          3.0;
      const double Value = 0.35 * ImmediateRank[I] + 0.25 * P.Positive +
                           0.20 * P.Accepted + 0.20 * P.Elite -
                           // Score action support before storing the maximum
                           // of action and state support for retention.
                           0.12 * std::max(0.0, P.Support - 2.5) +
                           0.15 * ConsequenceMean;
      Children[I].PathScore =
          0.92 * Population[ParentIndices[I]].PathScore + Value;
      Children[I].ActionScore = Value;
      const bool Duplicate = containsOrder(VisitedByHash, VisitedStates,
                                           Children[I].Hash, Children[I].Order);
      Result.Duplicates += Duplicate;
      if (!Duplicate) {
        VisitedByHash[Children[I].Hash].push_back(VisitedStates.size());
        VisitedStates.push_back(Children[I]);
      }
    }
    SmallVector<State, 0> Combined(Population.begin(), Population.end());
    Combined.append(Children.begin(), Children.end());
    SmallVector<unsigned> Ranked =
        descendingScoreOrder(retentionScores(Combined));
    Population.clear();
    for (unsigned I = 0; I != std::min(Beam, unsigned(Ranked.size())); ++I)
      Population.push_back(Combined[Ranked[I]]);
  }

  // Select directly from the final active population. Endpoint coefficients
  // intentionally differ from active retention.
  SmallVector<double> Path, Endpoint, Novelty, Support;
  for (const State &S : Population) {
    Path.push_back(S.PathScore);
    Endpoint.push_back(S.Endpoint);
    Novelty.push_back(S.Novelty);
    Support.push_back(S.Support);
  }
  auto PRank = rank01(Path), ERank = rank01(Endpoint);
  auto NRank = rank01(Novelty), SRank = rank01(Support);
  SmallVector<double> EndpointScores(Population.size());
  for (unsigned I = 0; I != Population.size(); ++I)
    EndpointScores[I] =
        0.60 * PRank[I] + 0.25 * ERank[I] + 0.15 * NRank[I] - 0.15 * SRank[I];
  unsigned Selected = 0;
  for (unsigned I = 1; I != Population.size(); ++I)
    if (EndpointScores[I] > EndpointScores[Selected])
      Selected = I;
  const State &Winner = Population[Selected];
  Result.SelectedOrder = Winner.Order;
  Result.SelectedActionFeatures.assign(Winner.ActionFeatures.begin(),
                                       Winner.ActionFeatures.end());
  Result.SelectedEndpoint = Winner.Endpoint;
  Result.SelectedPathScore = Winner.PathScore;
  Result.SelectedSupport = Winner.Support;
  Result.SelectedNovelty = Winner.Novelty;
  Result.SelectedActionScore = Winner.ActionScore;
  Result.SelectedLineageDepth = Winner.LineageDepth;
  Result.SelectedRelocationDepth = Winner.RelocationDepth;
  SmallVector<unsigned> Ranked =
      descendingScoreOrder(ArrayRef<double>(EndpointScores));
  for (unsigned Index : Ranked) {
    const State &S = Population[Index];
    Result.ActiveBeam.push_back({S.Order, S.Endpoint, S.PathScore, S.Support,
                                 S.Novelty, S.LineageDepth, S.RelocationDepth});
  }
  Result.Unique = VisitedStates.size();
  return Result;
}
