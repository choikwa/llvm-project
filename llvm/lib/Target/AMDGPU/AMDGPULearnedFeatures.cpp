//===- AMDGPULearnedFeatures.cpp - Learned pre-RA features ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedFeatures.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSchedSearch.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace llvm;
using namespace llvm::AMDGPU;

uint8_t LearnedPreRARegionFeatures::classify(StringRef Opcode) {
  std::string Upper = Opcode.upper();
  StringRef U(Upper);
  if (U.contains("LOAD") &&
      (U.contains("BUFFER") || U.contains("GLOBAL") || U.contains("FLAT")))
    return 0;
  if (U.contains("DS_"))
    return 1;
  if (U.contains("MFMA"))
    return 2;
  if (U.starts_with("V_"))
    return 3;
  if (U.contains("WAIT"))
    return 4;
  return 5;
}

LearnedPreRARegionFeatures::LearnedPreRARegionFeatures(
    const MachineSchedSearchRegion &Region, ArrayRef<unsigned> Founder,
    const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
    const MachineRegisterInfo &MRI, float AllocatedVGPRs)
    : NodeCount(Region.size()), AllocatedVGPRs(AllocatedVGPRs),
      Predecessors(NodeCount), Successors(NodeCount),
      BaselinePositions(NodeCount) {
  DenseMap<const SUnit *, unsigned> Ordinals;
  for (unsigned Node = 0; Node != NodeCount; ++Node)
    Ordinals[&Region.getSUnit(Node)] = Node;
  for (auto [Position, Node] : enumerate(Founder))
    BaselinePositions[Node] = Position;
  for (unsigned Node = 0; Node != NodeCount; ++Node) {
    const SUnit &SU = Region.getSUnit(Node);
    Categories.push_back(classify(TII.getName(SU.getInstr()->getOpcode())));
  }

  for (unsigned To = 0; To != NodeCount; ++To)
    for (unsigned From : Region.predecessors(To)) {
      Predecessors[To].push_back(From);
      Successors[From].push_back(To);
    }
  // Preserve the frozen feature semantics independently of the stronger
  // legality graph above: artificial edges constrain search but were not part
  // of the model's generic latency features.
  for (unsigned To = 0; To != NodeCount; ++To)
    for (const SDep &Pred : Region.getSUnit(To).Preds) {
      auto From = Ordinals.find(Pred.getSUnit());
      if (From == Ordinals.end() || Pred.isArtificial())
        continue;
      GenericPairs.push_back({From->second, To});
      GenericLatencies.push_back(float(Pred.getLatency()));
    }

  SmallVector<Register> VRegs;
  struct RawOperand {
    unsigned Node;
    Register Reg;
    bool IsDef;
    bool IsUse;
    float Weight;
  };
  SmallVector<RawOperand> RawOperands;
  for (unsigned I = 0; I != NodeCount; ++I) {
    const MachineInstr &MI = *Region.getSUnit(I).getInstr();
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      Register Reg = MO.getReg();
      VRegs.push_back(Reg);
      const TargetRegisterClass *RC = MRI.getRegClass(Reg);
      StringRef ClassName = TRI.getRegClassName(RC);
      const bool IsVector = ClassName.contains("VGPR") ||
                            ClassName.contains("VReg") ||
                            ClassName.starts_with("AV_");
      const float Weight =
          IsVector ? float(std::max<uint64_t>(
                         1, divideCeil(TRI.getRegSizeInBits(*RC), 32u)))
                   : 0.0f;
      RawOperands.push_back({I, Reg, MO.isDef(), MO.isUse(), Weight});
    }
  }
  llvm::sort(VRegs, [](Register A, Register B) {
    return A.virtRegIndex() < B.virtRegIndex();
  });
  VRegs.erase(std::unique(VRegs.begin(), VRegs.end()), VRegs.end());
  DenseMap<Register, unsigned> VRegIDs;
  for (unsigned I = 0; I != VRegs.size(); ++I)
    VRegIDs[VRegs[I]] = I;
  VRegCount = VRegs.size();

  for (Register Reg : VRegs) {
    SmallVector<unsigned> Defs;
    SmallVector<unsigned> Uses;
    for (const RawOperand &Op : RawOperands)
      if (Op.Reg == Reg) {
        if (Op.IsDef)
          Defs.push_back(Op.Node);
        if (Op.IsUse)
          Uses.push_back(Op.Node);
      }
    for (unsigned Def : Defs)
      for (unsigned Use : Uses) {
        if (Def == Use)
          continue;
        if (Categories[Def] == 0)
          M1Pairs.push_back({Def, Use});
        if (Categories[Def] == 1 && Categories[Use] == 2)
          M2Pairs.push_back({Def, Use});
      }
  }
  for (const RawOperand &Op : RawOperands)
    if (Op.Weight > 0.0f)
      Operands.push_back(
          {Op.Node, VRegIDs[Op.Reg], Op.IsDef, Op.IsUse, Op.Weight});
}

void LearnedPreRARegionFeatures::positions(
    ArrayRef<unsigned> Order, SmallVectorImpl<unsigned> &Result) const {
  Result.assign(NodeCount, 0);
  for (unsigned I = 0; I != Order.size(); ++I)
    Result[Order[I]] = I;
}

bool LearnedPreRARegionFeatures::isPermutation(ArrayRef<unsigned> Order) const {
  if (Order.size() != NodeCount)
    return false;
  BitVector Seen(NodeCount);
  for (unsigned Node : Order) {
    if (Node >= NodeCount || Seen.test(Node))
      return false;
    Seen.set(Node);
  }
  return true;
}

bool LearnedPreRARegionFeatures::isLegal(ArrayRef<unsigned> Order) const {
  if (!isPermutation(Order))
    return false;
  SmallVector<unsigned> Position;
  positions(Order, Position);
  for (const Pair &Edge : GenericPairs)
    if (Position[Edge.First] >= Position[Edge.Second])
      return false;
  return true;
}

static double quantile(SmallVectorImpl<double> &Values, double Q) {
  if (Values.empty())
    return 0.0;
  llvm::sort(Values);
  const double Index = (Values.size() - 1) * Q;
  const size_t Lower = size_t(std::floor(Index));
  const size_t Upper = size_t(std::ceil(Index));
  const double Fraction = Index - Lower;
  return Values[Lower] + Fraction * (Values[Upper] - Values[Lower]);
}

static void appendStats(ArrayRef<LearnedPreRARegionFeatures::Pair> Pairs,
                        ArrayRef<unsigned> Position, double Scale,
                        SmallVectorImpl<float> &Result) {
  if (Pairs.empty()) {
    Result.append(4, 0.0f);
    return;
  }
  SmallVector<double> Values;
  double Sum = 0.0;
  double Minimum = std::numeric_limits<double>::infinity();
  for (const auto &Pair : Pairs) {
    const double Value = double(Position[Pair.Second]) - Position[Pair.First];
    Values.push_back(Value);
    Sum += Value;
    Minimum = std::min(Minimum, Value);
  }
  SmallVector<double> Sorted10(Values);
  SmallVector<double> Sorted90(Values);
  Result.push_back(float((Sum / Values.size()) / Scale));
  Result.push_back(float(Minimum / Scale));
  Result.push_back(float(quantile(Sorted10, 0.1) / Scale));
  Result.push_back(float(quantile(Sorted90, 0.9) / Scale));
}

void LearnedPreRARegionFeatures::stateFeatures(
    ArrayRef<unsigned> Order, SmallVectorImpl<float> &Result) const {
  SmallVector<unsigned> Position;
  positions(Order, Position);
  const double Scale = std::max(1u, NodeCount - 1);
  Result.clear();
  Result.reserve(22);
  appendStats(M1Pairs, Position, Scale, Result);
  appendStats(M2Pairs, Position, Scale, Result);

  double DeficitSum = 0.0;
  double DeficitMax = 0.0;
  float LatencyMax = 1.0f;
  for (float Latency : GenericLatencies)
    LatencyMax = std::max(LatencyMax, Latency);
  for (unsigned I = 0; I != GenericPairs.size(); ++I) {
    const Pair &Edge = GenericPairs[I];
    const double Distance =
        double(Position[Edge.Second]) - Position[Edge.First];
    const double Deficit =
        std::max(0.0, double(GenericLatencies[I]) - Distance);
    DeficitSum += Deficit;
    DeficitMax = std::max(DeficitMax, Deficit);
  }
  Result.push_back(
      GenericPairs.empty()
          ? 0.0f
          : float((DeficitSum / GenericPairs.size()) / LatencyMax));
  Result.push_back(float(DeficitMax / LatencyMax));

  SmallVector<unsigned> Starts(VRegCount, NodeCount);
  SmallVector<int> Ends(VRegCount, -1);
  SmallVector<float> Weights(VRegCount, 0.0f);
  for (const Operand &Op : Operands) {
    if (Op.IsDef)
      Starts[Op.VReg] = std::min(Starts[Op.VReg], Position[Op.Node]);
    if (Op.IsUse)
      Ends[Op.VReg] = std::max(Ends[Op.VReg], int(Position[Op.Node]));
    Weights[Op.VReg] = std::max(Weights[Op.VReg], Op.Weight);
  }
  SmallVector<float> Diff(NodeCount + 1, 0.0f);
  for (unsigned I = 0; I != VRegCount; ++I)
    if (Starts[I] < NodeCount && Ends[I] >= int(Starts[I]))
      Diff[Starts[I]] += Weights[I];
  for (unsigned I = 0; I != VRegCount; ++I)
    if (Starts[I] < NodeCount && Ends[I] >= int(Starts[I]))
      Diff[unsigned(Ends[I]) + 1] -= Weights[I];
  float Current = 0.0f, Peak = 0.0f, Area = 0.0f;
  for (unsigned I = 0; I != NodeCount; ++I) {
    Current += Diff[I];
    Peak = std::max(Peak, Current);
    Area += Current;
  }
  const float LiveMean = NodeCount ? Area / NodeCount : 0.0f;
  const float VGPRScale = std::max(1.0f, AllocatedVGPRs);
  Result.push_back(Peak / VGPRScale);
  Result.push_back(Area / std::max(1.0f, NodeCount * VGPRScale));
  Result.push_back(LiveMean / VGPRScale);

  uint64_t DisplacementSum = 0;
  unsigned DisplacementMax = 0, Displaced = 0;
  for (unsigned Node = 0; Node != NodeCount; ++Node) {
    unsigned D = Position[Node] > BaselinePositions[Node]
                     ? Position[Node] - BaselinePositions[Node]
                     : BaselinePositions[Node] - Position[Node];
    DisplacementSum += D;
    DisplacementMax = std::max(DisplacementMax, D);
    Displaced += D != 0;
  }
  Result.push_back(float((double(DisplacementSum) / NodeCount) / Scale));
  Result.push_back(float(double(DisplacementMax) / Scale));
  Result.push_back(float(double(Displaced) / NodeCount));
  for (unsigned Category = 0; Category != 6; ++Category) {
    uint64_t Sum = 0;
    unsigned Count = 0;
    for (unsigned Node = 0; Node != NodeCount; ++Node)
      if (Categories[Node] == Category) {
        Sum += Position[Node];
        ++Count;
      }
    Result.push_back(Count ? float((double(Sum) / Count) / Scale) : 0.0f);
  }
}

void LearnedPreRARegionFeatures::actionFeatures(
    ArrayRef<unsigned> Parent, ArrayRef<unsigned> Child,
    const Relocation &Action, ArrayRef<float> ParentFeatures,
    ArrayRef<float> ChildFeatures, SmallVectorImpl<float> &Result) const {
  (void)Parent;
  (void)Child;
  Result.clear();
  Result.reserve(55);
  Result.append(ParentFeatures.begin(), ParentFeatures.end());
  for (unsigned I = 0; I != 22; ++I)
    Result.push_back(ChildFeatures[I] - ParentFeatures[I]);
  for (unsigned I = 0; I != 6; ++I)
    Result.push_back(I == Action.Category ? 1.0f : 0.0f);
  const float Scale = float(std::max(1u, NodeCount - 1));
  Result.push_back(Action.OldPos / Scale);
  Result.push_back(Action.NewPos / Scale);
  Result.push_back((int(Action.NewPos) - int(Action.OldPos)) / Scale);
  Result.push_back(Action.Distance / Scale);
  Result.push_back(float(std::log2(std::max(1u, Action.Depth)) / 8.0));
}
