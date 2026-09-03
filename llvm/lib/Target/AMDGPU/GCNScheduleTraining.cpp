//===- GCNScheduleTraining.cpp - GCN schedule trajectory endpoint --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedFeatures.h"
#include "GCNCompleteScheduleOptimizer.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineSchedSearch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <string>

using namespace llvm;
using namespace llvm::AMDGPU;

static cl::opt<std::string> TrainingFunction(
    "amdgpu-prera-training-function", cl::Hidden, cl::init(""),
    cl::desc("Restrict pre-RA schedule data generation to one function"));

static cl::opt<std::string> TrainingReplaySchedule(
    "amdgpu-prera-training-replay-schedule", cl::Hidden, cl::init(""),
    cl::desc("Replay versioned post-MaxOccupancy schedule records"));

static cl::opt<std::string> TrainingRecordSchedule(
    "amdgpu-prera-training-record-schedule", cl::Hidden, cl::init(""),
    cl::desc("Append versioned post-MaxOccupancy schedule records"));

static cl::opt<std::string> TrainingRecordTrajectory(
    "amdgpu-prera-training-record-trajectory", cl::Hidden, cl::init(""),
    cl::desc("Append schedule states and transitions as JSON Lines"));

static cl::opt<unsigned> TrainingMutationDepth(
    "amdgpu-prera-training-mutation-depth", cl::Hidden, cl::init(0),
    cl::desc("Generate an exact-depth legal relocation walk"));

static cl::opt<unsigned> TrainingMutationRegion(
    "amdgpu-prera-training-mutation-region", cl::Hidden, cl::init(0),
    cl::desc("Scheduler region to mutate during trajectory generation"));

static cl::opt<uint64_t> TrainingSeed(
    "amdgpu-prera-training-seed", cl::Hidden, cl::init(1),
    cl::desc("Deterministic seed for schedule trajectory generation"));

namespace {

constexpr StringLiteral ScheduleFormat = "amdgpu-prera-schedule-v1";
constexpr StringLiteral TrajectoryFormat = "amdgpu-prera-trajectory-v1";

class XorShift64 {
  uint64_t State;

public:
  explicit XorShift64(uint64_t Seed)
      : State(Seed ? Seed : 0x9e3779b97f4a7c15ULL) {}

  uint64_t next() {
    State ^= State << 13;
    State ^= State >> 7;
    State ^= State << 17;
    return State;
  }

  unsigned bounded(unsigned Bound) {
    assert(Bound && "random bound must be nonzero");
    return unsigned(next() % Bound);
  }
};

uint64_t stableStringHash(StringRef Text) {
  uint64_t Hash = 1469598103934665603ULL;
  for (uint8_t Byte : arrayRefFromStringRef(Text)) {
    Hash ^= Byte;
    Hash *= 1099511628211ULL;
  }
  return Hash;
}

std::string orderText(ArrayRef<unsigned> Order) {
  std::string Text;
  raw_string_ostream OS(Text);
  for (auto [Index, Node] : enumerate(Order)) {
    if (Index)
      OS << ',';
    OS << Node;
  }
  return Text;
}

std::string scheduleHash(ArrayRef<unsigned> Order) {
  std::string Text = orderText(Order);
  return toHex(SHA256::hash(arrayRefFromStringRef(Text)), true);
}

std::string regionFingerprint(const MachineSchedSearchRegion &Region,
                              StringRef FunctionName, unsigned RegionIndex,
                              ArrayRef<unsigned> Founder) {
  std::string Description;
  raw_string_ostream OS(Description);
  OS << FunctionName << '\n'
     << RegionIndex << '\n'
     << orderText(Founder) << '\n';
  for (unsigned Node = 0; Node != Region.size(); ++Node) {
    OS << Node << '\t' << *Region.getSUnit(Node).getInstr() << '\t';
    for (unsigned Pred : Region.predecessors(Node))
      OS << Pred << ',';
    OS << '\n';
  }
  OS.flush();
  return toHex(SHA256::hash(arrayRefFromStringRef(Description)), true);
}

json::Array jsonUnsignedArray(ArrayRef<unsigned> Values) {
  json::Array Result;
  Result.reserve(Values.size());
  for (unsigned Value : Values)
    Result.emplace_back(int64_t(Value));
  return Result;
}

json::Array jsonFloatArray(ArrayRef<float> Values) {
  json::Array Result;
  Result.reserve(Values.size());
  for (float Value : Values)
    Result.emplace_back(double(Value));
  return Result;
}

void appendJSONLine(StringRef Path, json::Object Record) {
  if (Path.empty())
    return;
  auto BypassSandbox = sys::sandbox::scopedDisable();
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::CD_OpenAlways, sys::fs::FA_Write,
                    sys::fs::OF_Append);
  if (EC)
    report_fatal_error(Twine("cannot append AMDGPU pre-RA trajectory: ") +
                       EC.message());
  json::OStream JSON(OS);
  JSON.value(json::Value(std::move(Record)));
  OS << '\n';
}

void appendSchedule(StringRef Path, StringRef FunctionName,
                    unsigned RegionIndex, StringRef Fingerprint,
                    ArrayRef<unsigned> Order) {
  if (Path.empty())
    return;
  auto BypassSandbox = sys::sandbox::scopedDisable();
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::CD_OpenAlways, sys::fs::FA_Write,
                    sys::fs::OF_Append);
  if (EC)
    report_fatal_error(Twine("cannot append AMDGPU pre-RA schedule: ") +
                       EC.message());
  OS << ScheduleFormat << '\t' << FunctionName << '\t' << RegionIndex << '\t'
     << Fingerprint << '\t' << orderText(Order) << '\n';
}

SmallVector<unsigned> parseOrder(StringRef Text, StringRef Path) {
  SmallVector<StringRef> Fields;
  Text.split(Fields, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  SmallVector<unsigned> Order;
  Order.reserve(Fields.size());
  for (StringRef Field : Fields) {
    unsigned Node;
    if (Field.getAsInteger(10, Node))
      report_fatal_error(Twine("invalid SUnit ordinal in ") + Path);
    Order.push_back(Node);
  }
  return Order;
}

struct ReplayRecord {
  std::string Fingerprint;
  SmallVector<unsigned> Order;
};

class GCNScheduleTrainingOptimizer final : public GCNCompleteScheduleOptimizer {
  MachineSchedContext &Context;
  DenseMap<unsigned, ReplayRecord> ReplayOrders;
  bool ReplayLoaded = false;
  bool MutatedRequestedRegion = false;
  unsigned RegionIndex = 0;

  bool handlesFunction() const {
    return TrainingFunction.empty() ||
           Context.MF->getName() == TrainingFunction.getValue();
  }

  void loadReplayFile() {
    if (ReplayLoaded || TrainingReplaySchedule.empty())
      return;
    ReplayLoaded = true;
    auto BypassSandbox = sys::sandbox::scopedDisable();
    auto Buffer = MemoryBuffer::getFile(TrainingReplaySchedule);
    if (!Buffer)
      report_fatal_error(Twine("cannot read AMDGPU pre-RA schedule: ") +
                         Buffer.getError().message());

    SmallVector<StringRef> Lines;
    (*Buffer)->getBuffer().split(Lines, '\n');
    for (StringRef Line : Lines) {
      Line = Line.trim();
      if (Line.empty() || Line.starts_with("#"))
        continue;
      SmallVector<StringRef> Fields;
      Line.split(Fields, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
      if (Fields.size() != 5 || Fields[0] != ScheduleFormat)
        report_fatal_error(Twine("invalid AMDGPU pre-RA schedule record in ") +
                           TrainingReplaySchedule);
      if (Fields[1] != Context.MF->getName())
        continue;
      unsigned Index;
      if (Fields[2].getAsInteger(10, Index))
        report_fatal_error(Twine("invalid scheduler region in ") +
                           TrainingReplaySchedule);
      ReplayRecord Record{Fields[3].str(),
                          parseOrder(Fields[4], TrainingReplaySchedule)};
      if (!ReplayOrders.try_emplace(Index, std::move(Record)).second)
        report_fatal_error(Twine("duplicate scheduler region in ") +
                           TrainingReplaySchedule);
    }
  }

  json::Object pressureJSON(const MachineSchedSearchRegion &Region,
                            ArrayRef<unsigned> Order) const {
    const GCNSubtarget &ST = Context.MF->getSubtarget<GCNSubtarget>();
    const unsigned DynamicVGPRBlockSize =
        Context.MF->getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
    GCNRegPressure Pressure =
        getGCNCompleteSchedulePressure(Region, Order, *Context.LIS);
    return json::Object{
        {"vgpr", int64_t(Pressure.getVGPRNum(ST.hasGFX90AInsts()))},
        {"sgpr", int64_t(Pressure.getSGPRNum())},
        {"occupancy",
         int64_t(Pressure.getOccupancy(ST, DynamicVGPRBlockSize))}};
  }

  void recordState(const MachineSchedSearchRegion &Region,
                   ArrayRef<unsigned> Founder, ArrayRef<unsigned> Order,
                   unsigned CurrentRegion, StringRef Fingerprint,
                   StringRef Role, LearnedPreRARegionFeatures &Features) const {
    if (TrainingRecordTrajectory.empty())
      return;
    SmallVector<float, LearnedPreRAStateFeatureCount> StateFeatures;
    Features.stateFeatures(Order, StateFeatures);
    json::Object Record{
        {"format", TrajectoryFormat.str()},
        {"kind", "state"},
        {"feature_schema_sha256", LearnedPreRAFeatureSchemaSHA256.str()},
        {"function", Context.MF->getName().str()},
        {"region", int64_t(CurrentRegion)},
        {"region_fingerprint", Fingerprint.str()},
        {"role", Role.str()},
        {"schedule_hash", scheduleHash(Order)},
        {"founder", jsonUnsignedArray(Founder)},
        {"order", jsonUnsignedArray(Order)},
        {"state_features", jsonFloatArray(StateFeatures)},
        {"pressure", pressureJSON(Region, Order)}};
    appendJSONLine(TrainingRecordTrajectory, std::move(Record));
  }

  void recordAction(const MachineSchedSearchRegion &Region,
                    ArrayRef<unsigned> Founder, ArrayRef<unsigned> Parent,
                    ArrayRef<unsigned> Child, const Relocation &Action,
                    unsigned CurrentRegion, unsigned Step, StringRef Kind,
                    StringRef Fingerprint,
                    LearnedPreRARegionFeatures &Features) const {
    if (TrainingRecordTrajectory.empty())
      return;
    SmallVector<float, LearnedPreRAStateFeatureCount> ParentFeatures;
    SmallVector<float, LearnedPreRAStateFeatureCount> ChildFeatures;
    SmallVector<float, LearnedPreRAActionFeatureCount> ActionFeatures;
    Features.stateFeatures(Parent, ParentFeatures);
    Features.stateFeatures(Child, ChildFeatures);
    Features.actionFeatures(Parent, Child, Action, ParentFeatures,
                            ChildFeatures, ActionFeatures);
    json::Object MoveObject{{"node", int64_t(Action.SUnitIdx)},
                            {"from", int64_t(Action.OldPos)},
                            {"to", int64_t(Action.NewPos)},
                            {"distance", int64_t(Action.Distance)},
                            {"depth", int64_t(Action.Depth)}};
    json::Object Record{
        {"format", TrajectoryFormat.str()},
        {"kind", Kind.str()},
        {"feature_schema_sha256", LearnedPreRAFeatureSchemaSHA256.str()},
        {"function", Context.MF->getName().str()},
        {"region", int64_t(CurrentRegion)},
        {"region_fingerprint", Fingerprint.str()},
        {"seed", int64_t(TrainingSeed.getValue())},
        {"requested_depth", int64_t(TrainingMutationDepth.getValue())},
        {"step", int64_t(Step)},
        {"parent_hash", scheduleHash(Parent)},
        {"child_hash", scheduleHash(Child)},
        {"founder", jsonUnsignedArray(Founder)},
        {"parent", jsonUnsignedArray(Parent)},
        {"child", jsonUnsignedArray(Child)},
        {"move", std::move(MoveObject)},
        {"parent_features", jsonFloatArray(ParentFeatures)},
        {"child_features", jsonFloatArray(ChildFeatures)},
        {"action_features", jsonFloatArray(ActionFeatures)},
        {"parent_pressure", pressureJSON(Region, Parent)},
        {"child_pressure", pressureJSON(Region, Child)}};
    appendJSONLine(TrainingRecordTrajectory, std::move(Record));
  }

protected:
  bool optimizeGCNCompleteSchedule(const MachineSchedSearchRegion &Region,
                                   ArrayRef<unsigned> Founder,
                                   SmallVectorImpl<unsigned> &Result) override {
    const unsigned CurrentRegion = RegionIndex++;
    if (!handlesFunction())
      return false;
    if (!Region.getDAG() || !Context.LIS || Founder.empty())
      report_fatal_error(
          "AMDGPU pre-RA training endpoint requires a live scheduling DAG");

    loadReplayFile();
    const std::string Fingerprint = regionFingerprint(
        Region, Context.MF->getName(), CurrentRegion, Founder);
    SmallVector<unsigned> Current(Founder.begin(), Founder.end());
    bool Changed = false;

    if (auto Replay = ReplayOrders.find(CurrentRegion);
        Replay != ReplayOrders.end()) {
      if (Replay->second.Fingerprint != Fingerprint)
        report_fatal_error(
            "AMDGPU pre-RA replay schedule region fingerprint mismatch");
      if (!Region.isLegalOrder(Replay->second.Order) ||
          !preservesLocalVRegOrder(Region, Founder, Replay->second.Order))
        report_fatal_error("invalid AMDGPU pre-RA replay schedule");
      Current = Replay->second.Order;
      Changed = Current != Founder;
    }

    const GCNSubtarget &ST = Context.MF->getSubtarget<GCNSubtarget>();
    GCNRegPressure FounderPressure =
        getGCNCompleteSchedulePressure(Region, Founder, *Context.LIS);
    const ScheduleDAGMI &DAG = *Region.getDAG();
    LearnedPreRARegionFeatures Features(
        Region, Founder, *DAG.TII, *DAG.TRI, DAG.MRI,
        FounderPressure.getVGPRNum(ST.hasGFX90AInsts()));
    recordState(Region, Founder, Current, CurrentRegion, Fingerprint,
                Changed ? "replayed" : "founder", Features);

    if (TrainingMutationDepth && CurrentRegion == TrainingMutationRegion) {
      const SmallVector<unsigned> MutationFounder(Current.begin(),
                                                  Current.end());
      uint64_t Seed = TrainingSeed.getValue() ^
                      stableStringHash(Context.MF->getName()) ^
                      (uint64_t(CurrentRegion) * 0x9e3779b97f4a7c15ULL);
      XorShift64 RNG(Seed);
      unsigned Completed = 0;
      unsigned Attempts = 0;
      unsigned TotalDistance = 0;
      Relocation EndpointAction;
      const unsigned AttemptLimit =
          std::max(1024u, TrainingMutationDepth.getValue() * 4096u);
      while (Completed != TrainingMutationDepth && Attempts++ != AttemptLimit) {
        const unsigned From = RNG.bounded(Current.size());
        const unsigned Node = Current[From];
        MachineSchedSearchRegion::MoveRange Range;
        if (!Region.getLegalMoveRange(Current, Node, Range))
          report_fatal_error("invalid current schedule during mutation walk");
        const unsigned Width = Range.End - Range.Begin + 1;
        if (Width <= 1)
          continue;
        unsigned Offset = RNG.bounded(Width - 1);
        unsigned To = Range.Begin + Offset;
        if (To >= From)
          ++To;

        MachineSchedSearchRegion::Relocation Move{Node, From, To};
        SmallVector<unsigned> Child;
        if (!Region.applyRelocation(Current, Move, Child) ||
            !preservesLocalVRegOrder(Region, Founder, Child) ||
            (Completed + 1 == TrainingMutationDepth &&
             Child == MutationFounder))
          continue;
        const unsigned Distance = From > To ? From - To : To - From;
        Relocation StepAction{
            Node, From, To, Distance, Features.getCategory(Node), 1};
        if (Completed == 0) {
          EndpointAction = StepAction;
          EndpointAction.Depth = TrainingMutationDepth;
        }
        TotalDistance += Distance;
        recordAction(Region, Founder, Current, Child, StepAction, CurrentRegion,
                     Completed + 1, "transition", Fingerprint, Features);
        Current = std::move(Child);
        ++Completed;
        Changed = true;
      }
      if (Completed != TrainingMutationDepth)
        report_fatal_error("AMDGPU pre-RA training endpoint could not generate "
                           "the requested exact-depth mutation walk");
      EndpointAction.Distance = TotalDistance;
      recordAction(Region, Founder, MutationFounder, Current, EndpointAction,
                   CurrentRegion, Completed, "endpoint", Fingerprint, Features);
      MutatedRequestedRegion = true;
    }

    recordState(Region, Founder, Current, CurrentRegion, Fingerprint,
                "selected", Features);
    appendSchedule(TrainingRecordSchedule, Context.MF->getName(), CurrentRegion,
                   Fingerprint, Current);
    if (!Changed)
      return false;
    Result.assign(Current.begin(), Current.end());
    return true;
  }

public:
  explicit GCNScheduleTrainingOptimizer(MachineSchedContext *C) : Context(*C) {
    if (TrainingMutationDepth && TrainingFunction.empty())
      report_fatal_error("AMDGPU pre-RA training mutation requires an exact "
                         "-amdgpu-prera-training-function");
  }

  ~GCNScheduleTrainingOptimizer() override {
    if (handlesFunction() && TrainingMutationDepth && !MutatedRequestedRegion)
      report_fatal_error(
          "AMDGPU pre-RA training mutation region was not found");
  }
};

} // namespace

bool llvm::AMDGPU::isGCNScheduleTrainingEnabled() {
  return !TrainingReplaySchedule.empty() || !TrainingRecordSchedule.empty() ||
         !TrainingRecordTrajectory.empty() || TrainingMutationDepth != 0;
}

ScheduleDAGInstrs *
llvm::AMDGPU::createGCNScheduleTrainingScheduler(MachineSchedContext *C) {
  return createGCNPostScheduleOptimizerScheduler(
      C, std::make_unique<GCNScheduleTrainingOptimizer>(C));
}
