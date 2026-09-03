//===- GCNCompleteScheduleOptimizer.cpp - GCN schedule search ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNCompleteScheduleOptimizer.h"
#include "AMDGPUBarrierLatency.h"
#include "AMDGPUExportClustering.h"
#include "AMDGPUHazardLatency.h"
#include "AMDGPUIGroupLP.h"
#include "AMDGPUMacroFusion.h"
#include "GCNSchedStrategy.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSchedSearch.h"

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

class GCNCompleteScheduleReplayer final
    : public MachineSchedCompleteScheduleReplayer {
public:
  using MachineSchedCompleteScheduleReplayer::
      MachineSchedCompleteScheduleReplayer;

  bool shouldTrackPressure() const override { return true; }
  bool shouldTrackLaneMasks() const override { return true; }
};

void addGCNInitialScheduleMutations(ScheduleDAGMI &DAG,
                                    MachineSchedContext *C) {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  DAG.addMutation(createLoadClusterDAGMutation(DAG.TII, DAG.TRI));
  if (ST.shouldClusterStores())
    DAG.addMutation(createStoreClusterDAGMutation(DAG.TII, DAG.TRI));
  DAG.addMutation(createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::Initial));
  DAG.addMutation(createAMDGPUMacroFusionDAGMutation());
  DAG.addMutation(createAMDGPUExportClusteringDAGMutation());
  DAG.addMutation(createAMDGPUBarrierLatencyDAGMutation(C->MF));
  DAG.addMutation(createAMDGPUHazardLatencyDAGMutation(C->MF));
}

class GCNPostScheduleOptimizerDAG final : public GCNScheduleDAGMILive {
  std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer;

  void finalizeGCNSchedule() override {
    if (!Optimizer)
      return;

    std::unique_ptr<MachineSchedStrategy> OriginalStrategy =
        std::move(SchedImpl);
    SchedImpl =
        std::make_unique<GCNCompleteScheduleReplayer>(std::move(Optimizer));

    for (auto [Begin, End] : getRegions()) {
      if (Begin == End)
        continue;
      MachineBasicBlock *MBB = Begin->getParent();
      unsigned InstructionCount =
          llvm::count_if(make_range(Begin, End), [](const MachineInstr &MI) {
            return !MI.isDebugInstr();
          });
      ScheduleDAGMILive::startBlock(MBB);
      ScheduleDAGMILive::enterRegion(MBB, Begin, End, InstructionCount);
      ScheduleDAGMILive::schedule();
      ScheduleDAGMILive::exitRegion();
      ScheduleDAGMILive::finishBlock();
    }

    SchedImpl = std::move(OriginalStrategy);
  }

public:
  GCNPostScheduleOptimizerDAG(
      MachineSchedContext *C,
      std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer)
      : GCNScheduleDAGMILive(C,
                             std::make_unique<GCNMaxOccupancySchedStrategy>(C)),
        Optimizer(std::move(Optimizer)) {}
};

} // namespace

bool GCNCompleteScheduleOptimizer::optimizeCompleteSchedule(
    const MachineSchedSearchRegion &Region, ArrayRef<unsigned> Founder,
    SmallVectorImpl<unsigned> &Result) {
  this->Founder.assign(Founder.begin(), Founder.end());
  return optimizeGCNCompleteSchedule(Region, Founder, Result);
}

bool GCNCompleteScheduleOptimizer::validateCompleteSchedule(
    const MachineSchedSearchRegion &Region, ArrayRef<unsigned> Result) {
  if (Founder.size() != Region.size())
    return false;

  // ScheduleDAG dependencies are the primary legality source, but large
  // regions may omit ordering between partial virtual-register definitions
  // and uses that LiveIntervals still requires. Preserve every local def/use
  // relationship that was ordered in the founder.
  SmallVector<unsigned> FounderPosition(Region.size());
  SmallVector<unsigned> ResultPosition(Region.size());
  for (auto [Position, Node] : enumerate(Founder))
    FounderPosition[Node] = Position;
  for (auto [Position, Node] : enumerate(Result))
    ResultPosition[Node] = Position;

  DenseMap<Register, SmallVector<std::pair<unsigned, bool>, 4>> Operations;
  for (unsigned Node : Founder) {
    const MachineInstr &MI = *Region.getSUnit(Node).getInstr();
    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.getReg().isVirtual() &&
          (MO.isDef() || (MO.isUse() && !MO.isUndef())))
        Operations[MO.getReg()].push_back({Node, MO.isDef()});
  }

  for (const auto &Entry : Operations) {
    ArrayRef<std::pair<unsigned, bool>> RegisterOperations = Entry.second;
    for (const auto &[DefNode, IsDef] : RegisterOperations) {
      if (!IsDef)
        continue;
      for (const auto &[UseNode, UseIsDef] : RegisterOperations) {
        if (UseIsDef)
          continue;
        if (FounderPosition[DefNode] < FounderPosition[UseNode] &&
            ResultPosition[DefNode] >= ResultPosition[UseNode])
          return false;
      }
    }
  }

  return validateGCNCompleteSchedule(Region, Founder, Result);
}

ScheduleDAGInstrs *llvm::AMDGPU::createGCNPostScheduleOptimizerScheduler(
    MachineSchedContext *C,
    std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer) {
  auto *DAG = new GCNPostScheduleOptimizerDAG(C, std::move(Optimizer));
  addGCNInitialScheduleMutations(*DAG, C);
  return DAG;
}
