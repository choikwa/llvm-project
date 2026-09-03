//===- GCNLearnedScheduleOptimizer.cpp - Learned GCN schedule search ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedFeatures.h"
#include "AMDGPULearnedModelFormat.h"
#include "AMDGPULearnedPreRAAdvisor.h"
#include "AMDGPULearnedPreRAScheduler.h"
#include "GCNCompleteScheduleOptimizer.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineSchedSearch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <iterator>
#include <string>

#define DEBUG_TYPE "gcn-learned-schedule-optimizer"

using namespace llvm;
using namespace llvm::AMDGPU;

static cl::opt<bool> EnableLearnedPreRASearch(
    "amdgpu-learned-prera-sched", cl::Hidden, cl::init(false),
    cl::desc("Enable experimental neural search after GCN scheduling"));

static cl::opt<unsigned>
    LearnedPreRABudget("amdgpu-learned-prera-budget", cl::Hidden, cl::init(512),
                       cl::desc("Candidate budget for neural schedule search"));

static cl::opt<uint64_t> LearnedPreRASeed(
    "amdgpu-learned-prera-seed", cl::Hidden, cl::init(2212),
    cl::desc("Deterministic proposal seed for neural schedule search"));

static cl::opt<int> LearnedPreRARegion(
    "amdgpu-learned-prera-region", cl::Hidden, cl::init(-1),
    cl::desc("Restrict neural search to one scheduler region"));

static cl::opt<std::string>
    LearnedPreRAModelPath("amdgpu-learned-prera-model", cl::Hidden,
                          cl::init(""),
                          cl::desc("Path to the frozen neural model blob"));

namespace {

GCNRegPressure getSchedulePressure(const MachineSchedSearchRegion &Region,
                                   ArrayRef<unsigned> Order,
                                   const LiveIntervals &LIS) {
  const ScheduleDAGMI &DAG = *Region.getDAG();
  MachineBasicBlock *MBB =
      Region.getSUnit(Order.front()).getInstr()->getParent();
  auto BBEnd = MBB->end();
  GCNUpwardRPTracker Tracker(LIS);
  if (DAG.end() != BBEnd) {
    Tracker.reset(*DAG.end());
    Tracker.recede(*DAG.end());
  } else {
    Tracker.reset(*std::prev(BBEnd));
  }
  for (unsigned Node : reverse(Order))
    Tracker.recede(*Region.getSUnit(Node).getInstr());
  return Tracker.getMaxPressureAndReset();
}

class GCNNeuralScheduleOptimizer final : public GCNCompleteScheduleOptimizer {
  MachineSchedContext &Context;
  std::shared_ptr<const LearnedPreRAModel> Model;
  bool TriedToLoadModel = false;
  unsigned RegionIndex = 0;

  bool loadModel() {
    if (Model)
      return true;
    if (TriedToLoadModel)
      return false;
    TriedToLoadModel = true;
    if (LearnedPreRAModelPath.empty()) {
      LLVM_DEBUG(dbgs() << "GCN neural schedule search: model path is unset\n");
      return false;
    }
    auto ModelOrErr = getCachedLearnedPreRAModel(LearnedPreRAModelPath);
    if (!ModelOrErr) {
      std::string Message = toString(ModelOrErr.takeError());
      LLVM_DEBUG(dbgs() << "GCN neural schedule search: " << Message << '\n');
      return false;
    }
    Model = std::move(*ModelOrErr);
    return true;
  }

protected:
  bool optimizeGCNCompleteSchedule(const MachineSchedSearchRegion &Region,
                                   ArrayRef<unsigned> Founder,
                                   SmallVectorImpl<unsigned> &Result) override {
    const unsigned CurrentRegion = RegionIndex++;
    if (LearnedPreRARegion >= 0 &&
        unsigned(LearnedPreRARegion) != CurrentRegion)
      return false;
    if (Founder.size() < 3 || !Region.getDAG() || !Context.LIS ||
        Context.MF->getSubtarget<GCNSubtarget>().getCPU() != "gfx950" ||
        !loadModel())
      return false;

    const GCNSubtarget &ST = Context.MF->getSubtarget<GCNSubtarget>();
    GCNRegPressure FounderPressure =
        getSchedulePressure(Region, Founder, *Context.LIS);
    float FounderVGPRs = FounderPressure.getVGPRNum(ST.hasGFX90AInsts());
    const ScheduleDAGMI &DAG = *Region.getDAG();
    LearnedPreRARegionFeatures Features(Region, Founder, *DAG.TII, *DAG.TRI,
                                        DAG.MRI, FounderVGPRs);
    NeuralPreRASchedSearchAdvisor Advisor(Model);
    uint64_t Seed =
        uint64_t(LearnedPreRASeed) + uint64_t(CurrentRegion) * 0x9e3779b9ULL;
    auto Search = runLearnedPreRASearch(
        Features, Founder, Advisor, std::max(1u, LearnedPreRABudget.getValue()),
        Seed);
    if (!Search) {
      std::string Message = toString(Search.takeError());
      LLVM_DEBUG(dbgs() << "GCN neural schedule search: " << Message << '\n');
      return false;
    }

    Result.assign(Search->SelectedOrder.begin(), Search->SelectedOrder.end());
    return true;
  }

  bool validateGCNCompleteSchedule(const MachineSchedSearchRegion &Region,
                                   ArrayRef<unsigned>,
                                   ArrayRef<unsigned> Result) override {
    if (!Context.LIS || Result.empty())
      return false;
    const GCNSubtarget &ST = Context.MF->getSubtarget<GCNSubtarget>();
    const unsigned DynamicVGPRBlockSize =
        Context.MF->getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
    GCNRegPressure Pressure = getSchedulePressure(Region, Result, *Context.LIS);
    Context.MF->getInfo<SIMachineFunctionInfo>()->limitOccupancy(
        Pressure.getOccupancy(ST, DynamicVGPRBlockSize));
    return true;
  }

public:
  explicit GCNNeuralScheduleOptimizer(MachineSchedContext *C) : Context(*C) {}
};

} // namespace

bool llvm::AMDGPU::isGCNNeuralScheduleEnabled() {
  return EnableLearnedPreRASearch;
}

ScheduleDAGInstrs *
llvm::AMDGPU::createGCNNeuralPostScheduler(MachineSchedContext *C) {
  return createGCNPostScheduleOptimizerScheduler(
      C, std::make_unique<GCNNeuralScheduleOptimizer>(C));
}
