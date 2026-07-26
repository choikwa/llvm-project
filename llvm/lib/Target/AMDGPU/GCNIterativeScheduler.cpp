//===- GCNIterativeScheduler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the class GCNIterativeScheduler.
///
//===----------------------------------------------------------------------===//

#include "GCNIterativeScheduler.h"
#include "AMDGPUBarrierLatency.h"
#include "AMDGPUExportClustering.h"
#include "AMDGPUHazardLatency.h"
#include "AMDGPUMacroFusion.h"
#include "AMDGPUIGroupLP.h"
#include "GCNSchedStrategy.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

static cl::opt<unsigned> AnnealingIterations(
    "amdgpu-annealing-iterations", cl::Hidden, cl::init(1000),
    cl::desc("Number of iterations per tile for the experimental AMDGPU "
             "simulated annealing scheduler"));

static cl::opt<unsigned> AnnealingSeed(
    "amdgpu-annealing-seed", cl::Hidden, cl::init(1),
    cl::desc("Seed for the experimental AMDGPU simulated annealing scheduler"));

static cl::opt<unsigned> AnnealingTileSize(
    "amdgpu-annealing-tile-size", cl::Hidden, cl::init(32),
    cl::desc("Instruction tile size for AMDGPU simulated annealing"));

static cl::opt<unsigned> AnnealingTileOverlap(
    "amdgpu-annealing-tile-overlap", cl::Hidden, cl::init(8),
    cl::desc("Overlap between AMDGPU simulated annealing tiles"));

static cl::opt<unsigned> AnnealingSweeps(
    "amdgpu-annealing-sweeps", cl::Hidden, cl::init(2),
    cl::desc("Alternating tiled sweeps for AMDGPU simulated annealing"));

static cl::opt<unsigned> AnnealingStallLimit(
    "amdgpu-annealing-stall-limit", cl::Hidden, cl::init(250),
    cl::desc("Stop an AMDGPU annealing tile after this many proposals without "
             "a local improvement (zero disables early stopping)"));

static cl::opt<unsigned> AnnealingVMEMExposureWeight(
    "amdgpu-annealing-vmem-exposure-weight", cl::Hidden, cl::init(0),
    cl::desc("Weight for separating VMEM instructions from their earliest "
             "true-data consumers in production annealing"));

static cl::opt<unsigned> AnnealingMaxOccupancyLoss(
    "amdgpu-annealing-max-occupancy-loss", cl::Hidden, cl::init(0),
    cl::desc("Maximum waves per execution unit that production annealing may "
             "trade for a lower static score"));

static cl::opt<bool> AnnealingOracle(
    "amdgpu-annealing-oracle", cl::Hidden, cl::init(false),
    cl::desc("Explore and emit diverse pressure-safe AMDGPU schedules for "
             "hardware-oracle experiments"));

static cl::opt<bool> AnnealingOracleInputStart(
    "amdgpu-annealing-oracle-input-start", cl::Hidden, cl::init(false),
    cl::desc("Start AMDGPU hardware-oracle exploration from the incoming "
             "pre-scheduler instruction order"));

static cl::opt<bool> AnnealingHierarchical(
    "amdgpu-annealing-hierarchical", cl::Hidden, cl::init(false),
    cl::desc("Optimize recursively larger AMDGPU annealing tiles"));

static cl::opt<unsigned> AnnealingHierarchicalMinTile(
    "amdgpu-annealing-hierarchical-min-tile", cl::Hidden, cl::init(8),
    cl::desc("Smallest tile in hierarchical AMDGPU annealing"));

static cl::opt<bool> AnnealingOracleVMEMMoves(
    "amdgpu-annealing-oracle-vmem-moves", cl::Hidden, cl::init(false),
    cl::desc("Bias AMDGPU oracle insertion moves toward earlier VMEM issue"));

static cl::opt<bool> AnnealingOracleRelaxedMoves(
    "amdgpu-annealing-oracle-relaxed-moves", cl::Hidden, cl::init(false),
    cl::desc("Allow bidirectional VMEM, consumer, and compute insertion moves "
             "in AMDGPU oracle exploration"));

static cl::opt<double> AnnealingTemperatureScale(
    "amdgpu-annealing-temperature-scale", cl::Hidden, cl::init(0.05),
    cl::desc("Initial annealing temperature as a fraction of local score"));

static cl::opt<bool> AnnealingOracleRelaxPressure(
    "amdgpu-annealing-oracle-relax-pressure", cl::Hidden, cl::init(false),
    cl::desc("Allow oracle schedules with higher pressure when occupancy is "
             "unchanged; requires post-RA filtering"));

static cl::opt<std::string> AnnealingReplaySchedule(
    "amdgpu-annealing-replay-schedule", cl::Hidden, cl::init(""),
    cl::desc("Start AMDGPU annealing regions from a recorded schedule"));

static cl::opt<std::string> AnnealingRecordSchedule(
    "amdgpu-annealing-record-schedule", cl::Hidden, cl::init(""),
    cl::desc("Record emitted AMDGPU annealing region permutations"));

static cl::opt<unsigned> AnnealingMutationWalkDepth(
    "amdgpu-annealing-mutation-walk-depth", cl::Hidden, cl::init(0),
    cl::desc("Generate an exact-depth legal relocation walk for an external "
             "empirical schedule endpoint"));

static cl::opt<unsigned> AnnealingMutationWalkRegion(
    "amdgpu-annealing-mutation-walk-region", cl::Hidden, cl::init(0),
    cl::desc("Scheduling region mutated by the external empirical walk"));

static cl::opt<std::string> AnnealingRecordMutationPath(
    "amdgpu-annealing-record-mutation-path", cl::Hidden, cl::init(""),
    cl::desc("Record every legal relocation in an external empirical walk"));

namespace llvm {

std::vector<const SUnit *> makeMinRegSchedule(ArrayRef<const SUnit *> TopRoots,
                                              const ScheduleDAG &DAG);

std::vector<const SUnit *> makeGCNILPScheduler(ArrayRef<const SUnit *> BotRoots,
                                               const ScheduleDAG &DAG);
} // namespace llvm

// shim accessors for different order containers
static inline MachineInstr *getMachineInstr(MachineInstr *MI) { return MI; }
static inline MachineInstr *getMachineInstr(const SUnit *SU) {
  return SU->getInstr();
}
static inline MachineInstr *getMachineInstr(const SUnit &SU) {
  return SU.getInstr();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
static void
printRegion(raw_ostream &OS, MachineBasicBlock::iterator Begin,
            MachineBasicBlock::iterator End, const LiveIntervals *LIS,
            unsigned MaxInstNum = std::numeric_limits<unsigned>::max()) {
  auto *BB = Begin->getParent();
  OS << BB->getParent()->getName() << ":" << printMBBReference(*BB) << ' '
     << BB->getName() << ":\n";
  auto I = Begin;
  MaxInstNum = std::max(MaxInstNum, 1u);
  for (; I != End && MaxInstNum; ++I, --MaxInstNum) {
    if (!I->isDebugInstr() && LIS)
      OS << LIS->getInstructionIndex(*I);
    OS << '\t' << *I;
  }
  if (I != End) {
    OS << "\t...\n";
    I = std::prev(End);
    if (!I->isDebugInstr() && LIS)
      OS << LIS->getInstructionIndex(*I);
    OS << '\t' << *I;
  }
  if (End != BB->end()) { // print boundary inst if present
    OS << "----\n";
    if (LIS)
      OS << LIS->getInstructionIndex(*End) << '\t';
    OS << *End;
  }
}

LLVM_DUMP_METHOD
static void printLivenessInfo(raw_ostream &OS,
                              MachineBasicBlock::iterator Begin,
                              MachineBasicBlock::iterator End,
                              const LiveIntervals *LIS) {
  auto *const BB = Begin->getParent();
  const auto &MRI = BB->getParent()->getRegInfo();

  const auto LiveIns = getLiveRegsBefore(*Begin, *LIS);
  OS << "LIn RP: " << print(getRegPressure(MRI, LiveIns));

  const auto BottomMI = End == BB->end() ? std::prev(End) : End;
  const auto LiveOuts = getLiveRegsAfter(*BottomMI, *LIS);
  OS << "LOt RP: " << print(getRegPressure(MRI, LiveOuts));
}

LLVM_DUMP_METHOD
void GCNIterativeScheduler::printRegions(raw_ostream &OS) const {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  for (auto *const R : Regions) {
    OS << "Region to schedule ";
    printRegion(OS, R->Begin, R->End, LIS, 1);
    printLivenessInfo(OS, R->Begin, R->End, LIS);
    OS << "Max RP: " << print(R->MaxPressure, &ST);
  }
}

LLVM_DUMP_METHOD
void GCNIterativeScheduler::printSchedResult(raw_ostream &OS, const Region *R,
                                             const GCNRegPressure &RP) const {
  OS << "\nAfter scheduling ";
  printRegion(OS, R->Begin, R->End, LIS);
  printSchedRP(OS, R->MaxPressure, RP);
  OS << '\n';
}

LLVM_DUMP_METHOD
void GCNIterativeScheduler::printSchedRP(raw_ostream &OS,
                                         const GCNRegPressure &Before,
                                         const GCNRegPressure &After) const {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  OS << "RP before: " << print(Before, &ST)
     << "RP after:  " << print(After, &ST);
}
#endif

void GCNIterativeScheduler::swapIGLPMutations(const Region &R, bool IsReentry) {
  bool HasIGLPInstrs = false;
  const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(TII);
  for (MachineBasicBlock::iterator I = R.Begin; I != R.End; I++) {
    if (SII->isIGLPMutationOnly(I->getOpcode())) {
      HasIGLPInstrs = true;
      break;
    }
  }

  if (HasIGLPInstrs) {
    SavedMutations.clear();
    SavedMutations.swap(Mutations);
    auto SchedPhase = IsReentry ? AMDGPU::SchedulingPhase::PreRAReentry
                                : AMDGPU::SchedulingPhase::Initial;

    addMutation(createIGroupLPDAGMutation(SchedPhase));
  }
}

// DAG builder helper
class GCNIterativeScheduler::BuildDAG {
  GCNIterativeScheduler &Sch;
  SmallVector<SUnit *, 8> TopRoots;

  SmallVector<SUnit *, 8> BotRoots;

public:
  BuildDAG(const Region &R, GCNIterativeScheduler &_Sch, bool IsReentry = false)
      : Sch(_Sch) {
    auto *BB = R.Begin->getParent();
    Sch.BaseClass::startBlock(BB);
    Sch.BaseClass::enterRegion(BB, R.Begin, R.End, R.NumRegionInstrs);
    Sch.swapIGLPMutations(R, IsReentry);
    Sch.buildSchedGraph(Sch.AA, nullptr, nullptr, nullptr,
                        /*TrackLaneMask*/ true);
    Sch.postProcessDAG();
    Sch.Topo.InitDAGTopologicalSorting();
    Sch.findRootsAndBiasEdges(TopRoots, BotRoots);
  }

  ~BuildDAG() {
    Sch.BaseClass::exitRegion();
    Sch.BaseClass::finishBlock();
  }

  ArrayRef<const SUnit *> getTopRoots() const { return TopRoots; }
  ArrayRef<SUnit *> getBottomRoots() const { return BotRoots; }
};

class GCNIterativeScheduler::OverrideLegacyStrategy {
  GCNIterativeScheduler &Sch;
  Region &Rgn;
  std::unique_ptr<MachineSchedStrategy> SaveSchedImpl;
  GCNRegPressure SaveMaxRP;

public:
  OverrideLegacyStrategy(Region &R, MachineSchedStrategy &OverrideStrategy,
                         GCNIterativeScheduler &_Sch)
      : Sch(_Sch), Rgn(R), SaveSchedImpl(std::move(_Sch.SchedImpl)),
        SaveMaxRP(R.MaxPressure) {
    Sch.SchedImpl.reset(&OverrideStrategy);
    auto *BB = R.Begin->getParent();
    Sch.BaseClass::startBlock(BB);
    Sch.BaseClass::enterRegion(BB, R.Begin, R.End, R.NumRegionInstrs);
  }

  ~OverrideLegacyStrategy() {
    Sch.BaseClass::exitRegion();
    Sch.BaseClass::finishBlock();
    Sch.SchedImpl.release();
    Sch.SchedImpl = std::move(SaveSchedImpl);
  }

  void schedule() {
    assert(Sch.RegionBegin == Rgn.Begin && Sch.RegionEnd == Rgn.End);
    LLVM_DEBUG(dbgs() << "\nScheduling ";
               printRegion(dbgs(), Rgn.Begin, Rgn.End, Sch.LIS, 2));
    Sch.BaseClass::schedule();

    // Unfortunately placeDebugValues incorrectly modifies RegionEnd, restore
    Sch.RegionEnd = Rgn.End;
    // assert(Rgn.End == Sch.RegionEnd);
    Rgn.Begin = Sch.RegionBegin;
    Rgn.MaxPressure.clear();
  }

  void restoreOrder() {
    assert(Sch.RegionBegin == Rgn.Begin && Sch.RegionEnd == Rgn.End);
    // DAG SUnits are stored using original region's order
    // so just use SUnits as the restoring schedule
    Sch.scheduleRegion(Rgn, Sch.SUnits, SaveMaxRP);
  }
};

namespace {

// just a stub to make base class happy
class SchedStrategyStub : public MachineSchedStrategy {
public:
  bool shouldTrackPressure() const override { return false; }
  bool shouldTrackLaneMasks() const override { return false; }
  void initialize(ScheduleDAGMI *DAG) override {}
  SUnit *pickNode(bool &IsTopNode) override { return nullptr; }
  void schedNode(SUnit *SU, bool IsTopNode) override {}
  void releaseTopNode(SUnit *SU) override {}
  void releaseBottomNode(SUnit *SU) override {}
};

} // end anonymous namespace

GCNIterativeScheduler::GCNIterativeScheduler(MachineSchedContext *C,
                                             StrategyKind S,
                                             bool UseCurrentSchedule)
    : BaseClass(C, std::make_unique<SchedStrategyStub>()), Context(C),
      Strategy(S), UseCurrentSchedule(UseCurrentSchedule), UPTracker(*LIS) {}

namespace {
class GCNAnnealingScheduleDAGMILive final : public GCNScheduleDAGMILive {
  MachineSchedContext *Context;

  void finalizeGCNSchedule() override {
    GCNIterativeScheduler Annealer(
        Context, GCNIterativeScheduler::SCHEDULE_ANNEALING,
        /*UseCurrentSchedule=*/true);
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    Annealer.addMutation(
        createLoadClusterDAGMutation(Annealer.TII, Annealer.TRI));
    if (ST.shouldClusterStores())
      Annealer.addMutation(
          createStoreClusterDAGMutation(Annealer.TII, Annealer.TRI));
    Annealer.addMutation(
        createIGroupLPDAGMutation(AMDGPU::SchedulingPhase::Initial));
    Annealer.addMutation(createAMDGPUMacroFusionDAGMutation());
    Annealer.addMutation(createAMDGPUExportClusteringDAGMutation());
    Annealer.addMutation(createAMDGPUBarrierLatencyDAGMutation(&MF));
    Annealer.addMutation(createAMDGPUHazardLatencyDAGMutation(&MF));
    Annealer.annealRegions(getRegions());
  }

public:
  explicit GCNAnnealingScheduleDAGMILive(MachineSchedContext *C)
      : GCNScheduleDAGMILive(
            C, std::make_unique<GCNMaxOccupancySchedStrategy>(C)),
        Context(C) {}
};
} // namespace

ScheduleDAGMILive *llvm::createGCNAnnealingScheduler(MachineSchedContext *C) {
  return new GCNAnnealingScheduleDAGMILive(C);
}

void GCNIterativeScheduler::annealRegions(
    ArrayRef<std::pair<MachineBasicBlock::iterator,
                       MachineBasicBlock::iterator>> ScheduledRegions) {
  assert(Strategy == SCHEDULE_ANNEALING && UseCurrentSchedule);
  MachineBasicBlock *CurrentMBB = nullptr;
  for (const auto &[Begin, End] : ScheduledRegions) {
    if (Begin == End)
      continue;
    MachineBasicBlock *MBB = Begin->getParent();
    if (MBB != CurrentMBB) {
      if (CurrentMBB)
        BaseClass::finishBlock();
      BaseClass::startBlock(MBB);
      CurrentMBB = MBB;
    }
    unsigned NumRegionInstrs = 0;
    for (auto I = Begin; I != End; ++I)
      NumRegionInstrs += !I->isDebugInstr();
    BaseClass::enterRegion(MBB, Begin, End, NumRegionInstrs);
    if (NumRegionInstrs > 2)
      Regions.push_back(new (Alloc.Allocate()) Region{
          Begin, End, NumRegionInstrs, getRegionPressure(Begin, End), nullptr});
    BaseClass::exitRegion();
  }
  if (CurrentMBB)
    BaseClass::finishBlock();
  scheduleAnnealing();
}

// returns max pressure for a region
GCNRegPressure GCNIterativeScheduler::getRegionPressure(
    MachineBasicBlock::iterator Begin, MachineBasicBlock::iterator End) const {
  // For the purpose of pressure tracking bottom inst of the region should
  // be also processed. End is either BB end, BB terminator inst or sched
  // boundary inst.
  auto const BBEnd = Begin->getParent()->end();
  auto const BottomMI = End == BBEnd ? std::prev(End) : End;

  // scheduleRegions walks bottom to top, so its likely we just get next
  // instruction to track
  auto AfterBottomMI = std::next(BottomMI);
  if (AfterBottomMI == BBEnd ||
      &*AfterBottomMI != UPTracker.getLastTrackedMI()) {
    UPTracker.reset(*BottomMI);
  } else {
    assert(UPTracker.isValid());
  }

  for (auto I = BottomMI; I != Begin; --I)
    UPTracker.recede(*I);

  UPTracker.recede(*Begin);

  assert(UPTracker.isValid() || (dbgs() << "Tracked region ",
                                 printRegion(dbgs(), Begin, End, LIS), false));
  return UPTracker.getMaxPressureAndReset();
}

// returns max pressure for a tentative schedule
template <typename Range>
GCNRegPressure
GCNIterativeScheduler::getSchedulePressure(const Region &R,
                                           Range &&Schedule) const {
  auto const BBEnd = R.Begin->getParent()->end();
  GCNUpwardRPTracker RPTracker(*LIS);
  if (R.End != BBEnd) {
    // R.End points to the boundary instruction but the
    // schedule doesn't include it
    RPTracker.reset(*R.End);
    RPTracker.recede(*R.End);
  } else {
    // R.End doesn't point to the boundary instruction
    RPTracker.reset(*std::prev(BBEnd));
  }
  for (auto I = Schedule.end(), B = Schedule.begin(); I != B;) {
    RPTracker.recede(*getMachineInstr(*--I));
  }
  return RPTracker.getMaxPressureAndReset();
}

void GCNIterativeScheduler::enterRegion(MachineBasicBlock *BB, // overridden
                                        MachineBasicBlock::iterator Begin,
                                        MachineBasicBlock::iterator End,
                                        unsigned NumRegionInstrs) {
  BaseClass::enterRegion(BB, Begin, End, NumRegionInstrs);
  if (NumRegionInstrs > 2) {
    Regions.push_back(new (Alloc.Allocate()) Region{
        Begin, End, NumRegionInstrs, getRegionPressure(Begin, End), nullptr});
  }
}

void GCNIterativeScheduler::schedule() { // overridden
  // do nothing
  LLVM_DEBUG(printLivenessInfo(dbgs(), RegionBegin, RegionEnd, LIS);
             if (!Regions.empty() && Regions.back()->Begin == RegionBegin) {
               dbgs() << "Max RP: "
                      << print(Regions.back()->MaxPressure,
                               &MF.getSubtarget<GCNSubtarget>());
             } dbgs()
             << '\n';);
}

void GCNIterativeScheduler::finalizeSchedule() { // overridden
  if (Regions.empty())
    return;
  switch (Strategy) {
  case SCHEDULE_MINREGONLY:
    scheduleMinReg();
    break;
  case SCHEDULE_MINREGFORCED:
    scheduleMinReg(true);
    break;
  case SCHEDULE_LEGACYMAXOCCUPANCY:
    scheduleLegacyMaxOccupancy();
    break;
  case SCHEDULE_ILP:
    scheduleILP(false);
    break;
  case SCHEDULE_ANNEALING:
    scheduleAnnealing();
    break;
  }
}

// Detach schedule from SUnits and interleave it with debug values.
// Returned schedule becomes independent of DAG state.
std::vector<MachineInstr *>
GCNIterativeScheduler::detachSchedule(ScheduleRef Schedule) const {
  std::vector<MachineInstr *> Res;
  Res.reserve(Schedule.size() * 2);

  if (FirstDbgValue)
    Res.push_back(FirstDbgValue);

  const auto DbgB = DbgValues.begin(), DbgE = DbgValues.end();
  for (const auto *SU : Schedule) {
    Res.push_back(SU->getInstr());
    const auto &D = std::find_if(DbgB, DbgE, [SU](decltype(*DbgB) &P) {
      return P.second == SU->getInstr();
    });
    if (D != DbgE)
      Res.push_back(D->first);
  }
  return Res;
}

void GCNIterativeScheduler::setBestSchedule(Region &R, ScheduleRef Schedule,
                                            const GCNRegPressure &MaxRP) {
  R.BestSchedule.reset(new TentativeSchedule{detachSchedule(Schedule), MaxRP});
}

void GCNIterativeScheduler::scheduleBest(Region &R) {
  assert(R.BestSchedule.get() && "No schedule specified");
  scheduleRegion(R, R.BestSchedule->Schedule, R.BestSchedule->MaxPressure);
  R.BestSchedule.reset();
}

// minimal required region scheduler, works for ranges of SUnits*,
// SUnits or MachineIntrs*
template <typename Range>
void GCNIterativeScheduler::scheduleRegion(Region &R, Range &&Schedule,
                                           const GCNRegPressure &MaxRP) {
  assert(RegionBegin == R.Begin && RegionEnd == R.End);
  assert(LIS != nullptr);
#ifndef NDEBUG
  const auto SchedMaxRP = getSchedulePressure(R, Schedule);
#endif
  auto *BB = R.Begin->getParent();
  auto Top = R.Begin;
  for (const auto &I : Schedule) {
    auto MI = getMachineInstr(I);

    MachineBasicBlock::iterator MII = MI->getIterator();
    if (MII != Top) {
      bool NonDebugReordered =
          !MI->isDebugInstr() && skipDebugInstructionsForward(Top, MII) != MII;
      BB->remove(MI);
      BB->insert(Top, MI);
      if (NonDebugReordered)
        LIS->handleMove(*MI, true);
    }
    if (!MI->isDebugInstr()) {
      // Reset read - undef flags and update them later.
      for (auto &Op : MI->all_defs())
        Op.setIsUndef(false);

      RegisterOperands RegOpers;
      RegOpers.collect(*MI, *TRI, MRI, /*ShouldTrackLaneMasks*/ true,
                       /*IgnoreDead*/ false);
      // Adjust liveness and add missing dead+read-undef flags.
      auto SlotIdx = LIS->getInstructionIndex(*MI).getRegSlot();
      RegOpers.adjustLaneLiveness(*LIS, MRI, SlotIdx, MI);
    }
    Top = std::next(MI->getIterator());
  }
  RegionBegin = getMachineInstr(Schedule.front());

  // Schedule consisting of MachineInstr* is considered 'detached'
  // and already interleaved with debug values
  if (!std::is_same_v<decltype(*Schedule.begin()), MachineInstr *>) {
    placeDebugValues();
    // Unfortunately placeDebugValues incorrectly modifies RegionEnd, restore
    // assert(R.End == RegionEnd);
    RegionEnd = R.End;
  }

  R.Begin = RegionBegin;
  R.MaxPressure = MaxRP;

#ifndef NDEBUG
  const auto RegionMaxRP = getRegionPressure(R);
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
#endif
  assert(
      (SchedMaxRP == RegionMaxRP && (MaxRP.empty() || SchedMaxRP == MaxRP)) ||
      (dbgs() << "Max RP mismatch!!!\n"
                 "RP for schedule (calculated): "
              << print(SchedMaxRP, &ST)
              << "RP for schedule (reported): " << print(MaxRP, &ST)
              << "RP after scheduling: " << print(RegionMaxRP, &ST),
       false));
}

// Sort recorded regions by pressure - highest at the front
void GCNIterativeScheduler::sortRegionsByPressure(unsigned TargetOcc) {
  llvm::sort(Regions, [this, TargetOcc](const Region *R1, const Region *R2) {
    return R2->MaxPressure.less(MF, R1->MaxPressure, TargetOcc);
  });
}

///////////////////////////////////////////////////////////////////////////////
// Legacy MaxOccupancy Strategy

// Tries to increase occupancy applying minreg scheduler for a sequence of
// most demanding regions. Obtained schedules are saved as BestSchedule for a
// region.
// TargetOcc is the best achievable occupancy for a kernel.
// Returns better occupancy on success or current occupancy on fail.
// BestSchedules aren't deleted on fail.
unsigned GCNIterativeScheduler::tryMaximizeOccupancy(unsigned TargetOcc) {
  // TODO: assert Regions are sorted descending by pressure
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  const unsigned DynamicVGPRBlockSize =
      MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
  const auto Occ =
      Regions.front()->MaxPressure.getOccupancy(ST, DynamicVGPRBlockSize);
  LLVM_DEBUG(dbgs() << "Trying to improve occupancy, target = " << TargetOcc
                    << ", current = " << Occ << '\n');

  auto NewOcc = TargetOcc;
  for (auto *R : Regions) {
    // Always build the DAG to add mutations
    BuildDAG DAG(*R, *this);

    if (R->MaxPressure.getOccupancy(ST, DynamicVGPRBlockSize) >= NewOcc)
      continue;

    LLVM_DEBUG(printRegion(dbgs(), R->Begin, R->End, LIS, 3);
               printLivenessInfo(dbgs(), R->Begin, R->End, LIS));

    const auto MinSchedule = makeMinRegSchedule(DAG.getTopRoots(), *this);
    const auto MaxRP = getSchedulePressure(*R, MinSchedule);
    LLVM_DEBUG(dbgs() << "Occupancy improvement attempt:\n";
               printSchedRP(dbgs(), R->MaxPressure, MaxRP));

    NewOcc = std::min(NewOcc, MaxRP.getOccupancy(ST, DynamicVGPRBlockSize));
    if (NewOcc <= Occ)
      break;

    setBestSchedule(*R, MinSchedule, MaxRP);
  }
  LLVM_DEBUG(dbgs() << "New occupancy = " << NewOcc
                    << ", prev occupancy = " << Occ << '\n');
  if (NewOcc > Occ) {
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    MFI->increaseOccupancy(MF, NewOcc);
  }

  return std::max(NewOcc, Occ);
}

void GCNIterativeScheduler::scheduleLegacyMaxOccupancy(
    bool TryMaximizeOccupancy) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  auto TgtOcc = MFI->getMinAllowedOccupancy();
  unsigned DynamicVGPRBlockSize = MFI->getDynamicVGPRBlockSize();

  sortRegionsByPressure(TgtOcc);
  auto Occ =
      Regions.front()->MaxPressure.getOccupancy(ST, DynamicVGPRBlockSize);

  bool IsReentry = false;
  if (TryMaximizeOccupancy && Occ < TgtOcc) {
    Occ = tryMaximizeOccupancy(TgtOcc);
    IsReentry = true;
  }

  // This is really weird but for some magic scheduling regions twice
  // gives performance improvement
  const int NumPasses = Occ < TgtOcc ? 2 : 1;

  TgtOcc = std::min(Occ, TgtOcc);
  LLVM_DEBUG(dbgs() << "Scheduling using default scheduler, "
                       "target occupancy = "
                    << TgtOcc << '\n');
  GCNMaxOccupancySchedStrategy LStrgy(Context, /*IsLegacyScheduler=*/true);
  unsigned FinalOccupancy = std::min(Occ, MFI->getOccupancy());

  for (int I = 0; I < NumPasses; ++I) {
    // running first pass with TargetOccupancy = 0 mimics previous scheduling
    // approach and is a performance magic
    LStrgy.setTargetOccupancy(I == 0 ? 0 : TgtOcc);
    for (auto *R : Regions) {
      OverrideLegacyStrategy Ovr(*R, LStrgy, *this);
      IsReentry |= I > 0;
      swapIGLPMutations(*R, IsReentry);
      Ovr.schedule();
      const auto RP = getRegionPressure(*R);
      LLVM_DEBUG(printSchedRP(dbgs(), R->MaxPressure, RP));

      if (RP.getOccupancy(ST, DynamicVGPRBlockSize) < TgtOcc) {
        LLVM_DEBUG(dbgs() << "Didn't fit into target occupancy O" << TgtOcc);
        if (R->BestSchedule.get() && R->BestSchedule->MaxPressure.getOccupancy(
                                         ST, DynamicVGPRBlockSize) >= TgtOcc) {
          LLVM_DEBUG(dbgs() << ", scheduling minimal register\n");
          scheduleBest(*R);
        } else {
          LLVM_DEBUG(dbgs() << ", restoring\n");
          Ovr.restoreOrder();
          assert(R->MaxPressure.getOccupancy(ST, DynamicVGPRBlockSize) >=
                 TgtOcc);
        }
      }
      FinalOccupancy =
          std::min(FinalOccupancy, RP.getOccupancy(ST, DynamicVGPRBlockSize));
    }
  }
  MFI->limitOccupancy(FinalOccupancy);
}

///////////////////////////////////////////////////////////////////////////////
// Minimal Register Strategy

void GCNIterativeScheduler::scheduleMinReg(bool force) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const auto TgtOcc = MFI->getOccupancy();
  sortRegionsByPressure(TgtOcc);

  auto MaxPressure = Regions.front()->MaxPressure;
  for (auto *R : Regions) {
    if (!force && R->MaxPressure.less(MF, MaxPressure, TgtOcc))
      break;

    BuildDAG DAG(*R, *this);
    const auto MinSchedule = makeMinRegSchedule(DAG.getTopRoots(), *this);

    const auto RP = getSchedulePressure(*R, MinSchedule);
    LLVM_DEBUG(if (R->MaxPressure.less(MF, RP, TgtOcc)) {
      dbgs() << "\nWarning: Pressure becomes worse after minreg!";
      printSchedRP(dbgs(), R->MaxPressure, RP);
    });

    if (!force && MaxPressure.less(MF, RP, TgtOcc))
      break;

    scheduleRegion(*R, MinSchedule, RP);
    LLVM_DEBUG(printSchedResult(dbgs(), R, RP));

    MaxPressure = RP;
  }
}

///////////////////////////////////////////////////////////////////////////////
// ILP scheduler port

void GCNIterativeScheduler::scheduleILP(bool TryMaximizeOccupancy) {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  auto TgtOcc = MFI->getMinAllowedOccupancy();
  unsigned DynamicVGPRBlockSize = MFI->getDynamicVGPRBlockSize();

  sortRegionsByPressure(TgtOcc);
  auto Occ =
      Regions.front()->MaxPressure.getOccupancy(ST, DynamicVGPRBlockSize);

  bool IsReentry = false;
  if (TryMaximizeOccupancy && Occ < TgtOcc) {
    Occ = tryMaximizeOccupancy(TgtOcc);
    IsReentry = true;
  }

  TgtOcc = std::min(Occ, TgtOcc);
  LLVM_DEBUG(dbgs() << "Scheduling using default scheduler, "
                       "target occupancy = "
                    << TgtOcc << '\n');

  unsigned FinalOccupancy = std::min(Occ, MFI->getOccupancy());
  for (auto *R : Regions) {
    BuildDAG DAG(*R, *this, IsReentry);
    const auto ILPSchedule = makeGCNILPScheduler(DAG.getBottomRoots(), *this);

    const auto RP = getSchedulePressure(*R, ILPSchedule);
    LLVM_DEBUG(printSchedRP(dbgs(), R->MaxPressure, RP));

    if (RP.getOccupancy(ST, DynamicVGPRBlockSize) < TgtOcc) {
      LLVM_DEBUG(dbgs() << "Didn't fit into target occupancy O" << TgtOcc);
      if (R->BestSchedule.get() && R->BestSchedule->MaxPressure.getOccupancy(
                                       ST, DynamicVGPRBlockSize) >= TgtOcc) {
        LLVM_DEBUG(dbgs() << ", scheduling minimal register\n");
        scheduleBest(*R);
      }
    } else {
      scheduleRegion(*R, ILPSchedule, RP);
      LLVM_DEBUG(printSchedResult(dbgs(), R, RP));
      FinalOccupancy =
          std::min(FinalOccupancy, RP.getOccupancy(ST, DynamicVGPRBlockSize));
    }
  }
  MFI->limitOccupancy(FinalOccupancy);
}

///////////////////////////////////////////////////////////////////////////////
// Experimental simulated annealing scheduler

void GCNIterativeScheduler::scheduleAnnealing() {
  const auto &ST = MF.getSubtarget<GCNSubtarget>();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const unsigned DynamicVGPRBlockSize = MFI->getDynamicVGPRBlockSize();
  unsigned FinalOccupancy = MFI->getOccupancy();

  DenseMap<unsigned, SmallVector<unsigned>> ReplayOrders;
  if (!AnnealingReplaySchedule.empty()) {
    auto Buffer = MemoryBuffer::getFile(AnnealingReplaySchedule);
    if (Buffer) {
      SmallVector<StringRef> Lines;
      Buffer.get()->getBuffer().split(Lines, '\n');
      const std::string Prefix = (MF.getName() + "\t").str();
      for (StringRef Line : Lines) {
        if (!Line.consume_front(Prefix))
          continue;
        auto [RegionText, OrderText] = Line.split('\t');
        unsigned RegionIndex;
        if (RegionText.getAsInteger(10, RegionIndex))
          continue;
        SmallVector<StringRef> Ordinals;
        OrderText.split(Ordinals, ',', /*MaxSplit=*/-1,
                        /*KeepEmpty=*/false);
        SmallVector<unsigned> Order;
        for (StringRef Ordinal : Ordinals) {
          unsigned Value;
          if (Ordinal.getAsInteger(10, Value)) {
            Order.clear();
            break;
          }
          Order.push_back(Value);
        }
        if (!Order.empty())
          ReplayOrders[RegionIndex] = std::move(Order);
      }
    }
  }

  DenseMap<Region *, SmallVector<MachineInstr *, 0>> InputOrders;
  DenseMap<Region *, SmallVector<MachineInstr *, 0>> BaselineOrders;
  if (AnnealingOracle && AnnealingOracleInputStart) {
    for (Region *R : Regions)
      for (auto I = R->Begin; I != R->End; ++I)
        InputOrders[R].push_back(&*I);
  }

  if (!UseCurrentSchedule) {
    // Legacy entry point: materialize an occupancy-oriented schedule before
    // annealing. The production entry point supplies the exact completed
    // GCNScheduleDAGMILive order instead.
    GCNMaxOccupancySchedStrategy BaselineStrategy(Context);
    BaselineStrategy.setTargetOccupancy(MFI->getMinAllowedOccupancy());
    for (Region *R : Regions) {
      OverrideLegacyStrategy Override(*R, BaselineStrategy, *this);
      swapIGLPMutations(*R, /*IsReentry=*/false);
      Override.schedule();
      R->MaxPressure = getRegionPressure(*R);
    }
  }
  unsigned RegionIndex = 0;
  for (Region *R : Regions) {
    for (auto I = R->Begin; I != R->End; ++I)
      BaselineOrders[R].push_back(&*I);
  }

  for (Region *R : Regions) {
    BuildDAG DAG(*R, *this);
    DenseMap<const MachineInstr *, const SUnit *> Nodes;
    for (const SUnit &SU : SUnits)
      Nodes[SU.getInstr()] = &SU;
    std::vector<const SUnit *> Original;
    Original.reserve(SUnits.size());
    for (MachineInstr *MI : BaselineOrders[R])
      if (const SUnit *SU = Nodes.lookup(MI))
        Original.push_back(SU);
    if (Original.size() != SUnits.size()) {
      Original.clear();
      for (const SUnit &SU : SUnits)
        Original.push_back(&SU);
    }
    std::vector<const SUnit *> Input;
    if (AnnealingOracle && AnnealingOracleInputStart) {
      for (MachineInstr *MI : InputOrders[R])
        if (const SUnit *SU = Nodes.lookup(MI))
          Input.push_back(SU);
      if (Input.size() != Original.size())
        Input = Original;
    }
    if (Original.size() < 2)
      {
        ++RegionIndex;
        continue;
      }

    std::vector<const SUnit *> ReplayStart;
    if (auto It = ReplayOrders.find(RegionIndex); It != ReplayOrders.end()) {
      BitVector Seen(Original.size());
      for (unsigned Ordinal : It->second) {
        if (Ordinal >= Original.size() || Seen.test(Ordinal)) {
          ReplayStart.clear();
          break;
        }
        Seen.set(Ordinal);
        ReplayStart.push_back(Original[Ordinal]);
      }
      if (ReplayStart.size() != Original.size())
        ReplayStart.clear();
      if (!ReplayStart.empty()) {
        DenseMap<const SUnit *, unsigned> Positions;
        for (unsigned Pos = 0; Pos != ReplayStart.size(); ++Pos)
          Positions[ReplayStart[Pos]] = Pos;
        bool Legal = true;
        for (unsigned Pos = 0; Pos != ReplayStart.size() && Legal; ++Pos)
          for (const SDep &Pred : ReplayStart[Pos]->Preds)
            if (auto It = Positions.find(Pred.getSUnit());
                It != Positions.end() && It->second >= Pos) {
              Legal = false;
              break;
            }
        if (!Legal)
          ReplayStart.clear();
      }
    }

    const TargetSchedModel *TSM = getSchedModel();
    auto EstimatedCycles = [&](ArrayRef<const SUnit *> Schedule, unsigned Start,
                               unsigned End) {
      DenseMap<unsigned, unsigned> Completion;
      unsigned IssueCycle = 0;
      unsigned IssuedMicroOps = 0;
      unsigned LastIssueCycle = 0;
      const unsigned IssueWidth = std::max(1u, TSM->getIssueWidth());
      for (unsigned Pos = Start; Pos != End; ++Pos) {
        const SUnit *SU = Schedule[Pos];
        unsigned ReadyCycle = 0;
        for (const SDep &Pred : SU->Preds) {
          auto It = Completion.find(Pred.getSUnit()->NodeNum);
          if (It != Completion.end())
            ReadyCycle = std::max(ReadyCycle, It->second + Pred.getLatency());
        }
        if (ReadyCycle > IssueCycle) {
          IssueCycle = ReadyCycle;
          IssuedMicroOps = 0;
        }
        unsigned MicroOps = std::max(1u, TSM->getNumMicroOps(SU->getInstr()));
        if (IssuedMicroOps && IssuedMicroOps + MicroOps > IssueWidth) {
          ++IssueCycle;
          IssuedMicroOps = 0;
        }
        Completion[SU->NodeNum] = IssueCycle;
        LastIssueCycle = IssueCycle;
        IssuedMicroOps += MicroOps;
        if (IssuedMicroOps >= IssueWidth) {
          ++IssueCycle;
          IssuedMicroOps = 0;
        }
      }
      return End == Start ? 0u : LastIssueCycle + 1;
    };

    auto VMEMExposurePenalty = [&](ArrayRef<const SUnit *> Schedule,
                                   unsigned Start, unsigned End) {
      if (!AnnealingVMEMExposureWeight)
        return uint64_t(0);
      DenseMap<const SUnit *, unsigned> Positions;
      for (unsigned Pos = 0; Pos != Schedule.size(); ++Pos)
        Positions[Schedule[Pos]] = Pos;
      uint64_t Penalty = 0;
      for (unsigned Pos = Start; Pos != End; ++Pos) {
        const SUnit *Load = Schedule[Pos];
        if (!SIInstrInfo::isVMEM(*Load->getInstr()))
          continue;
        unsigned EarliestUse = Schedule.size();
        for (const SDep &Succ : Load->Succs) {
          if (Succ.getKind() != SDep::Data)
            continue;
          auto It = Positions.find(Succ.getSUnit());
          if (It != Positions.end() && It->second > Pos)
            EarliestUse = std::min(EarliestUse, It->second);
        }
        if (EarliestUse != Schedule.size())
          Penalty += Schedule.size() - (EarliestUse - Pos);
      }
      return Penalty * AnnealingVMEMExposureWeight;
    };

    GCNRegPressure BaselineRP = getSchedulePressure(*R, Original);
    const unsigned BaselineVGPRs = BaselineRP.getVGPRNum(ST.hasGFX90AInsts());
    const unsigned BaselineSGPRs = BaselineRP.getSGPRNum();
    const unsigned TargetOccupancy =
        std::max(1u, MFI->getOccupancy() -
                         std::min<unsigned>(MFI->getOccupancy() - 1,
                                            AnnealingMaxOccupancyLoss));
    auto FullScore = [&](ArrayRef<const SUnit *> Schedule) {
      GCNRegPressure RP = getSchedulePressure(*R, Schedule);
      unsigned Occupancy = RP.getOccupancy(ST, DynamicVGPRBlockSize);
      // Reject occupancy below the configured floor, but otherwise allow
      // pressure to grow so independent memory operations can remain live.
      return uint64_t(TargetOccupancy -
                      std::min(TargetOccupancy, Occupancy)) *
                 1000000000000ULL +
             EstimatedCycles(Schedule, 0, Schedule.size()) +
             VMEMExposurePenalty(Schedule, 0, Schedule.size());
    };

    std::vector<const SUnit *> Best =
        ReplayStart.empty() ? Original : ReplayStart;
    uint64_t BestScore = FullScore(Best);
    SmallVector<unsigned> TileSizes;
    if (AnnealingHierarchical) {
      unsigned Size =
          std::max(2u, std::min<unsigned>(AnnealingHierarchicalMinTile,
                                          Original.size()));
      for (;;) {
        TileSizes.push_back(Size);
        if (Size == Original.size())
          break;
        Size = std::min<unsigned>(Original.size(), Size * 2);
      }
    } else if (AnnealingOracle)
      TileSizes.push_back(Original.size());
    else
      TileSizes.push_back(
          std::max(2u, std::min<unsigned>(AnnealingTileSize, Original.size())));

    SmallVector<ArrayRef<const SUnit *>, 1> Starts;
    if (!ReplayStart.empty())
      Starts.push_back(ReplayStart);
    else
      Starts.push_back(AnnealingOracle && AnnealingOracleInputStart
                           ? ArrayRef<const SUnit *>(Input)
                           : ArrayRef<const SUnit *>(Original));

    auto IsPressureSafe = [&](ArrayRef<const SUnit *> Schedule) {
      GCNRegPressure RP = getSchedulePressure(*R, Schedule);
      if (RP.getOccupancy(ST, DynamicVGPRBlockSize) < TargetOccupancy)
        return false;
      return AnnealingOracleRelaxPressure ||
             (RP.getVGPRNum(ST.hasGFX90AInsts()) <= BaselineVGPRs &&
              RP.getSGPRNum() <= BaselineSGPRs);
    };

    // Start production annealing only from the materialized max-occupancy
    // schedule. This makes a zero-proposal run an exact control for the
    // annealer's starting schedule and attributes any replacement to proposals.
    for (unsigned StartIndex = 0; StartIndex < Starts.size(); ++StartIndex) {
      std::vector<const SUnit *> Current(Starts[StartIndex].begin(),
                                         Starts[StartIndex].end());

      // Keep each stream reproducible and independent of process-global state.
      uint64_t RandomState = (uint64_t(AnnealingSeed) << 32) ^
                             MF.getFunctionNumber() ^
                             (0x9e3779b97f4a7c15ULL * (StartIndex + 1));
      auto Random = [&]() {
        RandomState ^= RandomState << 13;
        RandomState ^= RandomState >> 7;
        RandomState ^= RandomState << 17;
        return RandomState;
      };

      // Hardware-scored experiments need endpoint schedules at an exact
      // structural depth without evaluating or accepting intermediate states.
      // Keep this path inside the existing pre-RA, post-max-occupancy hook:
      // every step is a legal insertion in the current scheduling DAG, while
      // only the final `Best` order reaches the downstream compiler pipeline.
      if (AnnealingMutationWalkDepth &&
          RegionIndex == AnnealingMutationWalkRegion) {
        DenseMap<const SUnit *, unsigned> Ordinals;
        for (unsigned Pos = 0; Pos != Original.size(); ++Pos)
          Ordinals[Original[Pos]] = Pos;
        unsigned Completed = 0;
        unsigned Attempts = 0;
        const unsigned AttemptLimit =
            std::max(1024u, AnnealingMutationWalkDepth * 4096u);
        while (Completed != AnnealingMutationWalkDepth &&
               Attempts++ != AttemptLimit) {
          unsigned From = Random() % Current.size();
          unsigned To = Random() % Current.size();
          if (From == To)
            continue;
          const SUnit *Moved = Current[From];
          bool Blocked = false;
          if (From < To) {
            for (unsigned Pos = From + 1; Pos <= To && !Blocked; ++Pos)
              Blocked = llvm::any_of(Current[Pos]->Preds,
                                     [Moved](const SDep &D) {
                                       return D.getSUnit() == Moved;
                                     });
          } else {
            for (unsigned Pos = To; Pos < From && !Blocked; ++Pos)
              Blocked = llvm::any_of(Moved->Preds, [&](const SDep &D) {
                return D.getSUnit() == Current[Pos];
              });
          }
          if (Blocked)
            continue;
          if (From < To)
            std::rotate(Current.begin() + From, Current.begin() + From + 1,
                        Current.begin() + To + 1);
          else
            std::rotate(Current.begin() + To, Current.begin() + From,
                        Current.begin() + From + 1);
          ++Completed;
          if (!AnnealingRecordMutationPath.empty()) {
            std::error_code EC;
            raw_fd_ostream OS(AnnealingRecordMutationPath.getValue(), EC,
                              sys::fs::CD_OpenAlways, sys::fs::FA_Write,
                              sys::fs::OF_Append);
            if (!EC) {
              OS << MF.getName() << '\t' << RegionIndex << '\t' << Completed
                 << '\t' << Ordinals.lookup(Moved) << '\t' << From << '\t'
                 << To << '\t';
              for (unsigned Pos = 0; Pos != Current.size(); ++Pos) {
                if (Pos)
                  OS << ',';
                OS << Ordinals.lookup(Current[Pos]);
              }
              OS << '\n';
            }
          }
        }
        if (Completed != AnnealingMutationWalkDepth)
          report_fatal_error("AMDGPU empirical mutation walk could not "
                             "construct the requested legal depth");
        Best = Current;
        BestScore = FullScore(Best);
        continue;
      }

      for (unsigned Level = 0; Level < TileSizes.size(); ++Level) {
        unsigned TileSize = TileSizes[Level];
        unsigned BudgetIndex = StartIndex * TileSizes.size() + Level;
        unsigned BudgetParts = Starts.size() * TileSizes.size();
        unsigned Iterations = AnnealingIterations / BudgetParts +
                              (BudgetIndex < AnnealingIterations % BudgetParts);
        if (!Iterations)
          continue;
        unsigned Overlap =
            AnnealingHierarchical
                ? 0
                : std::min<unsigned>(AnnealingTileOverlap, TileSize - 1);
        unsigned Stride = TileSize - Overlap;
        SmallVector<unsigned> TileStarts;
        for (unsigned Start = 0; Start + TileSize < Original.size();
             Start += Stride)
          TileStarts.push_back(Start);
        TileStarts.push_back(Original.size() - TileSize);
        llvm::sort(TileStarts);
        TileStarts.erase(std::unique(TileStarts.begin(), TileStarts.end()),
                         TileStarts.end());

        // Child tiles establish local schedules. Each larger level can move
        // instructions across child seams and scores dependencies spanning the
        // entire parent, culminating in a whole-region reconciliation level.
        for (unsigned Sweep = 0; Sweep < AnnealingSweeps; ++Sweep) {
          for (unsigned TileIndex = 0; TileIndex < TileStarts.size();
               ++TileIndex) {
            unsigned OrderedIndex =
                Sweep & 1 ? TileStarts.size() - TileIndex - 1 : TileIndex;
            unsigned Start = TileStarts[OrderedIndex];
            unsigned End = Start + TileSize;

            // Derive the live set at this tile's lower boundary once.
            // Instructions never move across the active tile boundary, so it
            // remains invariant for every proposal in this tile.
            GCNUpwardRPTracker BoundaryTracker(*LIS);
            auto BBEnd = R->Begin->getParent()->end();
            if (R->End != BBEnd) {
              BoundaryTracker.reset(*R->End);
              BoundaryTracker.recede(*R->End);
            } else {
              BoundaryTracker.reset(*std::prev(BBEnd));
            }
            for (unsigned Pos = Current.size(); Pos != End;)
              BoundaryTracker.recede(*Current[--Pos]->getInstr());
            GCNRPTracker::LiveRegSet TileLiveOut =
                BoundaryTracker.getLiveRegs();

            auto TileScore = [&](ArrayRef<const SUnit *> Schedule) {
              GCNUpwardRPTracker RPTracker(*LIS);
              RPTracker.reset(MRI, TileLiveOut);
              for (unsigned Pos = End; Pos != Start;)
                RPTracker.recede(*Schedule[--Pos]->getInstr());
              GCNRegPressure RP = RPTracker.getMaxPressureAndReset();
              unsigned Occupancy = RP.getOccupancy(ST, DynamicVGPRBlockSize);

              return uint64_t(TargetOccupancy -
                              std::min(TargetOccupancy, Occupancy)) *
                         1000000ULL +
                     EstimatedCycles(Schedule, Start, End) +
                     VMEMExposurePenalty(Schedule, Start, End);
            };

            uint64_t LocalScore = TileScore(Current);
            uint64_t BestLocalScore = LocalScore;
            std::vector<const SUnit *> BestLocal = Current;
            const double InitialTemperature = std::max(
                0.000001, double(LocalScore) * AnnealingTemperatureScale);
            unsigned StaleIterations = 0;
            for (unsigned I = 0; I < Iterations; ++I) {
              if (!AnnealingOracle && AnnealingStallLimit &&
                  StaleIterations >= AnnealingStallLimit)
                break;
              unsigned From = Start + Random() % TileSize;
              unsigned To = From;
              bool Insertion = false;
              bool PreferVMEM = false;
              bool PreferConsumer = false;
              unsigned MoveKind = 0;
              if (AnnealingOracle) {
                Insertion = Random() & 1;
                PreferVMEM = AnnealingOracleVMEMMoves &&
                             (!AnnealingOracleRelaxedMoves || (Random() & 1));
              } else {
                // Adjacent swap, arbitrary insertion, VMEM-earlier insertion,
                // consumer-later insertion, or bounded window rotation.
                MoveKind = Random() % 5;
                Insertion = MoveKind != 0;
                PreferVMEM = MoveKind == 2;
                PreferConsumer = MoveKind == 3;
              }
              if (PreferVMEM) {
                Insertion = true;
                unsigned Attempts = 0;
                while (!SIInstrInfo::isVMEM(*Current[From]->getInstr()) &&
                       ++Attempts != 16)
                  From = Start + Random() % TileSize;
                if (!SIInstrInfo::isVMEM(*Current[From]->getInstr()) ||
                    From == Start)
                  continue;
              }
              if (PreferConsumer) {
                unsigned Attempts = 0;
                auto IsVMEMConsumer = [&](unsigned Pos) {
                  return llvm::any_of(Current[Pos]->Preds, [](const SDep &D) {
                    return SIInstrInfo::isVMEM(*D.getSUnit()->getInstr());
                  });
                };
                while (!IsVMEMConsumer(From) && ++Attempts != 16)
                  From = Start + Random() % TileSize;
                if (!IsVMEMConsumer(From) || From + 1 == End)
                  continue;
              }
              if (Insertion) {
                if (PreferVMEM &&
                    (!AnnealingOracle || !AnnealingOracleRelaxedMoves))
                  To = Start + Random() % (From - Start + 1);
                else if (PreferConsumer)
                  To = From + 1 + Random() % (End - From - 1);
                else if (!AnnealingOracle && MoveKind == 4) {
                  if (TileSize <= 2)
                    continue;
                  unsigned Distance =
                      2 + Random() % std::min(7u, TileSize - 2);
                  To = Random() & 1 ? std::min(End - 1, From + Distance)
                                    : From - std::min(From - Start, Distance);
                } else
                  To = Start + Random() % TileSize;
                if (From == To)
                  continue;
                const SUnit *Moved = Current[From];
                bool Blocked = false;
                if (From < To) {
                  for (unsigned Pos = From + 1; Pos <= To && !Blocked; ++Pos)
                    Blocked = llvm::any_of(Current[Pos]->Preds,
                                           [Moved](const SDep &D) {
                                             return D.getSUnit() == Moved;
                                           });
                } else {
                  for (unsigned Pos = To; Pos < From && !Blocked; ++Pos)
                    Blocked = llvm::any_of(Moved->Preds, [&](const SDep &D) {
                      return D.getSUnit() == Current[Pos];
                    });
                }
                if (Blocked) {
                  ++StaleIterations;
                  continue;
                }
                if (From < To)
                  std::rotate(Current.begin() + From,
                              Current.begin() + From + 1,
                              Current.begin() + To + 1);
                else
                  std::rotate(Current.begin() + To, Current.begin() + From,
                              Current.begin() + From + 1);
              } else {
                From = Start + Random() % (TileSize - 1);
                To = From + 1;
                const SUnit *A = Current[From];
                const SUnit *B = Current[To];
                // Adjacent nodes may be exchanged exactly when there is no edge
                // between them; adjacency rules out a longer dependency path.
                if (llvm::any_of(B->Preds, [A](const SDep &D) {
                      return D.getSUnit() == A;
                    })) {
                  ++StaleIterations;
                  continue;
                }
                std::swap(Current[From], Current[To]);
              }
              uint64_t NewScore = TileScore(Current);
              unsigned CoolingPosition = AnnealingOracle ? I % 1000 : I;
              unsigned CoolingLength = AnnealingOracle ? 1000 : Iterations;
              double Temperature =
                  InitialTemperature *
                  (1.0 - double(CoolingPosition) / CoolingLength);
              bool Accept = NewScore <= LocalScore;
              if (!Accept && Temperature > 0.0) {
                double Probability =
                    std::exp(-double(NewScore - LocalScore) / Temperature);
                Accept = double(Random() >> 11) * 0x1.0p-53 < Probability;
              }
              if (Accept) {
                LocalScore = NewScore;
                if (NewScore < BestLocalScore) {
                  BestLocalScore = NewScore;
                  BestLocal = Current;
                  StaleIterations = 0;
                } else
                  ++StaleIterations;
              } else {
                if (!Insertion)
                  std::swap(Current[From], Current[To]);
                else if (From < To)
                  std::rotate(Current.begin() + From, Current.begin() + To,
                              Current.begin() + To + 1);
                else
                  std::rotate(Current.begin() + To, Current.begin() + To + 1,
                              Current.begin() + From + 1);
                ++StaleIterations;
              }
            }
            if (!AnnealingOracle)
              Current = std::move(BestLocal);
          }
          uint64_t NewFullScore = FullScore(Current);
          if (AnnealingOracle ? IsPressureSafe(Current)
                              : NewFullScore < BestScore) {
            Best = Current;
            BestScore = NewFullScore;
          }
        }
      }
    }

    GCNRegPressure RP = getSchedulePressure(*R, Best);
    if (!AnnealingRecordSchedule.empty()) {
      std::error_code EC;
      raw_fd_ostream OS(AnnealingRecordSchedule.getValue(), EC,
                        sys::fs::CD_OpenAlways, sys::fs::FA_Write,
                        sys::fs::OF_Append);
      if (!EC) {
        DenseMap<const SUnit *, unsigned> Ordinals;
        for (unsigned I = 0; I != Original.size(); ++I)
          Ordinals[Original[I]] = I;
        OS << MF.getName() << '\t' << RegionIndex << '\t';
        for (unsigned I = 0; I != Best.size(); ++I) {
          if (I)
            OS << ',';
          OS << Ordinals.lookup(Best[I]);
        }
        OS << '\n';
      }
    }
    scheduleRegion(*R, Best, RP);
    FinalOccupancy =
        std::min(FinalOccupancy, RP.getOccupancy(ST, DynamicVGPRBlockSize));
    ++RegionIndex;
  }
  MFI->limitOccupancy(FinalOccupancy);
}
