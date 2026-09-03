//===- GCNCompleteScheduleOptimizerTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNCompleteScheduleOptimizer.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineSchedSearch.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <array>

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

#include "MFCommon.inc"

class TestGCNCompleteScheduleOptimizer final
    : public GCNCompleteScheduleOptimizer {
  SmallVector<unsigned> ProposedResult;

public:
  explicit TestGCNCompleteScheduleOptimizer(ArrayRef<unsigned> Result)
      : ProposedResult(Result) {}

protected:
  bool optimizeGCNCompleteSchedule(const MachineSchedSearchRegion &,
                                   ArrayRef<unsigned>,
                                   SmallVectorImpl<unsigned> &Result) override {
    Result.assign(ProposedResult.begin(), ProposedResult.end());
    return true;
  }
};

struct TestRegion {
  LLVMContext Context;
  Module M{"GCNCompleteScheduleOptimizerTest", Context};
  std::unique_ptr<MachineFunction> MF = createMachineFunction(Context, M);
  MCInstrDesc Desc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<MachineInstr *, 3> Instructions;
  std::array<SUnit, 3> Nodes;

  TestRegion() {
    for (auto [Ordinal, MI] : enumerate(Instructions)) {
      MI = MF->CreateMachineInstr(Desc, DebugLoc());
      Nodes[Ordinal].setInstr(MI);
      Nodes[Ordinal].NodeNum = Ordinal;
    }
  }

  ~TestRegion() {
    for (MachineInstr *MI : Instructions)
      MF->deleteMachineInstr(MI);
  }
};

} // namespace

TEST(GCNCompleteScheduleOptimizer, UsesStableRegionOrdinals) {
  TestRegion Test;
  MachineSchedSearchRegion Region(Test.Nodes);
  TestGCNCompleteScheduleOptimizer Optimizer({0, 1, 2});
  SmallVector<unsigned> Result;

  ASSERT_TRUE(Optimizer.optimizeCompleteSchedule(Region, {2, 0, 1}, Result));
  EXPECT_EQ(Result, (SmallVector<unsigned, 3>{0, 1, 2}));
  EXPECT_TRUE(Optimizer.validateCompleteSchedule(Region, Result));
}

TEST(GCNCompleteScheduleOptimizer, PreservesLocalVirtualRegisterOrder) {
  TestRegion Test;
  Register VReg = Register::index2VirtReg(1);
  Test.Instructions[0]->addOperand(
      *Test.MF, MachineOperand::CreateReg(VReg, /*isDef=*/true));
  Test.Instructions[2]->addOperand(
      *Test.MF, MachineOperand::CreateReg(VReg, /*isDef=*/false));
  MachineSchedSearchRegion Region(Test.Nodes);

  TestGCNCompleteScheduleOptimizer SafeOptimizer({1, 0, 2});
  SmallVector<unsigned> SafeResult;
  ASSERT_TRUE(
      SafeOptimizer.optimizeCompleteSchedule(Region, {0, 1, 2}, SafeResult));
  EXPECT_TRUE(SafeOptimizer.validateCompleteSchedule(Region, SafeResult));

  TestGCNCompleteScheduleOptimizer UnsafeOptimizer({2, 1, 0});
  SmallVector<unsigned> UnsafeResult;
  ASSERT_TRUE(UnsafeOptimizer.optimizeCompleteSchedule(Region, {0, 1, 2},
                                                       UnsafeResult));
  EXPECT_FALSE(UnsafeOptimizer.validateCompleteSchedule(Region, UnsafeResult));
}
