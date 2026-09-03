//===- GCNCompleteScheduleOptimizer.h - GCN schedule search ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Common AMDGPU support for complete-schedule optimizers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNCOMPLETESCHEDULEOPTIMIZER_H
#define LLVM_LIB_TARGET_AMDGPU_GCNCOMPLETESCHEDULEOPTIMIZER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {
class MachineSchedContext;
class MachineSchedSearchRegion;
class ScheduleDAGInstrs;

namespace AMDGPU {

/// Target-side base for GCN optimizers that transform complete schedules.
///
/// Generic CodeGen validates permutations and strong scheduling dependencies.
/// This class adds AMDGPU's local virtual-register ordering guard.
class GCNCompleteScheduleOptimizer
    : public MachineSchedCompleteScheduleOptimizer {
  SmallVector<unsigned> Founder;

protected:
  virtual bool
  optimizeGCNCompleteSchedule(const MachineSchedSearchRegion &Region,
                              ArrayRef<unsigned> Founder,
                              SmallVectorImpl<unsigned> &Result) = 0;

  virtual bool
  validateGCNCompleteSchedule(const MachineSchedSearchRegion &Region,
                              ArrayRef<unsigned> Founder,
                              ArrayRef<unsigned> Result) {
    return true;
  }

public:
  bool optimizeCompleteSchedule(const MachineSchedSearchRegion &Region,
                                ArrayRef<unsigned> Founder,
                                SmallVectorImpl<unsigned> &Result) final;

  bool validateCompleteSchedule(const MachineSchedSearchRegion &Region,
                                ArrayRef<unsigned> Result) final;
};

/// Create a scheduler that runs the production GCN scheduling stages and then
/// replays the resulting regions through Optimizer.
ScheduleDAGInstrs *createGCNPostScheduleOptimizerScheduler(
    MachineSchedContext *C,
    std::unique_ptr<MachineSchedCompleteScheduleOptimizer> Optimizer);

/// Return whether the experimental neural post-scheduler is enabled.
bool isGCNNeuralScheduleEnabled();

/// Create the production GCN scheduler followed by neural complete-schedule
/// optimization.
ScheduleDAGInstrs *createGCNNeuralPostScheduler(MachineSchedContext *C);

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNCOMPLETESCHEDULEOPTIMIZER_H
