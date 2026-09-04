//===- AArch64ScheduleTraining.h - schedule search endpoint -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64SCHEDULETRAINING_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64SCHEDULETRAINING_H

#include <memory>

namespace llvm {
class MachineSchedCompleteScheduleOptimizer;
class MachineSchedContext;

namespace AArch64 {
bool isScheduleTrainingEnabled();
std::unique_ptr<MachineSchedCompleteScheduleOptimizer>
createScheduleTrainingOptimizer(MachineSchedContext *C);
} // namespace AArch64
} // namespace llvm

#endif
