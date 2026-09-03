//===- AMDGPULearnedPreRAScheduler.h - Neural pre-RA search ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDPRERASCHEDULER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDPRERASCHEDULER_H

#include "AMDGPULearnedFeatures.h"
#include "AMDGPULearnedPreRAAdvisor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm::AMDGPU {

struct LearnedPreRABeamEntry {
  SmallVector<unsigned> Order;
  double Endpoint = 0.0;
  double PathScore = 0.0;
  double Support = 0.0;
  double Novelty = 0.0;
  unsigned LineageDepth = 0;
  unsigned RelocationDepth = 0;
};

struct LearnedPreRASearchResult {
  SmallVector<unsigned> SelectedOrder;
  SmallVector<float> SelectedActionFeatures;
  SmallVector<LearnedPreRABeamEntry> ActiveBeam;
  double SelectedEndpoint = 0.0;
  double SelectedPathScore = 0.0;
  double SelectedSupport = 0.0;
  double SelectedNovelty = 0.0;
  double SelectedActionScore = 0.0;
  unsigned SelectedLineageDepth = 0;
  unsigned SelectedRelocationDepth = 0;
  unsigned Generated = 0;
  unsigned Unique = 0;
  unsigned Duplicates = 0;
  unsigned Batches = 0;
};

Expected<LearnedPreRASearchResult> runLearnedPreRASearch(
    const LearnedPreRARegionFeatures &Region, ArrayRef<unsigned> Founder,
    PreRASchedSearchAdvisor &Advisor, unsigned Budget, uint64_t Seed);

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDPRERASCHEDULER_H
