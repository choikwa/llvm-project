//===- AMDGPULearnedFeatures.h - Learned pre-RA features -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDFEATURES_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDFEATURES_H

#include "AMDGPULearnedPreRAAdvisor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace llvm {
class MachineRegisterInfo;
class MachineSchedSearchRegion;
class SUnit;
class TargetInstrInfo;
class TargetRegisterInfo;

namespace AMDGPU {

inline constexpr StringLiteral LearnedPreRAFeatureSchemaSHA256 =
    "f346529d24c027c55709e9dac6744d561cfa7d46ee30a727fe5ddc327515ee62";
inline constexpr unsigned LearnedPreRAStateFeatureCount = 22;
inline constexpr unsigned LearnedPreRAActionFeatureCount = 55;

/// Region-local, immutable adapter from LLVM's ScheduleDAG state to the exact
/// 22-state/55-action feature schema used by Learned PreRA Scheduler v0.
class LearnedPreRARegionFeatures {
public:
  struct Pair {
    unsigned First;
    unsigned Second;
  };
  struct Operand {
    unsigned Node;
    unsigned VReg;
    bool IsDef;
    bool IsUse;
    float Weight;
  };

private:
  unsigned NodeCount = 0;
  float AllocatedVGPRs = 0.0f;
  SmallVector<uint8_t> Categories;
  SmallVector<SmallVector<unsigned, 4>> Predecessors;
  SmallVector<SmallVector<unsigned, 4>> Successors;
  SmallVector<Pair> M1Pairs;
  SmallVector<Pair> M2Pairs;
  SmallVector<Pair> GenericPairs;
  SmallVector<float> GenericLatencies;
  SmallVector<Operand> Operands;
  unsigned VRegCount = 0;
  SmallVector<unsigned> BaselinePositions;

  static uint8_t classify(StringRef Opcode);

public:
  LearnedPreRARegionFeatures(const MachineSchedSearchRegion &Region,
                             ArrayRef<unsigned> Founder,
                             const TargetInstrInfo &TII,
                             const TargetRegisterInfo &TRI,
                             const MachineRegisterInfo &MRI,
                             float AllocatedVGPRs);

  unsigned size() const { return NodeCount; }
  unsigned getCategory(unsigned Node) const { return Categories[Node]; }
  ArrayRef<unsigned> getPredecessors(unsigned Node) const {
    return Predecessors[Node];
  }
  ArrayRef<unsigned> getSuccessors(unsigned Node) const {
    return Successors[Node];
  }

  bool isPermutation(ArrayRef<unsigned> Order) const;
  bool isLegal(ArrayRef<unsigned> Order) const;
  void positions(ArrayRef<unsigned> Order,
                 SmallVectorImpl<unsigned> &Result) const;
  void stateFeatures(ArrayRef<unsigned> Order,
                     SmallVectorImpl<float> &Result) const;
  void actionFeatures(ArrayRef<unsigned> Parent, ArrayRef<unsigned> Child,
                      const Relocation &Action, ArrayRef<float> ParentFeatures,
                      ArrayRef<float> ChildFeatures,
                      SmallVectorImpl<float> &Result) const;
};

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDFEATURES_H
