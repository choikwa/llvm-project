//===- AMDGPULearnedModelFormat.h - Learned model loader -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDMODELFORMAT_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDMODELFORMAT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <array>
#include <memory>

namespace llvm::AMDGPU {

/// Immutable external C33 shared neural model used by pre-RA search.
class LearnedPreRAModel {
private:
  std::unique_ptr<MemoryBuffer> Storage;
  std::array<uint8_t, 32> SourceSHA{};
  std::array<uint8_t, 32> SchemaSHA{};
  ArrayRef<float> NeuralActionMean;
  ArrayRef<float> NeuralActionStd;
  ArrayRef<float> NeuralStateMean;
  ArrayRef<float> NeuralStateStd;
  ArrayRef<float> NeuralTargetMean;
  ArrayRef<float> NeuralTargetStd;
  ArrayRef<float> NeuralWeights;

  LearnedPreRAModel() = default;

public:
  static Expected<std::shared_ptr<const LearnedPreRAModel>>
  load(StringRef Path);

  ArrayRef<uint8_t> getSourceSHA() const { return SourceSHA; }
  ArrayRef<uint8_t> getSchemaSHA() const { return SchemaSHA; }
  ArrayRef<float> getNeuralActionMean() const { return NeuralActionMean; }
  ArrayRef<float> getNeuralActionStd() const { return NeuralActionStd; }
  ArrayRef<float> getNeuralStateMean() const { return NeuralStateMean; }
  ArrayRef<float> getNeuralStateStd() const { return NeuralStateStd; }
  ArrayRef<float> getNeuralTargetMean() const { return NeuralTargetMean; }
  ArrayRef<float> getNeuralTargetStd() const { return NeuralTargetStd; }
  ArrayRef<float> getNeuralWeights() const { return NeuralWeights; }
  size_t getMappedBytes() const { return Storage->getBufferSize(); }
};

/// Cache the validated immutable blob once per compiler process/path.
Expected<std::shared_ptr<const LearnedPreRAModel>>
getCachedLearnedPreRAModel(StringRef Path);

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDMODELFORMAT_H
