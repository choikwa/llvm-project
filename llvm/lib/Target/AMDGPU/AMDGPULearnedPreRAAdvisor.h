//===- AMDGPULearnedPreRAAdvisor.h - Native learned scheduler advisor -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDPREADVISOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDPREADVISOR_H

#include "AMDGPULearnedModelFormat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <array>
#include <memory>

namespace llvm::AMDGPU {

struct Relocation {
  unsigned SUnitIdx = 0;
  unsigned OldPos = 0;
  unsigned NewPos = 0;
  unsigned Distance = 0;
  unsigned Category = 0;
  unsigned Depth = 1;
};

struct StatePrediction {
  double Endpoint = 0.0;
  double Support = 0.0;
};

struct ActionPrediction {
  double Immediate = 0.0;
  double Positive = 0.0;
  double Accepted = 0.0;
  double Elite = 0.0;
  std::array<double, 3> Consequence{};
  double Support = 0.0;
};

class PreRASchedSearchAdvisor {
public:
  virtual ~PreRASchedSearchAdvisor() = default;
  virtual Expected<StatePrediction> evaluateState(ArrayRef<float> Features) = 0;
  virtual Expected<ActionPrediction>
  evaluateAction(ArrayRef<float> Features) = 0;
};

class NeuralPreRASchedSearchAdvisor final : public PreRASchedSearchAdvisor {
  std::shared_ptr<const LearnedPreRAModel> Model;

public:
  explicit NeuralPreRASchedSearchAdvisor(
      std::shared_ptr<const LearnedPreRAModel> Model)
      : Model(std::move(Model)) {}

  Expected<StatePrediction> evaluateState(ArrayRef<float> Features) override;
  Expected<ActionPrediction> evaluateAction(ArrayRef<float> Features) override;
};

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPULEARNEDPREADVISOR_H
