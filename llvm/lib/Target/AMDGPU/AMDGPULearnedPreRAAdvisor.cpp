//===- AMDGPULearnedPreRAAdvisor.cpp - Native learned scheduler advisor ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedPreRAAdvisor.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cmath>

using namespace llvm;
using namespace llvm::AMDGPU;

static double supportDistance(ArrayRef<float> Features, ArrayRef<float> Mean,
                              ArrayRef<float> Std) {
  double Sum = 0.0;
  for (unsigned I = 0; I != Features.size(); ++I) {
    const double Z = (double(Features[I]) - Mean[I]) / Std[I];
    Sum += std::min(Z * Z, 100.0);
  }
  return std::sqrt(Sum / Features.size());
}

static float gelu(float X) {
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  return 0.5f * X * (1.0f + std::erf(X * InvSqrt2));
}

static void dense(ArrayRef<float> Input, ArrayRef<float> Weights,
                  ArrayRef<float> Bias, unsigned OutputDim,
                  SmallVectorImpl<float> &Output, bool ApplyGELU) {
  Output.resize(OutputDim);
  for (unsigned O = 0; O != OutputDim; ++O) {
    float Sum = Bias[O];
    const float *Row = Weights.data() + uint64_t(O) * Input.size();
    for (unsigned I = 0; I != Input.size(); ++I)
      Sum += Input[I] * Row[I];
    Output[O] = ApplyGELU ? gelu(Sum) : Sum;
  }
}

static ArrayRef<float> take(ArrayRef<float> &Values, size_t Count) {
  ArrayRef<float> Result = Values.take_front(Count);
  Values = Values.drop_front(Count);
  return Result;
}

static void denseLayer(ArrayRef<float> Input, ArrayRef<float> &Parameters,
                       unsigned OutputDim, SmallVectorImpl<float> &Output,
                       bool ApplyGELU) {
  ArrayRef<float> Weights =
      take(Parameters, uint64_t(OutputDim) * Input.size());
  ArrayRef<float> Bias = take(Parameters, OutputDim);
  dense(Input, Weights, Bias, OutputDim, Output, ApplyGELU);
}

static Expected<float> evaluateNeuralEndpoint(const LearnedPreRAModel &Model,
                                              ArrayRef<float> Features) {
  if (Features.size() != 22)
    return createStringError(inconvertibleErrorCode(),
                             "invalid neural state feature schema");
  SmallVector<float, 128> Input(Features.size());
  for (unsigned I = 0; I != Features.size(); ++I)
    Input[I] = (Features[I] - Model.getNeuralStateMean()[I]) /
               Model.getNeuralStateStd()[I];
  ArrayRef<float> Parameters = Model.getNeuralWeights();
  // Skip the shared action trunk and its seven heads.
  Parameters = Parameters.drop_front(256 * 55 + 256 + 256 * 256 + 256 +
                                     128 * 256 + 128 + 7 * 128 + 7);
  SmallVector<float, 128> L1;
  denseLayer(Input, Parameters, 128, L1, true);
  SmallVector<float, 64> L2;
  denseLayer(L1, Parameters, 64, L2, true);
  SmallVector<float, 1> Output;
  denseLayer(L2, Parameters, 1, Output, false);
  return Output[0] * Model.getNeuralTargetStd()[4] +
         Model.getNeuralTargetMean()[4];
}

static Expected<ActionPrediction>
evaluateNeuralAction(const LearnedPreRAModel &Model, ArrayRef<float> Features) {
  if (Features.size() != 55)
    return createStringError(inconvertibleErrorCode(),
                             "invalid neural action feature schema");
  SmallVector<float, 256> Input(Features.size());
  for (unsigned I = 0; I != Features.size(); ++I)
    Input[I] = (Features[I] - Model.getNeuralActionMean()[I]) /
               Model.getNeuralActionStd()[I];
  ArrayRef<float> Parameters = Model.getNeuralWeights();
  SmallVector<float, 256> L1;
  denseLayer(Input, Parameters, 256, L1, true);
  SmallVector<float, 256> L2;
  denseLayer(L1, Parameters, 256, L2, true);
  SmallVector<float, 128> Latent;
  denseLayer(L2, Parameters, 128, Latent, true);

  std::array<float, 7> Raw{};
  for (float &Value : Raw) {
    SmallVector<float, 1> Output;
    denseLayer(Latent, Parameters, 1, Output, false);
    Value = Output[0];
  }
  ActionPrediction Prediction;
  Prediction.Immediate =
      Raw[0] * Model.getNeuralTargetStd()[0] + Model.getNeuralTargetMean()[0];
  for (unsigned I = 0; I != 3; ++I)
    Prediction.Consequence[I] = Raw[I + 1] * Model.getNeuralTargetStd()[I + 1] +
                                Model.getNeuralTargetMean()[I + 1];
  auto sigmoid = [](float X) {
    X = std::clamp(X, -30.0f, 30.0f);
    return 1.0 / (1.0 + std::exp(-double(X)));
  };
  Prediction.Positive = sigmoid(Raw[4]);
  Prediction.Accepted = sigmoid(Raw[5]);
  Prediction.Elite = sigmoid(Raw[6]);
  Prediction.Support = supportDistance(Features, Model.getNeuralActionMean(),
                                       Model.getNeuralActionStd());
  return Prediction;
}

Expected<StatePrediction>
NeuralPreRASchedSearchAdvisor::evaluateState(ArrayRef<float> Features) {
  auto Endpoint = evaluateNeuralEndpoint(*Model, Features);
  if (!Endpoint)
    return Endpoint.takeError();
  return StatePrediction{*Endpoint,
                         supportDistance(Features, Model->getNeuralStateMean(),
                                         Model->getNeuralStateStd())};
}

Expected<ActionPrediction>
NeuralPreRASchedSearchAdvisor::evaluateAction(ArrayRef<float> Features) {
  return evaluateNeuralAction(*Model, Features);
}
