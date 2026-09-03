//===- AMDGPULearnedPreRAAdvisorTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedPreRAAdvisor.h"
#include "AMDGPULearnedModelFormat.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstring>

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

constexpr uint64_t HeaderSize = 256;
constexpr uint64_t ParameterCount = 125192;
constexpr uint8_t FeatureSchemaSHA[32] = {
    0xf3, 0x46, 0x52, 0x9d, 0x24, 0xc0, 0x27, 0xc5, 0x57, 0x09, 0xe9,
    0xda, 0xc6, 0x74, 0x4d, 0x56, 0x1c, 0xfa, 0x7d, 0x46, 0xee, 0x30,
    0xa7, 0x27, 0xfe, 0x5d, 0xdc, 0x32, 0x75, 0x15, 0xee, 0x62};

template <typename T>
void writeLE(SmallVectorImpl<char> &Bytes, uint64_t Offset, T Value) {
  support::endian::write<T>(Bytes.data() + Offset, Value,
                            llvm::endianness::little);
}

} // namespace

TEST(AMDGPULearnedPreRAAdvisor, EvaluatesNativeSharedNeuralModel) {
  constexpr uint64_t NormalizationCount = 2 * (55 + 22 + 5);
  constexpr uint64_t NormalizationOffset = HeaderSize;
  constexpr uint64_t WeightsOffset =
      NormalizationOffset + NormalizationCount * sizeof(float);
  constexpr uint64_t FileSize = WeightsOffset + ParameterCount * sizeof(float);
  SmallVector<char, 0> Bytes(FileSize, 0);
  std::memcpy(Bytes.data(), "AMDPRANN", 8);
  writeLE<uint32_t>(Bytes, 8, 1);
  writeLE<uint32_t>(Bytes, 12, 0x01020304);
  std::memcpy(Bytes.data() + 48, FeatureSchemaSHA, sizeof(FeatureSchemaSHA));
  writeLE<uint32_t>(Bytes, 80, 55);
  writeLE<uint32_t>(Bytes, 84, 22);
  writeLE<uint32_t>(Bytes, 88, 5);
  writeLE<uint32_t>(Bytes, 92, ParameterCount);
  writeLE<uint64_t>(Bytes, 96, NormalizationOffset);
  writeLE<uint64_t>(Bytes, 104, WeightsOffset);
  writeLE<uint64_t>(Bytes, 112, FileSize);

  auto writeNorm = [&](unsigned Index, float Value) {
    writeLE<float>(Bytes, NormalizationOffset + Index * sizeof(float), Value);
  };
  for (unsigned I = 55; I != 2 * 55; ++I)
    writeNorm(I, 1.0f);
  for (unsigned I = 2 * 55 + 22; I != 2 * (55 + 22); ++I)
    writeNorm(I, 1.0f);
  for (unsigned I = 0; I != 5; ++I) {
    writeNorm(2 * (55 + 22) + I, float(I + 1));
    writeNorm(2 * (55 + 22) + 5 + I, 1.0f);
  }

  SmallString<128> Path;
  int FD = -1;
  ASSERT_FALSE(
      sys::fs::createTemporaryFile("amdgpu-prera-nn", "bin", FD, Path));
  {
    raw_fd_ostream OS(FD, /*ShouldClose=*/true);
    OS.write(Bytes.data(), Bytes.size());
  }
  scope_exit Remove([&] { sys::fs::remove(Path); });

  auto Model = LearnedPreRAModel::load(Path);
  ASSERT_TRUE(bool(Model));
  NeuralPreRASchedSearchAdvisor Advisor(*Model);
  SmallVector<float, 22> StateFeatures(22, 0.0f);
  SmallVector<float, 55> ActionFeatures(55, 0.0f);
  auto State = Advisor.evaluateState(StateFeatures);
  auto Action = Advisor.evaluateAction(ActionFeatures);
  ASSERT_TRUE(bool(State));
  ASSERT_TRUE(bool(Action));
  EXPECT_DOUBLE_EQ(State->Endpoint, 5.0);
  EXPECT_DOUBLE_EQ(State->Support, 0.0);
  EXPECT_DOUBLE_EQ(Action->Immediate, 1.0);
  EXPECT_DOUBLE_EQ(Action->Consequence[0], 2.0);
  EXPECT_DOUBLE_EQ(Action->Consequence[1], 3.0);
  EXPECT_DOUBLE_EQ(Action->Consequence[2], 4.0);
  EXPECT_DOUBLE_EQ(Action->Positive, 0.5);
  EXPECT_DOUBLE_EQ(Action->Accepted, 0.5);
  EXPECT_DOUBLE_EQ(Action->Elite, 0.5);
  EXPECT_DOUBLE_EQ(Action->Support, 0.0);
}
