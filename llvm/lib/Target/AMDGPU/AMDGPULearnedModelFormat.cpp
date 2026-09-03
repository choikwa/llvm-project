//===- AMDGPULearnedModelFormat.cpp - Learned model loader ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULearnedModelFormat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/IOSandbox.h"
#include <cstring>
#include <mutex>

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {
constexpr char NeuralMagic[] = {'A', 'M', 'D', 'P', 'R', 'A', 'N', 'N'};
constexpr uint32_t Version = 1;
constexpr uint32_t EndianMarker = 0x01020304;
constexpr uint64_t HeaderSize = 256;
constexpr uint64_t NeuralParameterCount = 125192;
constexpr uint8_t FrozenFeatureSchemaSHA[32] = {
    0xf3, 0x46, 0x52, 0x9d, 0x24, 0xc0, 0x27, 0xc5, 0x57, 0x09, 0xe9,
    0xda, 0xc6, 0x74, 0x4d, 0x56, 0x1c, 0xfa, 0x7d, 0x46, 0xee, 0x30,
    0xa7, 0x27, 0xfe, 0x5d, 0xdc, 0x32, 0x75, 0x15, 0xee, 0x62};

template <typename T> Expected<T> readLE(StringRef Buffer, uint64_t Offset) {
  if (Offset > Buffer.size() || sizeof(T) > Buffer.size() - Offset)
    return createStringError(inconvertibleErrorCode(), "truncated model blob");
  return support::endian::read<T, support::unaligned>(Buffer.data() + Offset,
                                                      llvm::endianness::little);
}

Error checkRange(StringRef Buffer, uint64_t Offset, uint64_t Count,
                 uint64_t ElementSize, StringRef What) {
  if (Offset > Buffer.size() || Count > (Buffer.size() - Offset) / ElementSize)
    return createStringError(inconvertibleErrorCode(),
                             formatv("invalid {0} range", What));
  if (Offset % std::min<uint64_t>(ElementSize, 8))
    return createStringError(inconvertibleErrorCode(),
                             formatv("misaligned {0} range", What));
  return Error::success();
}
} // namespace

Expected<std::shared_ptr<const LearnedPreRAModel>>
LearnedPreRAModel::load(StringRef Path) {
  // The model path is an explicit compiler option, analogous to other
  // user-provided backend inputs. Clang enables the I/O sandbox while running
  // code generation, so narrowly disable it for this one read.
  auto BypassSandbox = sys::sandbox::scopedDisable();
  auto BufferOrErr = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  if (!BufferOrErr)
    return createStringError(BufferOrErr.getError(),
                             formatv("cannot read learned model {0}", Path));
  std::unique_ptr<MemoryBuffer> Buffer = std::move(*BufferOrErr);
  StringRef Bytes = Buffer->getBuffer();
  if (Bytes.size() < HeaderSize)
    return createStringError(inconvertibleErrorCode(),
                             "invalid learned model magic");
  if (Bytes.take_front(sizeof(NeuralMagic)) !=
      StringRef(NeuralMagic, sizeof(NeuralMagic)))
    return createStringError(inconvertibleErrorCode(),
                             "invalid learned model magic");
  auto FormatVersion = readLE<uint32_t>(Bytes, 8);
  auto Endian = readLE<uint32_t>(Bytes, 12);
  if (!FormatVersion || !Endian || *FormatVersion != Version ||
      *Endian != EndianMarker)
    return createStringError(inconvertibleErrorCode(),
                             "incompatible learned model header");
  if (std::memcmp(Bytes.data() + 48, FrozenFeatureSchemaSHA,
                  sizeof(FrozenFeatureSchemaSHA)) != 0)
    return createStringError(inconvertibleErrorCode(),
                             "incompatible learned feature schema");

  auto ActionDim = readLE<uint32_t>(Bytes, 80);
  auto StateDim = readLE<uint32_t>(Bytes, 84);
  auto TargetDim = readLE<uint32_t>(Bytes, 88);
  auto ParameterCount = readLE<uint32_t>(Bytes, 92);
  auto NormalizationOffset = readLE<uint64_t>(Bytes, 96);
  auto WeightsOffset = readLE<uint64_t>(Bytes, 104);
  auto FileSize = readLE<uint64_t>(Bytes, 112);
  if (!ActionDim || !StateDim || !TargetDim || !ParameterCount ||
      !NormalizationOffset || !WeightsOffset || !FileSize || *ActionDim != 55 ||
      *StateDim != 22 || *TargetDim != 5 ||
      *ParameterCount != NeuralParameterCount || *FileSize != Bytes.size())
    return createStringError(inconvertibleErrorCode(),
                             "incompatible neural model header");
  const uint64_t NormalizationCount =
      2 * uint64_t(*ActionDim + *StateDim + *TargetDim);
  if (Error E = checkRange(Bytes, *NormalizationOffset, NormalizationCount,
                           sizeof(float), "neural normalization"))
    return std::move(E);
  if (Error E = checkRange(Bytes, *WeightsOffset, *ParameterCount,
                           sizeof(float), "neural weights"))
    return std::move(E);
  auto Model = std::shared_ptr<LearnedPreRAModel>(new LearnedPreRAModel());
  Model->Storage = std::move(Buffer);
  std::memcpy(Model->SourceSHA.data(), Bytes.data() + 16, 32);
  std::memcpy(Model->SchemaSHA.data(), Bytes.data() + 48, 32);
  const float *Norm =
      reinterpret_cast<const float *>(Bytes.data() + *NormalizationOffset);
  Model->NeuralActionMean = ArrayRef(Norm, *ActionDim);
  Model->NeuralActionStd = ArrayRef(Norm + *ActionDim, *ActionDim);
  Model->NeuralStateMean = ArrayRef(Norm + 2 * *ActionDim, *StateDim);
  Model->NeuralStateStd =
      ArrayRef(Norm + 2 * *ActionDim + *StateDim, *StateDim);
  Model->NeuralTargetMean =
      ArrayRef(Norm + 2 * (*ActionDim + *StateDim), *TargetDim);
  Model->NeuralTargetStd =
      ArrayRef(Norm + 2 * (*ActionDim + *StateDim) + *TargetDim, *TargetDim);
  Model->NeuralWeights =
      ArrayRef(reinterpret_cast<const float *>(Bytes.data() + *WeightsOffset),
               *ParameterCount);
  return std::shared_ptr<const LearnedPreRAModel>(std::move(Model));
}

Expected<std::shared_ptr<const LearnedPreRAModel>>
llvm::AMDGPU::getCachedLearnedPreRAModel(StringRef Path) {
  static std::mutex Mutex;
  // Keep validated model storage alive for the process lifetime.
  static StringMap<std::shared_ptr<const LearnedPreRAModel>> Cache;
  std::lock_guard<std::mutex> Guard(Mutex);
  auto It = Cache.find(Path);
  if (It != Cache.end())
    return It->second;
  auto Loaded = LearnedPreRAModel::load(Path);
  if (!Loaded)
    return Loaded.takeError();
  Cache[Path] = *Loaded;
  return *Loaded;
}
