#!/usr/bin/env python3
# ===- export-amdgpu-prera-nn.py -----------------------------------------=== #
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===---------------------------------------------------------------------=== #

"""Export a frozen AMDGPU pre-RA C33 shared MLP to LLVM's native blob."""

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np
import torch


MAGIC = b"AMDPRANN"
VERSION = 1
ENDIAN_MARKER = 0x01020304
HEADER_SIZE = 256
SCHEMA_SHA = bytes.fromhex(
    "f346529d24c027c55709e9dac6744d561cfa7d46ee30a727fe5ddc327515ee62"
)
ACTION_HEADS = ("immediate", "m1", "m2", "m3", "positive", "accepted", "elite")


def as_f32(values):
    return np.asarray(values, dtype="<f4").reshape(-1)


def tensor(state, name):
    if name not in state:
        raise ValueError(f"checkpoint is missing {name}")
    return state[name].detach().cpu().numpy().astype("<f4", copy=False).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "shared":
        raise ValueError("only the shared C33 architecture is supported")
    norm = payload["normalization"]
    if norm["target"]["names"] != ["immediate", "m1", "m2", "m3", "endpoint"]:
        raise ValueError("unexpected target order")

    normalization = np.concatenate(
        [
            as_f32(norm["action"]["mean"]),
            as_f32(norm["action"]["std"]),
            as_f32(norm["state"]["mean"]),
            as_f32(norm["state"]["std"]),
            as_f32(norm["target"]["mean"]),
            as_f32(norm["target"]["std"]),
        ]
    )
    if normalization.size != 164:
        raise ValueError(f"unexpected normalization size {normalization.size}")

    state = payload["state_dict"]
    names = [
        "action_trunk.net.0.weight",
        "action_trunk.net.0.bias",
        "action_trunk.net.2.weight",
        "action_trunk.net.2.bias",
        "action_trunk.net.4.weight",
        "action_trunk.net.4.bias",
    ]
    for head in ACTION_HEADS:
        names += [f"action_heads.{head}.weight", f"action_heads.{head}.bias"]
    names += [
        "endpoint.0.net.0.weight",
        "endpoint.0.net.0.bias",
        "endpoint.0.net.2.weight",
        "endpoint.0.net.2.bias",
        "endpoint.1.weight",
        "endpoint.1.bias",
    ]
    parameters = np.concatenate([tensor(state, name) for name in names])
    if parameters.size != 125192:
        raise ValueError(f"unexpected parameter count {parameters.size}")

    normalization_offset = HEADER_SIZE
    weights_offset = normalization_offset + normalization.nbytes
    file_size = weights_offset + parameters.nbytes
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<8sII", header, 0, MAGIC, VERSION, ENDIAN_MARKER)
    header[16:48] = hashlib.sha256(args.checkpoint.read_bytes()).digest()
    header[48:80] = SCHEMA_SHA
    struct.pack_into(
        "<IIIIQQQ",
        header,
        80,
        55,
        22,
        5,
        parameters.size,
        normalization_offset,
        weights_offset,
        file_size,
    )
    args.output.write_bytes(header + normalization.tobytes() + parameters.tobytes())
    print(f"wrote {args.output} ({file_size} bytes, {parameters.size} parameters)")


if __name__ == "__main__":
    main()
