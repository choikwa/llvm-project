; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-iterations=200 \
; RUN:   -amdgpu-annealing-tile-size=8 -amdgpu-annealing-tile-overlap=3 \
; RUN:   -amdgpu-annealing-sweeps=2 \
; RUN:   -amdgpu-annealing-seed=7 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-oracle \
; RUN:   -amdgpu-annealing-iterations=500 -amdgpu-annealing-sweeps=3 \
; RUN:   -amdgpu-annealing-seed=11 -verify-machineinstrs -o /dev/null < %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-oracle \
; RUN:   -amdgpu-annealing-oracle-input-start \
; RUN:   -amdgpu-annealing-iterations=500 -amdgpu-annealing-sweeps=3 \
; RUN:   -amdgpu-annealing-seed=13 -verify-machineinstrs -o /dev/null < %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-oracle \
; RUN:   -amdgpu-annealing-hierarchical \
; RUN:   -amdgpu-annealing-hierarchical-min-tile=4 \
; RUN:   -amdgpu-annealing-iterations=500 -amdgpu-annealing-sweeps=2 \
; RUN:   -amdgpu-annealing-seed=17 -verify-machineinstrs -o /dev/null < %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-oracle \
; RUN:   -amdgpu-annealing-oracle-vmem-moves \
; RUN:   -amdgpu-annealing-oracle-relax-pressure \
; RUN:   -amdgpu-annealing-iterations=20 -amdgpu-annealing-sweeps=1 \
; RUN:   -amdgpu-annealing-seed=19 -verify-machineinstrs -o /dev/null < %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-max-occupancy \
; RUN:   -verify-machineinstrs -o %t.maxocc.s < %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-iterations=0 \
; RUN:   -verify-machineinstrs -o %t.zero.s < %s
; RUN: diff %t.maxocc.s %t.zero.s
; RUN: rm -f %t.schedule
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-oracle \
; RUN:   -amdgpu-annealing-iterations=200 -amdgpu-annealing-seed=23 \
; RUN:   -amdgpu-annealing-record-schedule=%t.schedule \
; RUN:   -verify-machineinstrs -o %t.parent.s < %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 -O3 \
; RUN:   -misched=gcn-annealing -amdgpu-annealing-oracle \
; RUN:   -amdgpu-annealing-iterations=0 \
; RUN:   -amdgpu-annealing-replay-schedule=%t.schedule \
; RUN:   -verify-machineinstrs -o %t.child.s < %s
; RUN: diff %t.parent.s %t.child.s

; Exercise independent instructions, a dependent chain, and loads so adjacent
; swaps explore a non-trivial scheduling region.
define amdgpu_kernel void @annealing_test(ptr addrspace(1) %out,
                                          ptr addrspace(1) %a,
                                          ptr addrspace(1) %b) {
; CHECK-LABEL: annealing_test:
; CHECK: global_load
; CHECK: v_fma
; CHECK: global_store
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %id
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %id
  %op = getelementptr float, ptr addrspace(1) %out, i32 %id
  %av = load float, ptr addrspace(1) %ap, align 4
  %bv = load float, ptr addrspace(1) %bp, align 4
  %x0 = call float @llvm.fma.f32(float %av, float 1.000100e+00, float %bv)
  %x1 = call float @llvm.fma.f32(float %x0, float 9.999000e-01, float %av)
  %x2 = call float @llvm.fma.f32(float %x1, float 1.000100e+00, float %bv)
  store float %x2, ptr addrspace(1) %op, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.fma.f32(float, float, float)
