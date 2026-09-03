; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs \
; RUN:   -o %t.disabled %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs \
; RUN:   -amdgpu-learned-prera-sched -o %t.unconfigured %s
; RUN: cmp %t.disabled %t.unconfigured
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs \
; RUN:   -amdgpu-learned-prera-sched -amdgpu-learned-prera-model=%s \
; RUN:   -o %t.invalid-model %s
; RUN: cmp %t.disabled %t.invalid-model
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -verify-machineinstrs \
; RUN:   -o %t.gfx942-disabled %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -verify-machineinstrs \
; RUN:   -amdgpu-learned-prera-sched -amdgpu-learned-prera-model=%s \
; RUN:   -o %t.gfx942-enabled %s
; RUN: cmp %t.gfx942-disabled %t.gfx942-enabled
;
; The external trained model is intentionally not a test dependency. Verify
; that enabling the experimental optimizer without a usable model preserves
; the exact production GCN schedule and does not weaken MachineVerifier.

define amdgpu_kernel void @learned_prera_fallback(ptr addrspace(1) %out,
                                                  ptr addrspace(1) %in) {
entry:
  %x0 = load float, ptr addrspace(1) %in, align 4
  %p1 = getelementptr float, ptr addrspace(1) %in, i64 1
  %x1 = load float, ptr addrspace(1) %p1, align 4
  %a0 = fadd float %x0, 1.0
  %a1 = fmul float %x1, 2.0
  %r = fadd float %a0, %a1
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}
