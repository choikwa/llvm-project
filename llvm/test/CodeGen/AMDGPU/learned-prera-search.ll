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
; RUN: rm -f %t.founder.schedule %t.founder.jsonl %t.child.schedule \
; RUN:   %t.child.jsonl %t.replayed.schedule
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs \
; RUN:   -amdgpu-prera-training-function=learned_prera_fallback \
; RUN:   -amdgpu-prera-training-record-schedule=%t.founder.schedule \
; RUN:   -amdgpu-prera-training-record-trajectory=%t.founder.jsonl \
; RUN:   -o %t.founder.s %s
; RUN: cmp %t.disabled %t.founder.s
; RUN: FileCheck %s --check-prefix=TRAIN-SCHEDULE < %t.founder.schedule
; RUN: FileCheck %s --check-prefix=TRAIN-STATE < %t.founder.jsonl
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs \
; RUN:   -amdgpu-prera-training-function=learned_prera_fallback \
; RUN:   -amdgpu-prera-training-replay-schedule=%t.founder.schedule \
; RUN:   -amdgpu-prera-training-mutation-depth=4 \
; RUN:   -amdgpu-prera-training-mutation-region=0 \
; RUN:   -amdgpu-prera-training-seed=7 \
; RUN:   -amdgpu-prera-training-record-schedule=%t.child.schedule \
; RUN:   -amdgpu-prera-training-record-trajectory=%t.child.jsonl \
; RUN:   -o %t.child.s %s
; RUN: not cmp %t.founder.schedule %t.child.schedule
; RUN: grep -c '"kind":"transition"' %t.child.jsonl | \
; RUN:   FileCheck %s --check-prefix=TRANSITION-COUNT
; RUN: FileCheck %s --check-prefix=TRAIN-TRANSITION < %t.child.jsonl
; RUN: %python -c "import json; rows=[json.loads(x) for x in open(r'%t.child.jsonl')]; transitions=[x for x in rows if x['kind']=='transition']; endpoints=[x for x in rows if x['kind']=='endpoint']; assert len(transitions)==4 and len(endpoints)==1; assert all(len(x['parent_features'])==22 and len(x['child_features'])==22 and len(x['action_features'])==55 for x in transitions+endpoints)"
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs \
; RUN:   -amdgpu-prera-training-function=learned_prera_fallback \
; RUN:   -amdgpu-prera-training-replay-schedule=%t.child.schedule \
; RUN:   -amdgpu-prera-training-record-schedule=%t.replayed.schedule \
; RUN:   -o %t.replayed.s %s
; RUN: cmp %t.child.schedule %t.replayed.schedule
; RUN: cmp %t.child.s %t.replayed.s
;
; The external trained model is intentionally not a test dependency. Verify
; that enabling the experimental optimizer without a usable model preserves
; the exact production GCN schedule and does not weaken MachineVerifier.
;
; TRAIN-SCHEDULE: amdgpu-prera-schedule-v1 learned_prera_fallback 0 {{[0-9a-f]+}} {{[0-9,]+}}
; TRAIN-STATE: "feature_schema_sha256":"f346529d24c027c55709e9dac6744d561cfa7d46ee30a727fe5ddc327515ee62"
; TRAIN-STATE-SAME: "format":"amdgpu-prera-trajectory-v1"
; TRAIN-STATE-SAME: "kind":"state"
; TRAIN-STATE-SAME: "state_features":[
; TRANSITION-COUNT: 4
; TRAIN-TRANSITION: "action_features":[
; TRAIN-TRANSITION-SAME: "child":[
; TRAIN-TRANSITION-SAME: "child_pressure":{
; TRAIN-TRANSITION-SAME: "kind":"transition"
; TRAIN-TRANSITION-SAME: "move":{
; TRAIN-TRANSITION-SAME: "parent_features":[

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
