; RUN: rm -f %t.founder %t.founder.jsonl %t.child %t.child.jsonl %t.replayed
; RUN: llc %s -o /dev/null -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 \
; RUN:   -verify-machineinstrs -aarch64-prera-training-function=kernel \
; RUN:   -aarch64-prera-training-record-schedule=%t.founder \
; RUN:   -aarch64-prera-training-record-trajectory=%t.founder.jsonl
; RUN: FileCheck %s --check-prefix=FOUNDER < %t.founder
; RUN: FileCheck %s --check-prefix=FOUNDER-JSON < %t.founder.jsonl
; RUN: llc %s -o /dev/null -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 \
; RUN:   -verify-machineinstrs -aarch64-prera-training-function=kernel \
; RUN:   -aarch64-prera-training-replay-schedule=%t.founder \
; RUN:   -aarch64-prera-training-mutation-region=0 \
; RUN:   -aarch64-prera-training-mutation-depth=1 \
; RUN:   -aarch64-prera-training-seed=17 \
; RUN:   -aarch64-prera-training-record-schedule=%t.child \
; RUN:   -aarch64-prera-training-record-trajectory=%t.child.jsonl
; RUN: FileCheck %s --check-prefix=TRANSITION < %t.child.jsonl
; RUN: not diff %t.founder %t.child
; RUN: llc %s -o /dev/null -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 \
; RUN:   -verify-machineinstrs -aarch64-prera-training-function=kernel \
; RUN:   -aarch64-prera-training-replay-schedule=%t.child \
; RUN:   -aarch64-prera-training-record-schedule=%t.replayed
; RUN: diff %t.child %t.replayed

; FOUNDER: aarch64-prera-schedule-v1	kernel	0
; FOUNDER-JSON: "format":"aarch64-prera-trajectory-v1"
; FOUNDER-JSON-SAME: "kind":"state"
; FOUNDER-JSON-SAME: "role":"founder"
; TRANSITION: "kind":"transition"
; TRANSITION-SAME: "requested_depth":1

define i64 @kernel(i64 %a, i64 %b, ptr %p) {
entry:
  %x0 = add i64 %a, 1
  %x1 = add i64 %b, 2
  %x2 = mul i64 %x0, %x0
  %x3 = mul i64 %x1, %x1
  %x4 = add i64 %x2, %x3
  %x5 = load i64, ptr %p, align 8
  %x6 = add i64 %x4, %x5
  ret i64 %x6
}
