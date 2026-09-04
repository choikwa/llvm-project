; RUN: rm -f %t.founder %t.child
; RUN: llc %s -o /dev/null -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 \
; RUN:   -verify-machineinstrs -aarch64-prera-training-function=vector_kernel \
; RUN:   -aarch64-prera-training-record-schedule=%t.founder
; RUN: llc %s -o /dev/null -mtriple=aarch64-linux-gnu -mcpu=cortex-a53 \
; RUN:   -verify-machineinstrs -aarch64-prera-training-function=vector_kernel \
; RUN:   -aarch64-prera-training-replay-schedule=%t.founder \
; RUN:   -aarch64-prera-training-mutation-region=1 \
; RUN:   -aarch64-prera-training-mutation-depth=4 \
; RUN:   -aarch64-prera-training-seed=17578002864718460215 \
; RUN:   -aarch64-prera-training-record-schedule=%t.child
; RUN: not diff %t.founder %t.child

define <4 x float> @vector_kernel(ptr %a, ptr %b, i64 %count) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %sum0, %loop ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %sum1, %loop ]
  %ap = getelementptr float, ptr %a, i64 %iv
  %bp = getelementptr <4 x float>, ptr %b, i64 %iv
  %scalar0 = load float, ptr %ap, align 4
  %scalar1p = getelementptr float, ptr %ap, i64 16
  %scalar1 = load float, ptr %scalar1p, align 4
  %vector = load <4 x float>, ptr %bp, align 16
  %s0 = insertelement <4 x float> poison, float %scalar0, i64 0
  %splat0 = shufflevector <4 x float> %s0, <4 x float> poison, <4 x i32> zeroinitializer
  %s1 = insertelement <4 x float> poison, float %scalar1, i64 0
  %splat1 = shufflevector <4 x float> %s1, <4 x float> poison, <4 x i32> zeroinitializer
  %mul0 = fmul fast <4 x float> %vector, %splat0
  %mul1 = fmul fast <4 x float> %vector, %splat1
  %sum0 = fadd fast <4 x float> %acc0, %mul0
  %sum1 = fadd fast <4 x float> %acc1, %mul1
  %next = add nuw i64 %iv, 1
  %done = icmp eq i64 %next, %count
  br i1 %done, label %exit, label %loop

exit:
  %result = fadd fast <4 x float> %sum0, %sum1
  ret <4 x float> %result
}
