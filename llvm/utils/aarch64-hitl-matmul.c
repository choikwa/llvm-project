// Deterministic cache-resident matrix multiply for Cortex-A53 HITL search.
// It implements the ABI expected by aarch64-hitl-runner.c.  Iterations is the
// number of full 16x16 matrix products performed by one timed invocation.

#define N 16

__attribute__((visibility("default"), noinline))
unsigned long hitl_kernel(unsigned long Seed, unsigned long Iterations) {
  float A[N * N];
  float B[N * N];
  float C[N * N];

  for (unsigned I = 0; I != N * N; ++I) {
    unsigned long X = Seed + I * 0x9e3779b97f4a7c15UL;
    A[I] = (float)((X >> 17) & 255) * (1.0f / 256.0f);
    B[I] = (float)((X >> 29) & 255) * (1.0f / 256.0f);
    C[I] = (float)(I & 7) * (1.0f / 1024.0f);
  }

#pragma clang loop unroll(disable)
  for (unsigned long Repeat = 0; Repeat != Iterations; ++Repeat) {
#pragma clang loop unroll(disable)
    for (unsigned I = 0; I != N; ++I) {
#pragma clang loop unroll(disable)
      for (unsigned J = 0; J != N; ++J) {
        float Sum0 = 0.0f;
        float Sum1 = 0.0f;
        float Sum2 = 0.0f;
        float Sum3 = 0.0f;
        // Keep four independent accumulation chains.  This is deliberately
        // written out rather than relying on the optimizer's unroller so the
        // pre-RA schedule endpoint has a real ILP-bearing region to explore.
#pragma clang loop unroll(disable)
#pragma clang loop vectorize(disable)
        for (unsigned K = 0; K != N; K += 4) {
          Sum0 += A[I * N + K] * B[K * N + J];
          Sum1 += A[I * N + K + 1] * B[(K + 1) * N + J];
          Sum2 += A[I * N + K + 2] * B[(K + 2) * N + J];
          Sum3 += A[I * N + K + 3] * B[(K + 3) * N + J];
        }
        float Sum = (Sum0 + Sum1) + (Sum2 + Sum3);
        C[I * N + J] = Sum + C[I * N + J] * (1.0f / 65536.0f);
      }
    }
  }

  unsigned long Checksum = 0;
  for (unsigned I = 0; I != N * N; ++I)
    Checksum = Checksum * 131 + (unsigned long)(C[I] * 4096.0f);
  return Checksum;
}
