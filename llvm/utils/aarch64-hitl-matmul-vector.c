// Vectorized, cache-resident matrix multiply for Cortex-A53 HITL search.
//
// A 4x8 output tile keeps eight independent four-lane accumulators live.  Two
// K iterations are written in each loop body to expose independent vector
// loads and multiply-adds to the pre-RA complete-schedule search endpoint.

#define N 32

typedef float Float4 __attribute__((ext_vector_type(4)));

static __attribute__((always_inline)) Float4 load4(const float *P) {
  Float4 V;
  __builtin_memcpy(&V, P, sizeof(V));
  return V;
}

static __attribute__((always_inline)) void store4(float *P, Float4 V) {
  __builtin_memcpy(P, &V, sizeof(V));
}

static __attribute__((always_inline)) Float4 splat(float X) {
  return (Float4){X, X, X, X};
}

__attribute__((visibility("default"), noinline))
unsigned long hitl_kernel(unsigned long Seed, unsigned long Iterations) {
  float A[N * N] __attribute__((aligned(16)));
  float B[N * N] __attribute__((aligned(16)));
  float C[N * N] __attribute__((aligned(16)));

  for (unsigned I = 0; I != N * N; ++I) {
    unsigned long X = Seed + I * 0x9e3779b97f4a7c15UL;
    A[I] = (float)((X >> 17) & 255) * (1.0f / 256.0f);
    B[I] = (float)((X >> 29) & 255) * (1.0f / 256.0f);
    C[I] = (float)(I & 7) * (1.0f / 1024.0f);
  }

#pragma clang loop unroll(disable)
  for (unsigned long Repeat = 0; Repeat != Iterations; ++Repeat) {
#pragma clang loop unroll(disable)
    for (unsigned I = 0; I != N; I += 4) {
#pragma clang loop unroll(disable)
      for (unsigned J = 0; J != N; J += 8) {
        Float4 C00 = (Float4){0, 0, 0, 0};
        Float4 C01 = C00;
        Float4 C10 = C00;
        Float4 C11 = C00;
        Float4 C20 = C00;
        Float4 C21 = C00;
        Float4 C30 = C00;
        Float4 C31 = C00;

#pragma clang loop unroll(disable)
        for (unsigned K = 0; K != N; K += 2) {
          Float4 B00 = load4(&B[K * N + J]);
          Float4 B01 = load4(&B[K * N + J + 4]);
          Float4 B10 = load4(&B[(K + 1) * N + J]);
          Float4 B11 = load4(&B[(K + 1) * N + J + 4]);

          C00 += splat(A[(I + 0) * N + K]) * B00;
          C01 += splat(A[(I + 0) * N + K]) * B01;
          C10 += splat(A[(I + 1) * N + K]) * B00;
          C11 += splat(A[(I + 1) * N + K]) * B01;
          C20 += splat(A[(I + 2) * N + K]) * B00;
          C21 += splat(A[(I + 2) * N + K]) * B01;
          C30 += splat(A[(I + 3) * N + K]) * B00;
          C31 += splat(A[(I + 3) * N + K]) * B01;

          C00 += splat(A[(I + 0) * N + K + 1]) * B10;
          C01 += splat(A[(I + 0) * N + K + 1]) * B11;
          C10 += splat(A[(I + 1) * N + K + 1]) * B10;
          C11 += splat(A[(I + 1) * N + K + 1]) * B11;
          C20 += splat(A[(I + 2) * N + K + 1]) * B10;
          C21 += splat(A[(I + 2) * N + K + 1]) * B11;
          C30 += splat(A[(I + 3) * N + K + 1]) * B10;
          C31 += splat(A[(I + 3) * N + K + 1]) * B11;
        }

        store4(&C[(I + 0) * N + J], C00);
        store4(&C[(I + 0) * N + J + 4], C01);
        store4(&C[(I + 1) * N + J], C10);
        store4(&C[(I + 1) * N + J + 4], C11);
        store4(&C[(I + 2) * N + J], C20);
        store4(&C[(I + 2) * N + J + 4], C21);
        store4(&C[(I + 3) * N + J], C30);
        store4(&C[(I + 3) * N + J + 4], C31);
      }
    }
  }

  unsigned long Checksum = 0;
  for (unsigned I = 0; I != N * N; ++I)
    Checksum = Checksum * 131 + (unsigned long)(C[I] * 4096.0f);
  return Checksum;
}
