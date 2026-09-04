// Multi-buffered vector matrix multiply for Cortex-A53 schedule search.
// Build with -DTILE_BUFFERS=1, 2, or 3.  Each buffer adds an independent 4x8
// output tile to the same K-loop (eight more four-lane accumulator chains).

#ifndef TILE_BUFFERS
#define TILE_BUFFERS 2
#endif

#if TILE_BUFFERS < 1 || TILE_BUFFERS > 3
#error TILE_BUFFERS must be 1, 2, or 3
#endif

#define N 48
#define TILE_COLUMNS (8 * TILE_BUFFERS)

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
      for (unsigned J = 0; J != N; J += TILE_COLUMNS) {
        Float4 Acc[4][2 * TILE_BUFFERS] = {0};

#pragma clang loop unroll(disable)
        for (unsigned K = 0; K != N; K += 2) {
#pragma clang loop unroll(full)
          for (unsigned T = 0; T != TILE_BUFFERS; ++T) {
            Float4 B00 = load4(&B[K * N + J + T * 8]);
            Float4 B01 = load4(&B[K * N + J + T * 8 + 4]);
            Float4 B10 = load4(&B[(K + 1) * N + J + T * 8]);
            Float4 B11 = load4(&B[(K + 1) * N + J + T * 8 + 4]);
#pragma clang loop unroll(full)
            for (unsigned R = 0; R != 4; ++R) {
              float A0 = A[(I + R) * N + K];
              float A1 = A[(I + R) * N + K + 1];
              Acc[R][2 * T] += splat(A0) * B00;
              Acc[R][2 * T + 1] += splat(A0) * B01;
              Acc[R][2 * T] += splat(A1) * B10;
              Acc[R][2 * T + 1] += splat(A1) * B11;
            }
          }
        }

#pragma clang loop unroll(full)
        for (unsigned R = 0; R != 4; ++R) {
#pragma clang loop unroll(full)
          for (unsigned T = 0; T != TILE_BUFFERS; ++T) {
            store4(&C[(I + R) * N + J + T * 8], Acc[R][2 * T]);
            store4(&C[(I + R) * N + J + T * 8 + 4], Acc[R][2 * T + 1]);
          }
        }
      }
    }
  }

  unsigned long Checksum = 0;
  for (unsigned I = 0; I != N * N; ++I)
    Checksum = Checksum * 131 + (unsigned long)(C[I] * 4096.0f);
  return Checksum;
}
