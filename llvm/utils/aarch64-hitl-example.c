// Minimal workload for exercising the Cortex-A53 HITL transport and runner.
// This is a smoke-test kernel, not a representative training benchmark.

__attribute__((visibility("default"), noinline))
unsigned long hitl_kernel(unsigned long Seed, unsigned long Iterations) {
  unsigned long A = Seed | 1;
  unsigned long B = Seed ^ 0x9e3779b97f4a7c15UL;
  for (unsigned long I = 0; I != Iterations; ++I) {
    A ^= A << 7;
    B ^= B >> 9;
    A *= 0xd1342543de82ef95UL;
    B *= 0x94d049bb133111ebUL;
    A += B ^ I;
    B += A >> 11;
  }
  return A ^ B;
}
