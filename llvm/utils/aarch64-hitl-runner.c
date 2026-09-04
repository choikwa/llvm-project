// Reference native timing runner for AArch64 hardware-in-the-loop search.
// Candidate shared objects must export:
//   uint64_t hitl_kernel(uint64_t seed, uint64_t iterations);

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef uint64_t (*kernel_fn)(uint64_t, uint64_t);

static uint64_t now_ns(void) {
  struct timespec TS;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &TS) != 0) {
    perror("clock_gettime");
    exit(2);
  }
  return (uint64_t)TS.tv_sec * UINT64_C(1000000000) + (uint64_t)TS.tv_nsec;
}

static double measure(kernel_fn Kernel, uint64_t Seed, uint64_t Iterations,
                      uint64_t *Result) {
  uint64_t Begin = now_ns();
  *Result = Kernel(Seed, Iterations);
  uint64_t End = now_ns();
  return (double)(End - Begin) / 1000.0;
}

static int compare_double(const void *Left, const void *Right) {
  double A = *(const double *)Left;
  double B = *(const double *)Right;
  return (A > B) - (A < B);
}

static double median(const double *Samples, unsigned Count) {
  double *Copy = malloc(sizeof(double) * Count);
  if (!Copy) {
    perror("malloc");
    exit(2);
  }
  for (unsigned I = 0; I != Count; ++I)
    Copy[I] = Samples[I];
  qsort(Copy, Count, sizeof(double), compare_double);
  double Result = Count & 1 ? Copy[Count / 2]
                            : (Copy[Count / 2 - 1] + Copy[Count / 2]) / 2.0;
  free(Copy);
  return Result;
}

static kernel_fn load_kernel(const char *Path, void **Handle) {
  *Handle = dlopen(Path, RTLD_NOW | RTLD_LOCAL);
  if (!*Handle) {
    fprintf(stderr, "dlopen %s: %s\n", Path, dlerror());
    exit(2);
  }
  dlerror();
  kernel_fn Kernel = (kernel_fn)dlsym(*Handle, "hitl_kernel");
  const char *Error = dlerror();
  if (Error) {
    fprintf(stderr, "dlsym hitl_kernel in %s: %s\n", Path, Error);
    exit(2);
  }
  return Kernel;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s BASELINE.so CANDIDATE.so ITERATIONS TRIALS\n",
            argv[0]);
    return 2;
  }
  char *End = NULL;
  errno = 0;
  uint64_t Iterations = strtoull(argv[3], &End, 0);
  if (errno || !End || *End || !Iterations) {
    fprintf(stderr, "invalid iteration count\n");
    return 2;
  }
  unsigned long TrialsLong = strtoul(argv[4], &End, 0);
  if (errno || !End || *End || !TrialsLong || TrialsLong > 10000) {
    fprintf(stderr, "invalid trial count\n");
    return 2;
  }
  unsigned Trials = (unsigned)TrialsLong;

  void *BaselineHandle = NULL, *CandidateHandle = NULL;
  kernel_fn Baseline = load_kernel(argv[1], &BaselineHandle);
  kernel_fn Candidate = load_kernel(argv[2], &CandidateHandle);
  const uint64_t Seed = UINT64_C(0x243f6a8885a308d3);
  uint64_t BaselineResult = 0, CandidateResult = 0;
  for (unsigned I = 0; I != 3; ++I) {
    BaselineResult = Baseline(Seed, Iterations);
    CandidateResult = Candidate(Seed, Iterations);
  }

  double *BaselineSamples = calloc(Trials, sizeof(double));
  double *CandidateSamples = calloc(Trials, sizeof(double));
  if (!BaselineSamples || !CandidateSamples) {
    perror("calloc");
    return 2;
  }
  int Correct = BaselineResult == CandidateResult;
  for (unsigned I = 0; I != Trials; ++I) {
    if ((I / 2) & 1) {
      CandidateSamples[I] =
          measure(Candidate, Seed, Iterations, &CandidateResult);
      BaselineSamples[I] = measure(Baseline, Seed, Iterations, &BaselineResult);
    } else {
      BaselineSamples[I] = measure(Baseline, Seed, Iterations, &BaselineResult);
      CandidateSamples[I] =
          measure(Candidate, Seed, Iterations, &CandidateResult);
    }
    Correct &= BaselineResult == CandidateResult;
  }

  printf("{\"baseline_runtime_us\":%.9g,\"candidate_runtime_us\":%.9g,"
         "\"correct\":%s,\"baseline_samples_us\":[",
         median(BaselineSamples, Trials), median(CandidateSamples, Trials),
         Correct ? "true" : "false");
  for (unsigned I = 0; I != Trials; ++I)
    printf("%s%.9g", I ? "," : "", BaselineSamples[I]);
  printf("],\"candidate_samples_us\":[");
  for (unsigned I = 0; I != Trials; ++I)
    printf("%s%.9g", I ? "," : "", CandidateSamples[I]);
  printf("],\"iterations\":%" PRIu64 ",\"trials\":%u}\n", Iterations, Trials);

  free(BaselineSamples);
  free(CandidateSamples);
  dlclose(BaselineHandle);
  dlclose(CandidateHandle);
  return 0;
}
