# Complete-schedule search in the machine scheduler

LLVM's usual machine-scheduling interface asks a `MachineSchedStrategy` to
choose one ready `SUnit` at a time.  `MachineSchedCompleteScheduleOptimizer`
supports a different kind of scheduler: one that explores complete instruction
orders and returns one selected order.

This interface is useful for search algorithms and learned optimizers whose
natural state is a complete schedule.  It does not prescribe a search
algorithm, scoring function, model, or training system.

## Choose an integration point

There are two ways to use a complete-schedule optimizer.

**Refine an existing schedule (usually preferred).**  First run the target's
normal scheduler, then give the materialized schedule to the optimizer as its
founder.  Install the optimizer with
`ScheduleDAGMI::setPostScheduleOptimizer()`.  This is the appropriate path when
the search or model was trained relative to a particular production scheduler.

```text
build DAG -> normal scheduler -> complete founder -> optimizer -> final order
```

**Replace the scheduling strategy.**  Wrap the optimizer in
`MachineSchedCompleteScheduleReplayer`.  The optimizer receives the DAG's
initial ordinal order (or a stable topological order when that order is not
legal), and the selected complete schedule is replayed through LLVM's normal
incremental scheduling machinery.

```text
build DAG -> complete-schedule optimizer -> replay selected order
```

In both cases LLVM validates the returned permutation and every strong DAG
dependency before changing the instruction stream.  Returning `false`, or
returning an invalid order, preserves the founder.

## Implement an optimizer

Include `llvm/CodeGen/MachineSchedSearch.h` and derive from
`MachineSchedCompleteScheduleOptimizer`:

```cpp
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineSchedSearch.h"

using namespace llvm;

class EarliestLegalMove final
    : public MachineSchedCompleteScheduleOptimizer {
public:
  bool optimizeCompleteSchedule(const MachineSchedSearchRegion &Region,
                                ArrayRef<unsigned> Founder,
                                SmallVectorImpl<unsigned> &Result) override {
    Result.assign(Founder.begin(), Founder.end());

    // A real implementation would select this node and destination using a
    // cost model, stochastic search, or learned policy.  This example performs
    // the first nontrivial legal move it finds.
    for (unsigned Node : Founder) {
      MachineSchedSearchRegion::MoveRange Range;
      if (!Region.getLegalMoveRange(Result, Node, Range))
        return false;

      unsigned OldPosition = llvm::find(Result, Node) - Result.begin();
      if (Range.Begin == OldPosition)
        continue;

      return Region.applyRelocation(
          Result, {Node, OldPosition, Range.Begin}, Result);
    }
    return false;
  }
};
```

`Founder` and `Result` are permutations of **region-local SUnit ordinals** in
the range `[0, Region.size())`.  An ordinal is a stable identity for one node;
it is not the node's position in `Founder`, and it need not equal
`SUnit::NodeNum`.  For example:

```text
SUnit ordinal:        7
position in Founder: 12
position in Result:   4
```

Useful operations on `MachineSchedSearchRegion` include:

- `getSUnit(Node)` to inspect the `SUnit` and its `MachineInstr`;
- `predecessors(Node)` and `successors(Node)` for strong dependencies;
- `isLegalOrder(Order)` for complete permutation and dependency validation;
- `getLegalMoveRange(Order, Node, Range)` for a dependency-preserving
  remove-and-reinsert mutation;
- `applyRelocation(Order, Move, Result)` to apply that mutation using the same
  position semantics as other complete-schedule clients; and
- `getInitialOrder()` and `getTopologicalOrder()` for deterministic starting
  orders.

Weak DAG edges are scheduling preferences, not legality constraints.  A target
with additional constraints may override `validateCompleteSchedule()`.  Such a
validator should reject a candidate rather than silently repair it, so search
results remain reproducible and diagnosable.

The search callback must not mutate the `MachineFunction`, `SUnit` storage, or
the DAG.  It may inspect them through `Region`; `Region.getDAG()` is non-null
when the view was constructed for a live scheduler DAG.  LLVM applies only the
selected result and repairs generic liveness information afterward.

## Refine the output of an existing scheduler

A target DAG can install a post-schedule optimizer before it is returned from
`TargetMachine::createMachineScheduler()`:

```cpp
ScheduleDAGInstrs *MyTargetMachine::createMachineScheduler(
    MachineSchedContext *C) const {
  auto *DAG = new ScheduleDAGMILive(C, std::make_unique<MySchedStrategy>(C));
  DAG->setPostScheduleOptimizer(std::make_unique<EarliestLegalMove>());
  return DAG;
}
```

The callback runs once for each completed scheduling region.  Its founder is
the order that `MySchedStrategy` actually materialized, expressed using the
stable region-local ordinals.  The post-schedule hook is skipped if scheduling
was cut off before the region was completed.

`ScheduleDAGMILive` moves the selected instructions and, when liveness and
pressure tracking are active, recomputes the generic liveness flags needed by
the changed order.  Targets whose final order changes additional metadata can
override `applyCompleteSchedule()`, perform their target-specific update, and
delegate instruction movement and generic liveness maintenance to the base
implementation.

## Use an optimizer as the scheduling strategy

To search from the DAG's initial legal order instead of refining another
scheduler's result, use the replay adapter:

```cpp
ScheduleDAGInstrs *MyTargetMachine::createMachineScheduler(
    MachineSchedContext *C) const {
  auto Strategy = std::make_unique<MachineSchedCompleteScheduleReplayer>(
      std::make_unique<EarliestLegalMove>());
  return new ScheduleDAGMILive(C, std::move(Strategy));
}
```

The adapter computes the complete result before list scheduling begins.  It
then returns the selected nodes top-down, one at a time, so instruction motion,
ready-queue state, live intervals, and register-pressure bookkeeping remain in
the existing scheduler framework.

This adapter is not intended for algorithms that naturally choose one ready
node at a time.  Those algorithms should implement `MachineSchedStrategy`
directly.

## AMDGPU post-MaxOccupancy use

AMDGPU provides `GCNCompleteScheduleOptimizer` and
`createGCNPostScheduleOptimizerScheduler()` in
`lib/Target/AMDGPU/GCNCompleteScheduleOptimizer.h`.  This path runs the normal
GCN scheduling stages first and presents their post-MaxOccupancy schedule as
the founder:

```cpp
class MyGCNOptimizer final : public GCNCompleteScheduleOptimizer {
  bool optimizeGCNCompleteSchedule(const MachineSchedSearchRegion &Region,
                                   ArrayRef<unsigned> Founder,
                                   SmallVectorImpl<unsigned> &Result) override {
    // Search complete orders here.
    Result.assign(Founder.begin(), Founder.end());
    return false;
  }
};

ScheduleDAGInstrs *createMyGCNScheduler(MachineSchedContext *C) {
  return createGCNPostScheduleOptimizerScheduler(
      C, std::make_unique<MyGCNOptimizer>());
}
```

The AMDGPU adapter enables lane-mask and register-pressure tracking during
replay and preserves local virtual-register def/use relationships that are not
yet represented by strong DAG edges.  A derived optimizer may add further
target-specific checks in `validateGCNCompleteSchedule()`.

The complete-order callback is where a static simulated annealer, beam search,
greedy policy, or another in-process optimizer is selected.  The base API does
not contain a search-mode enum: search algorithms are clients of the API, not
policies built into it.

A static simulated annealer, for example, can be structured entirely inside
the callback:

```text
current = best = Founder
repeat until the static-evaluation budget is exhausted:
  choose a node and query getLegalMoveRange(current, node)
  construct one remove-and-reinsert child
  score (current, action, child) with the frozen static scorer
  accept the child according to the chosen annealing rule
  update best
Result = best
```

The same callback can instead call a beam-search or learned-policy library.  In
all cases it should return the best complete legal order it selected, not its
internal search state.

## Generate AMDGPU schedules and training trajectories

AMDGPU also provides a file-based endpoint for external search and training
data generation.  Like the historical deep-SA harness, it launches a fresh
compiler for each candidate.  The external process retains the search state;
LLVM reconstructs the post-MaxOccupancy region, replays the parent, creates a
legal child, emits its features, and compiles the selected endpoint.

The initial implementation is restricted to `gfx950` so that every emitted
feature vector has the same target semantics as the frozen C33 schema.

Record the production founder and its state features:

```console
$ build/bin/llc input.ll -o founder.s \
    -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -amdgpu-prera-training-function=matmul_kernel \
    -amdgpu-prera-training-record-schedule=founder.schedule \
    -amdgpu-prera-training-record-trajectory=founder.jsonl
```

Generate a reproducible depth-8 child from that founder:

```console
$ build/bin/llc input.ll -o child.s \
    -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -amdgpu-prera-training-function=matmul_kernel \
    -amdgpu-prera-training-replay-schedule=founder.schedule \
    -amdgpu-prera-training-mutation-region=0 \
    -amdgpu-prera-training-mutation-depth=8 \
    -amdgpu-prera-training-seed=17 \
    -amdgpu-prera-training-record-schedule=child.schedule \
    -amdgpu-prera-training-record-trajectory=child.jsonl
```

The schedule file is a versioned tab-separated interchange format:

```text
amdgpu-prera-schedule-v1<TAB>function<TAB>region<TAB>fingerprint<TAB>order
```

The fingerprint covers the function, region, post-MaxOccupancy founder,
instructions, and strong dependency graph.  Replay fails rather than silently
falling back when the fingerprint, permutation, DAG legality, or AMDGPU local
virtual-register ordering does not match.

The trajectory file is JSON Lines.  It contains a starting state record, one
transition record for every completed relocation, an endpoint record for the
whole depth-N proposal, and a selected-state record.  The endpoint's action
features use the same first-move, cumulative-distance, and depth semantics as
neural deployment.  Action records include:

- function, region, fingerprint, seed, requested depth, and step;
- parent and child schedule hashes and complete ordinal orders;
- moved node and old/new positions;
- the exact 22-element parent and child state vectors;
- the exact 55-element action vector consumed by the neural model;
- the frozen feature-schema SHA-256; and
- estimated parent/child VGPR, SGPR, and occupancy values.

Output files are opened in append mode so one compilation can record multiple
functions and regions.  The controlling harness should remove or rotate them
before starting a new candidate.  `-amdgpu-prera-training-function` is strongly
recommended when a module contains more than the one kernel being optimized.

An external hardware-in-the-loop controller can therefore use the same pattern
as the ROCm deep-SA experiments:

```text
record founder once
repeat:
  invoke compiler to replay parent and generate one legal child
  verify child by replaying and re-recording it
  execute child and collect runtime
  perform Metropolis acceptance externally
  append runtime and acceptance labels using the emitted hashes
```

Only the endpoint of a depth-N walk reaches the downstream compiler pipeline.
Every intermediate relocation is nevertheless legal and is present in the
JSONL trajectory.

## Run the experimental AMDGPU neural optimizer

The current AMDGPU client implements iterative complete-schedule search with a
frozen C33 shared neural model.  It is experimental, is restricted to `gfx950`,
and is disabled by default.

First export a compatible PyTorch checkpoint to LLVM's native model format:

```console
$ python3 llvm/utils/export-amdgpu-prera-nn.py \
    /path/to/NN_SHARED_MEDIUM_TRITON_L0_seed221202.pt \
    /path/to/model.amdprann
```

The exporter requires Python, NumPy, and PyTorch.  LLVM itself does not.  The
native blob contains the normalization data, dense-network parameters, source
checkpoint SHA-256, and feature-schema SHA-256.

Then invoke `llc`:

```console
$ build/bin/llc input.ll -o output.s \
    -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 \
    -amdgpu-learned-prera-sched \
    -amdgpu-learned-prera-model=/absolute/path/to/model.amdprann \
    -amdgpu-learned-prera-budget=131072 \
    -amdgpu-learned-prera-seed=2212
```

The budget is the maximum number of statically scored candidates.  The seed
controls deterministic proposal generation; it does not start multiple search
chains.  `-amdgpu-learned-prera-region=N` can restrict an investigation to one
scheduler region.  If the model is absent, invalid, incompatible, or the target
is not `gfx950`, the production schedule is preserved.

This search performs no hardware measurements.  It extracts 22 complete-state
features and 55 parent/action/child features, runs native float32 inference,
and iteratively retains promising complete schedules.

## Training and hardware-in-the-loop ownership

The API and AMDGPU endpoint generate training data, but they are not a training
framework.  Model training remains an external step.  A model whose input
schema or label semantics change must be retrained and exported with a matching
schema identifier; merely moving the same optimizer behind this API does not
require retraining.

Hardware-in-the-loop search requires an external controller.  The file-based
AMDGPU endpoint is the compiler/controller protocol for a process-per-candidate
workflow, but LLVM does not launch GPU workloads, feed runtime measurements
back to the search, or implement empirical simulated-annealing acceptance.

A hardware-in-the-loop system can use this API as its compiler-side endpoint:

1. Record the scheduling region and founder.
2. Have LLVM generate a legal endpoint, or have the controller submit a
   previously recorded complete order.
3. Recompile/replay each candidate through the training endpoint.
4. Run and measure the candidate externally.
5. Feed the runtime to the persistent controller, which chooses the next
   proposal.
6. Replay the controller's selected order for the final compilation.

The controller can remain a normal script launching one compiler process per
candidate.  A persistent compiler worker or RPC transport could reduce startup
cost, but is an optional optimization rather than a requirement.

### Basic hardware-in-the-loop SA controller

`llvm/utils/amdgpu-prera-hitl-sa.py` is a small reference controller for the
process-per-candidate workflow.  It records the post-MaxOccupancy founder,
generates exact-depth legal relocation walks, verifies every selected endpoint
by replaying it through a second compiler invocation, and performs serial
Metropolis acceptance using measured hardware runtime.

The controller deliberately does not know how to launch a particular kernel.
The command after `--benchmark-command` is supplied by the experiment and must
print one JSON object:

```json
{"baseline_runtime_us": 10.0, "candidate_runtime_us": 9.5, "correct": true}
```

The benchmark should measure the frozen founder and candidate in the same
invocation, preferably interleaved.  The annealing energy is the normalized
ratio `candidate_runtime_us / baseline_runtime_us`, which reduces sensitivity
to clock and thermal drift between evaluations.  The command supports
`{baseline}`, `{candidate}`, `{function}`, `{region}`, and `{output_dir}`
placeholders and is executed directly without a shell.

For example:

```console
$ export ROCR_VISIBLE_DEVICES=7
$ python3 llvm/utils/amdgpu-prera-hitl-sa.py \
    --llc build/bin/llc \
    --input kernel.ll \
    --function matmul_kernel \
    --region 6 \
    --output-dir hitl-sa-run \
    --budget 16 \
    --depths 1,2,4,8 \
    --seed 17 \
    --benchmark-command python3 benchmark.py \
      --baseline '{baseline}' --candidate '{candidate}'
```

The output contains a frozen configuration, append-only `search.jsonl`, a
summary, all compiler-emitted trajectories, and copies of the final current
and best schedules.  The compiler-side trajectory has structural features and
pressure observations; `search.jsonl` adds the hardware measurements and
accept/reject decisions needed to construct supervised datasets later.

### Cortex-A53 pre-RA search over SSH

AArch64 provides an experimental pre-RA record/replay and mutation endpoint
for `cortex-a53`.  The corresponding controller keeps compilation and SA state
on the host while a benchmark command measures objects on the target board.
This places search after the normal AArch64 pre-RA scheduling strategy has
produced a complete founder schedule and before physical-register allocation.

`llvm/utils/aarch64-hitl-remote.py` caches objects by SHA-256 on the remote
machine, pins the supplied benchmark to one CPU, and records frequency,
temperature, and Raspberry Pi throttling telemetry around each invocation.
Only times reported by the remote benchmark enter the SA objective; SSH and
file-transfer latency do not.

The reference native runner expects each object to export
`uint64_t hitl_kernel(uint64_t seed, uint64_t iterations)`.  Install
`aarch64-hitl-runner.c` and `aarch64-hitl-link-and-run.py` on the board, then
use the remote adapter as the controller's benchmark command:

```console
$ python3 llvm/utils/aarch64-prera-hitl-sa.py \
    --llc build/bin/llc --input kernel.ll --function hitl_kernel --region 0 \
    --output-dir a53-hitl --budget 16 --depths 1,2,4,8 --seed 17 \
    --llc-arg=-relocation-model=pic \
    --benchmark-command python3 llvm/utils/aarch64-hitl-remote.py \
      --host doge@pi3doge --baseline '{baseline}' --candidate '{candidate}' \
      --remote-command /path/to/aarch64-hitl-link-and-run.py \
        --baseline '{remote_baseline}' --candidate '{remote_candidate}' \
        --runner /path/to/aarch64-hitl-runner
```

The remote command must compare founder and candidate in one invocation and
print the same benchmark JSON consumed by the generic controller.  The
reference runner uses `CLOCK_MONOTONIC_RAW`, warmups, alternating execution
order, median runtimes, and output equality checking.  A currently throttled
measurement is recorded and retried rather than used for acceptance.

## Facilities intentionally not supplied by the base API

The generic interface does not provide:

- a global registry or command-line enum for choosing a search algorithm;
- a generic feature schema, objective, model format, or inference engine;
- hardware execution and runtime feedback.

The AMDGPU client supplies schedule serialization and JSONL trajectory output,
while hardware execution and labels remain external.  Other targets can expose
their own recorders and construct the corresponding
`MachineSchedCompleteScheduleOptimizer`.  If several targets later need the
same serialization facility, it can be promoted independently without
coupling the complete-order representation to one search method.

## Correctness checklist

Before enabling a new optimizer by default, verify all of the following:

- every result is a complete permutation of `[0, Region.size())`;
- every result passes `Region.isLegalOrder()`;
- target constraints missing from strong DAG edges have an explicit validator;
- the same DAG mutations are installed in training and deployment;
- the founder scheduler and scheduling phase match those used to collect data;
- liveness, lane-mask, pressure, occupancy, and spill-sensitive behavior are
  checked for the target;
- invalid models and rejected candidates preserve the founder exactly; and
- final schedules pass `-verify-machineinstrs` and target CodeGen tests.
