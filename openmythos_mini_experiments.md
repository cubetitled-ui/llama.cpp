# Mini-experiment findings

Run:

```text
python3 scripts/openmythos_mini_experiments.py
```

## Confirmed mechanics

1. A stable scalar recurrence can converge in a toy nonlinear system. This does not prove that a
   transformer recurrence is stable; `|A| < 1` alone is insufficient once the nonlinear block is
   included.
2. Independent branches can provide oracle gains. In the toy run, the gain fell as branch errors
   became correlated: branch independence is the real resource.
3. Noisy rollback loses value as verifier false-positive rate rises. A verifier must be measured
   against correctness before it controls search.
4. Equal core-call budgets do not imply equal memory: scratch memory grows with branch count.

## What was wrong in earlier reasoning

- “More loops always help” was wrong. They can converge to the wrong attractor.
- “A stable A is enough” was wrong. Stability of the LTI part does not bound the full Jacobian.
- “Two branches are better” was incomplete. They help only when their errors are complementary and
  a verifier can select correctly.
- “Rollback guarantees improvement” was wrong. Rollback is only as good as its score; noisy scores
  can reduce correctness.
- “Equal FLOPs means a fair branch/depth comparison” was incomplete. Branching has a memory cost
  and cache-isolation overhead that must be reported separately.

## What remains untested

This script does not test a real language model. It cannot establish quality, conceptual branch
diversity, or 6 GB VRAM behavior. The next experiment must use a real GGUF and objective tasks.

## Consequence for the ideal design

The ideal system cannot be only a recurrence or only a branch search. It requires:

```text
diversity source whose complementarity is measurable
verifier whose correctness correlation is measured
best-state rollback
strict temporary-state isolation
```

The first implementation should therefore expose instrumentation before adding complexity:

```text
branch separation
verifier score
correctness outcome
rollback decisions
scratch memory
KV writes
```
