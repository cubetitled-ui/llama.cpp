# Criteria review: what an ideal inference-only system can actually guarantee

## 1. No training

This is compatible with a ready GGUF only if the controller uses signals already exposed by the
model or by the task. It must not assume that an arbitrary hidden vector has a learned semantic
meaning after an invented projection.

Safe operations:

```text
existing model forward
existing embedding/unembedding
token probabilities
hidden-state norms and cosine distances
deterministic transformations
external task checks when available
```

Unsafe as a first principle:

```text
new latent slots with invented semantics
random hidden perturbations
untrained gates treated as reasoning modules
gradient descent on entropy
```

## 2. Uncorrelated hypotheses

This is the hardest criterion. Lexical top-k alternatives are not conceptual alternatives. Two
different tokens can express the same plan, and two identical first tokens can lead to opposite
plans.

Without training, the most defensible source of conceptual diversity is **input decomposition**:

```text
branch A: original prompt
branch B: prompt facts/constraints isolated
branch C: prompt with an explicit counterfactual instruction
branch D: prompt with an explicit verification instruction
```

These branches are semantically different, but they normally require extra prompt tokens and
therefore ordinary KV/context work. A latent-only equivalent cannot honestly claim conceptual
diversity unless it is validated against an external task signal.

Conclusion: replace the absolute requirement “uncorrelated hypotheses” with a measurable one:

```text
branch diversity must predict complementarity on held-out tasks,
not merely cosine distance or different next tokens.
```

## 3. External verification signal

There is no universal external verifier for arbitrary text generation. This criterion must be
typed by task:

```text
code       compiler/tests
math       exact answer/parser/calculator
tool use   tool result/schema validation
retrieval  source evidence/entailment check
free text  calibration/consistency only (weak)
```

For a pure language model with no tools or answer labels, “external” cannot be manufactured from
the same logits. Self-consistency is useful, but it is not external verification.

Therefore the universal system needs a **verifier interface**, not one universal verifier. When no
verifier is available it must report reduced confidence, not pretend to have checked correctness.

## 4. Compute scaling

“More compute = better” is too strong. Any iterative system can overshoot or amplify errors.
The realistic requirement is:

```text
quality improves on average with budget up to a task/model-dependent saturation point,
with a bounded fallback to the best earlier state.
```

Rollback is essential. Keep the best state according to the verifier; never force the final loop
to replace an earlier better state.

## 5. KV isolation

This is fully achievable and should be an invariant:

```text
temporary branch state -> scratch graph/cache only
accepted final state   -> normal model memory once
```

Do not reuse the normal autoregressive KV cache for branches. For hybrid SSM models, temporary
state isolation must include native recurrent/SSM state, not only attention K/V.

## 6. Model-size universality

The controller should use normalized, dimension-independent signals:

```text
RMS-normalized hidden states
normalized logit margins
KL divergence of distributions
relative state deltas
```

Budget should scale with difficulty and available compute, not parameter count alone. Larger models
may need less compensation for missing representation but can exploit more loops because their
recurrent transformation is more capable. This is an empirical scaling law, not a guarantee.

## Simplification: separate the universal core from optional verifiers

The ideal design should have only three universal operations:

```text
1. proposal: run a temporary latent trajectory
2. score: ask a verifier for a scalar score
3. controller: continue, branch, rollback, or stop
```

Do not start with latent slots, loop-index embeddings, arbitrary layer routing, or gradient-based
optimization. They add assumptions before the basic loop/search effect is measured.

## Recommended architecture: Verifier-Guided Latent Search (VGLS)

```text
normal model prefix -> anchor e
                         |
                  scratch trajectory
                         |
              proposal -> verifier -> score
                         |
              continue / branch / rollback
                         |
                  best accepted state
                         |
                   normal model suffix
```

The recurrent computation is only a search proposal. Correctness comes from the verifier where one
exists. This prevents the architecture from making a false universal claim.

### Budget schedule

Use a geometric schedule rather than a fixed arbitrary loop count:

```text
B = 1, 2, 4, 8 ...
```

At each budget checkpoint:

```text
score(new) >= score(best) + margin -> accept
otherwise                         -> keep best / stop
```

This makes compute scaling observable and prevents degradation after extra loops.

### Branch creation

Do not claim latent branches are conceptual until tested. Start with two controlled proposals:

```text
proposal 0: ordinary trajectory
proposal 1: verifier-conditioned alternative
```

For code/math, the alternative can be generated by a task-specific verifier prompt or by a
different top-level decoding constraint. For pure text, use no branching initially because there
is no reliable external signal to select it.

## What is genuinely universal

The universal part is not “the model generates independent thoughts”. It is:

```text
the same model can spend a variable amount of temporary compute,
preserve the best verified intermediate state,
and commit only once to the real generation memory.
```

That is implementable for 7B and 120B without training. Conceptual branch diversity and correctness
verification are adapters supplied by the task, not properties that can be guaranteed from GGUF
alone.

## Final decision

Simplify the first implementation to:

```text
VGLS = temporary recurrence + verifier callback + rollback + KV isolation
```

First supported verifiers:

```text
exact numeric verifier
process exit/test verifier
JSON/schema verifier
```

Only after this works should we add a generic weak verifier for unconstrained language. This order
minimizes the most dangerous failure mode: a system that sounds more confident while becoming less
correct.
