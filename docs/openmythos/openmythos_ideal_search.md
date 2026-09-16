# Searching for the ideal inference-time reasoning mechanism

## Definition of ideal

The ideal OpenMythos extension should satisfy all five constraints:

1. It runs on an unchanged pretrained checkpoint.
2. It adds no persistent model parameters.
3. Extra compute changes the *decision process*, not merely confidence.
4. It has a verifier that can reject bad internal trajectories.
5. Its cost and memory are bounded and measurable.

Anything that only repeats a block, lowers entropy, or samples more answers is not sufficient.

## Fundamental limit

Inference-time compute cannot add missing facts. It can improve:

```text
composition, search, planning, error correction, constraint satisfaction
```

It cannot reliably improve:

```text
unknown factual recall, tokenizer knowledge, missing languages, missing domains
```

Therefore the target is not “a smaller model becomes a larger model everywhere”. The strongest
valid target is “a smaller model reaches a larger model's reasoning efficiency on tasks whose
required facts are already represented in its weights”.

## Why the obvious approaches are not ideal

### More loops

```text
h -> F(h) -> F(F(h))
```

Failure: one trajectory, repeated error, diminishing returns, no correctness signal.

### More branches

```text
h -> {h1,h2,h3}
```

Failure: branches are not meaningful unless generated from a semantic uncertainty signal. A
verifier based only on agreement rewards shared mistakes.

### Entropy minimization

Failure: confidence is not truth. This can turn uncertainty into hallucinated certainty.

### Beam search in hidden space

Failure: it often becomes ordinary beam search in disguise, paying KV/cache costs without a new
reasoning mechanism.

### Alternating arbitrary model layers

Failure: pretrained layers have positional roles. Moving them into a recurrent loop changes their
distribution and can skip or duplicate semantic stages.

## Candidate 6: Constraint-carrying latent state (CCLS)

The best direction is not free-form branching. It is to separate a temporary state into:

```text
proposal state   p  — what the model currently wants to answer
constraint state  c  — conditions that must remain true
error state       r  — detected conflicts/uncertainty
```

Each loop performs three operations using the same frozen model:

```text
p' = Reason(p, c)
r' = Difference(p', c)
c' = Preserve(c, r')
```

The key invariant is that the original anchor and extracted constraints are never overwritten by
the proposal. This directly addresses OpenMythos's single-stream failure mode.

## Inference-only implementation approximation

We cannot safely invent semantic projections for an arbitrary checkpoint. Therefore start with
three *views* of the same hidden vector, not three new learned modules:

```text
c = stop_gradient(anchor)
p = h
r = h - anchor
```

At every loop:

```text
u = RMSNorm(p + c)
d = Core(u)
conflict = cosine(d, r)
gate = clamp(0.5 + 0.5*conflict, gmin, gmax)
p = RMSNorm(p + gate*d)
r = decay*r + (p - c)
```

This is only a prototype. The actual useful version should use model-derived signals rather than
an arbitrary cosine gate.

## Candidate 7: Counterfactual latent verification (CLV)

This is the strongest practical alternative to ordinary self-consistency.

1. Run a base latent trajectory to get proposal `p`.
2. Create one counterfactual by removing or rotating the highest-uncertainty direction.
3. Run the same compute budget on the counterfactual `p_cf`.
4. Compare final output distributions and constraint preservation.
5. Accept only changes that survive the counterfactual.

```text
proposal       -> answer distribution p
counterfactual -> answer distribution q
verified score = agreement(p,q) + stability - contradiction
```

This tests whether the answer is robust to a controlled latent perturbation. It is not proof of
truth, but it is a better error detector than raw confidence.

### Main risk

The perturbation may be unrelated to the real error. This is why the perturbation must be derived
from top-k output/embedding directions or hidden-state principal directions, never random noise.

## Candidate 8: Deliberate compute allocation by disagreement

Instead of choosing a fixed `T`, spend compute where the model is internally unstable:

```text
initial branch disagreement -> allocate more loops
stable branch              -> stop
```

For a batch of latent states, define:

```text
D_t = mean pairwise cosine distance of normalized hidden states
U_t = KL(mean logits || branch logits)
```

Do not interpret high disagreement as an answer. Interpret it only as a signal to allocate more
compute. This avoids the invalid assumption that confidence equals correctness.

## Candidate 9: Reversible latent search

Keep checkpoints of latent state:

```text
h0, h1, h2, ...
```

If a later verification score degrades, revert to the best previous state and explore a different
direction. This is cheap in memory for one token and prevents monotonic accumulation of bad
updates.

```text
best = argmax(score(h_t))
if score(h_(t+1)) < score(best) - margin:
    h = best
```

This is a missing capability in ordinary recurrence: loops need not be irreversible.

## Synthesis: the ideal minimal system

The best balance is a three-part controller:

```text
1. proposal recurrence       — generate a deeper latent solution
2. counterfactual verifier   — test robustness, not just confidence
3. reversible budget control — continue, revert, or stop
```

Call this **Robust Latent Deliberation (RLD)**.

### RLD loop

```text
anchor e
h = e
best = h
best_score = -inf

for t in 1..Tmax:
    h = Core(RMSNorm(h + e))
    score = verify(h, counterfactual(h,e))

    if score > best_score:
        best, best_score = h, score
    elif score < best_score - margin:
        h = best

    if score >= target and delta_output < epsilon:
        break

return best
```

The counterfactual should be cheap and temporary. Persistent KV receives only the final `best`.

## Why RLD is preferable to the other ideas

| Property | More loops | Branching | CCLS | RLD |
|---|---:|---:|---:|---:|
| unchanged checkpoint | yes | yes | yes | yes |
| catches trajectory failure | no | partly | partly | yes |
| avoids confidence trap | no | no | partly | better |
| reversible | no | optional | no | yes |
| bounded compute | yes | harder | yes | yes |
| first prototype complexity | low | medium | medium | medium |

RLD is still not guaranteed to establish truth. Its claim is narrower and testable: robust latent
trajectories should outperform equal-budget irreversible recurrence on tasks with verifiable
structure.

## First falsifiable experiment

Do not modify the public API first. Build an offline scalar simulator and then a llama.cpp
prototype behind an experimental flag.

Compare at equal core calls:

```text
baseline:        T=1
recurrence:      T=8, single trajectory
RLD:             T=8, one counterfactual at steps 2,4,6
RLD-reversible:  T=8, with checkpoint/revert
```

Tasks must include automatically verifiable answers:

```text
GSM8K/MATH subset with exact numeric answer
MBPP/HumanEval tests
synthetic multi-constraint puzzles
```

Report:

```text
accuracy
calibration
trajectory score vs correctness correlation
core evaluations
tokens/sec
temporary memory
KV writes
```

Success criterion:

```text
RLD accuracy > recurrence accuracy at equal core calls,
and the gain remains after removing high-confidence but incorrect samples.
```

If RLD only increases confidence or verbosity, reject it.

## Final recommendation

The ideal path is not “add more architecture”. It is **controlled latent search with a falsifiable
verifier and rollback**. Start with RLD because it targets the precise weakness of OpenMythos — a
single irreversible trajectory — while preserving fixed weights, bounded memory, and a clean
inference-time API.
