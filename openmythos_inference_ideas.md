# OpenMythos: inference-time scaling ideas

## Goal

Keep the checkpoint size fixed while increasing useful reasoning compute at inference time.
The target is not to clone OpenMythos. OpenMythos is evidence that recurrent latent depth can
be useful; this document compares extensions that can run on an already-trained GGUF model.

The honest target is task-dependent: improve multi-step reasoning and planning without claiming
to create facts that are absent from the checkpoint.

## Common accounting

Let `N` be the number of transformer blocks and `R` the number of additional core executions.
The persistent parameter count stays `P`. A method is attractive when it improves quality per
additional core execution and does not corrupt the real KV cache.

All candidates below must obey:

```text
temporary reasoning state -> no persistent KV writes
final selected state       -> one normal coda/KV update
```

## Idea 1: Adaptive latent depth (ALD)

Run the existing recurrent core for a variable number of iterations. Stop when both hidden
state movement and output distribution movement are small:

```text
delta_h = ||h_t - h_(t-1)|| / ||h_(t-1)||
delta_p = KL(p_t || p_(t-1))
stop if delta_h < eh and delta_p < ep
```

### Advantages

- Smallest change and easiest ablation.
- Works with existing single-stream recurrence.
- Saves compute on easy inputs.
- No branch management or extra model weights.

### Risks

- Stable does not mean correct: a wrong answer can converge.
- Logits at every iteration are expensive.
- A token-local halt may break a multi-token reasoning chain.
- No new search capability; it only allocates depth.

### Cost estimate

For mean loop count `Tmean`, core cost is approximately `Tmean / Tmax` of fixed-depth mode.
Additional memory is `O(d)` per stream.

### Verdict

Good control experiment, weak candidate for a breakthrough. Implement early, but do not make it
the main theory.

## Idea 2: Latent branch-and-bound (LBB)

Create `B` temporary hidden hypotheses from one anchor, run each for `T` core steps, score them,
prune the worst half, and merge or continue the survivors.

```text
e -> {h1, h2, ..., hB}
each branch: h_i <- Core(h_i + e)
score: confidence + stability + agreement - oscillation
prune -> continue -> select/merge
```

### Advantages

- Adds breadth, not only depth.
- Can recover when one trajectory enters a bad basin.
- Same checkpoint and no training required.
- Directly tests whether latent alternatives are useful.

### Risks

- Perturbations may be arbitrary and meaningless.
- Agreement can amplify a shared model error.
- Memory and compute scale with `B`.
- Branch merge may destroy incompatible but useful hypotheses.
- KV isolation is difficult for attention/SSM models.

### Cost estimate

At fixed core budget `C`, choose `B*T=C`. For example:

```text
B=1,T=8: 8 core evaluations
B=2,T=4: 8 core evaluations
B=4,T=2: 8 core evaluations
```

Temporary activation memory is `O(B*d)` (or `O(B*d*sequence)` for prompt batches).

### Verdict

Most interesting inference-only hypothesis, but only if branch creation is principled. Random
noise is not enough; use top-k embedding directions or hidden-space orthogonal directions and
measure whether branches actually diverge and later improve task accuracy.

## Idea 3: Test-time latent optimization (TTLO)

Treat the hidden state as an optimizable variable. Run a short forward evaluation, derive a
model-internal loss signal, and update the latent state before another forward pass.

```text
h0 -> logits
loss_proxy(h0) -> gradient/direction
h1 = h0 - eta * direction
```

Possible loss proxies: entropy, contradiction between two readouts, next-token instability,
or agreement between a forward and reverse local projection.

### Advantages

- Turns extra compute into directed refinement rather than blind looping.
- Could correct a trajectory instead of only extending it.
- State remains temporary; parameter count is unchanged.

### Risks

- Entropy minimization rewards confident nonsense.
- Gradients through quantized inference may be noisy or unavailable.
- A useful loss proxy requires a real verifier; otherwise this is self-deception.
- Backpropagation at every token is likely too slow.

### Cost estimate

One optimization step may require one forward plus one backward-like graph, roughly `2–4x` a
normal core evaluation and substantial activation memory.

### Verdict

High upside, high risk. Do not implement until LBB produces a measurable verifier signal. It is
an experiment for later, not the first production path.

## Idea 4: Latent speculative tree (LST)

Use top-k next-token alternatives to create hidden continuations, but do not append them to the
user-visible context. Each continuation is embedded and processed for a few latent steps; the
root branch is selected using the final model distribution.

```text
root -> token a -> latent refine
     -> token b -> latent refine
     -> token c -> latent refine
```

### Advantages

- Branches have semantic meaning: they correspond to actual model alternatives.
- More principled than random hidden perturbations.
- Naturally useful for ambiguous planning and code completion.
- Can reuse existing token embeddings and vocabulary logits.

### Risks

- It partially recreates beam search and may increase context/KV cost.
- Top-k alternatives are often lexical, not conceptual alternatives.
- Greedy branch selection can harm diversity.
- For every generated token, tree management can become expensive.
- It may improve decoding quality without increasing latent reasoning quality.

### Cost estimate

With beam width `B` and lookahead `L`, naive cost is `O(B*L)` token/core work. Prefix sharing
can reduce this, but attention cache isolation remains implementation-heavy.

### Verdict

Strong practical candidate for coding/planning, weaker universal reasoning candidate. Consider
after LBB because its branch semantics are easier to validate.

## Idea 5: Multi-timescale memory / recurrent workspace (MTW)

Maintain several temporary states with different update rates instead of one state:

```text
fast  state: updated every loop
mid   state: updated every 2 loops
slow  state: anchor/global constraints
h = gated_read(fast, mid, slow)
```

The same model block reads a concatenation or gated mixture of these states. This is a latent
working memory, not a new parameter store.

### Advantages

- Addresses a real weakness of one-stream recurrence: overwriting old constraints.
- Better fit for long planning and multi-constraint problems.
- Can be implemented with cheap tensor operations around the existing core.
- Less arbitrary than branch noise.

### Risks

- Without training, the model may not interpret separated states correctly.
- Mixing states can destroy normalization statistics.
- It adds memory but not necessarily useful information.
- Hard to compare fairly with ordinary recurrence.

### Cost estimate

With `K` state streams and one shared core, core FLOPs stay near `T` only if states are mixed
before one core call; independent processing costs `K*T`. Temporary memory is `O(K*d)`.

### Verdict

Best architectural direction if we can add a tiny learned adapter. For strict no-training
inference, use it only as a controlled experiment after ALD.

## Decision matrix

Scores are hypotheses, not measured results. 1 = poor, 5 = strong.

| Idea | Novelty | No-training viability | Verification ease | Compute efficiency | Main failure |
|---|---:|---:|---:|---:|---|
| ALD | 2 | 5 | 5 | 5 | converged wrong answer |
| LBB | 5 | 4 | 3 | 3 | meaningless branches/shared error |
| TTLO | 5 | 2 | 2 | 1 | optimizing confidence, not truth |
| LST | 4 | 4 | 4 | 2 | expensive beam search |
| MTW | 4 | 2 | 3 | 4 | untrained state semantics |

## Selected path

Do not start with the most exotic idea. Use a staged path that kills bad hypotheses quickly:

1. **ALD baseline:** establish whether additional latent depth helps and find saturation.
2. **LBB-2:** two branches, equal total core calls to ALD. Use deterministic top-k embedding
   directions, not random noise. Keep persistent KV untouched.
3. **Verifier ablation:** compare hidden cosine, logit KL, stability, and combinations.
4. **MTW-lite:** add a slow anchor state only if LBB shows branch diversity is useful.
5. **TTLO:** only if a verifier score correlates with benchmark correctness.

The first falsifiable claim is:

```text
At equal core-evaluation budget C, LBB(B=2,T=C/2) > ALD(B=1,T=C)
on multi-step tasks, without improving only confidence or verbosity.
```

If this fails across math, code, and planning, abandon branching rather than adding complexity.
If it succeeds, OpenMythos has been extended from latent depth to latent breadth with a measurable
inference-time scaling law.

## Non-negotiable measurements

- Exact baseline and variant prompts/seeds.
- Accuracy, not just perplexity or confidence.
- Branch cosine separation before and after refinement.
- Calibration: confidence versus correctness.
- Tokens/sec and peak memory.
- KV-cache bytes written during temporary reasoning.
- Quality at fixed total core evaluations.

No claim of “larger-model level” is valid until the variant closes a measured fraction of the
gap to a larger reference model on held-out tasks.
