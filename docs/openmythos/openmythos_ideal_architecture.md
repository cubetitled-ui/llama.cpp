# Toward an ideal inference-time intelligence system

## The question

Can a fixed pretrained GGUF become substantially more capable by spending more inference-time
compute, without training, parameter growth, or persistent KV pollution?

The answer is **yes for computation, search, composition, planning, and error correction**; it is
**not guaranteed for missing factual information**. The ideal system must therefore increase the
model's ability to use its information, not pretend to add information that is absent.

## Do not confuse three kinds of independence

“Independent hypotheses” can mean three different things:

1. **Geometric independence**: hidden vectors are far apart.
2. **Semantic independence**: branches represent different plans or explanations.
3. **Epistemic independence**: branches have different error causes and do not share the same
   unsupported assumption.

Only the third is useful for correctness. Random noise guarantees none of them. Token top-k gives
mostly lexical diversity. Prompt rewriting gives semantic diversity but costs context and is not
latent-only.

The ideal system must generate diversity by changing the *reasoning operator*, not just the
initial vector:

```text
branch A: derive from premises
branch B: search for a counterexample
branch C: construct an independent solution
branch D: audit constraints and boundary cases
```

This suggests a **role-conditioned controller**, but the roles must be induced from existing model
capabilities or a small external verifier; inventing arbitrary hidden directions is insufficient.

## The universal verifier paradox

A universal external correctness signal for arbitrary open-ended text does not exist. A system that
claims one is hiding an assumption. The ideal solution is a verifier *stack*:

```text
level 0: executable/task verifier (strong)
level 1: retrieval/evidence verifier (strong when sources exist)
level 2: symbolic/constraint verifier (strong for structured outputs)
level 3: independent model/view verifier (medium)
level 4: trajectory stability/calibration (weak)
```

The controller must expose the level used and attach uncertainty to the final answer. More loops
cannot upgrade a level-4 signal into a level-1 proof.

## Ideal architecture: Recursive Deliberation with Evidence and Rollback

Call the full hypothesis **RDE-R**.

```text
                         ┌──────────────┐
prompt → representation →│ role router  │
                         └──────┬───────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
        proposer            counterexample       auditor
            │                   │                   │
            └────────── temporary latent DAG ──────┘
                                │
                         evidence verifier
                                │
                    score / uncertainty / errors
                                │
                    expand, merge, rollback, stop
                                │
                         final state only
                                │
                         normal generation
```

This is not a single recurrent chain. It is a temporary **latent reasoning DAG**. Each node stores:

```text
state: hidden representation
provenance: parent and operation used
claims: structured assertions extracted from the state
score: verifier result and uncertainty
```

The model weights remain fixed. The DAG is temporary activation memory.

## Why a DAG instead of a beam

A beam only keeps alternatives. A DAG also preserves shared premises and records where branches
diverged:

```text
shared premises → independent derivations → cross-check → merged conclusion
```

This allows the controller to avoid paying again for common computation and to detect when two
branches are only lexically different but logically identical.

## How to create genuine conceptual diversity without training

Use **operator diversity** plus state diversity:

```text
O0: normal forward reasoning
O1: premise-first reconstruction
O2: contradiction/counterexample search
O3: backward goal decomposition
O4: boundary-case audit
```

An operator is not necessarily a new neural layer. It may be a different inference program around
the same layer:

```text
different attention-visible scratch views
different verifier constraints
different readout targets
different reversible update direction
```

For a ready GGUF, the controller can create operator prompts or structured latent views. The
strong version needs a small learned role adapter; the strict no-training version can use existing
token embeddings and task constraints, but must label diversity as a hypothesis rather than a fact.

## The ideal scoring function

No single confidence score is sufficient. A node score should be a vector:

```text
S(node) = (
    task_validity,
    evidence_support,
    cross_branch_independence,
    contradiction_rate,
    output_stability,
    calibration,
    compute_cost
)
```

Selection is lexicographic where possible:

```text
hard verifier failure       → reject
schema/test failure         → reject
unsupported contradiction  → penalize
independent agreement       → reward
stability                   → tie-breaker, never proof
confidence                  → last tie-breaker
```

This prevents the common mistake of allowing high-confidence hallucinations to beat a lower-
confidence verified answer.

## Compute scaling law

Let `C` be the temporary compute budget. The ideal system should have:

```text
Q(C+Δ) >= Q(C) - ε
```

because rollback preserves the best known state. Raw final-loop quality need not be monotonic; best-
so-far quality should be.

The expected quality model is saturating:

```text
Q(C) = Q∞ - a exp(-b C)
```

but `b` may differ by task and model size. Larger models should generally have higher `Q∞` and may
need fewer search expansions for the same task, while smaller models can spend more compute to
close part of the reasoning gap.

## KV and state isolation

Every DAG node needs an isolated temporary state:

```text
attention K/V: scratch arena per node or copy-on-write pages
SSM/recurrent state: scratch snapshot per node
position state: branch-local and never advanced globally
persistent KV: written exactly once after final selection
```

This is harder than simply avoiding attention-cache writes. Hybrid architectures make the native
recurrent state part of the same isolation contract.

## What “universal across 7B and 120B” really means

The controller must be dimension-agnostic:

```text
all thresholds use normalized values
all budgets are expressed in core evaluations
verifier adapters operate on decoded/structured outputs
no fixed hidden-size assumptions
```

The model-size advantage is not only compensation. A 120B model's branches should be more useful
because each branch contains a stronger representation. The same search controller can therefore
turn extra compute into better results more efficiently at scale:

```text
branch quality(120B) > branch quality(7B)
verification reliability(120B) > verification reliability(7B)
```

This is the mechanism behind the desired “120B can reason above ordinary 120B” effect, without
claiming that it has acquired 199B's missing facts.

## The ideal is possible, but not from a plain loop alone

The theoretical requirements are compatible:

```text
fixed weights             ✓
temporary latent compute  ✓
conceptual alternatives   ✓, via role/operator diversity
external checking        ✓, via verifier stack
predictable scaling       ✓, with best-state rollback
KV isolation              ✓, with branch-local state
model-size universality   ✓, with normalized controller
```

The impossible version is:

```text
arbitrary open-ended correctness
from one frozen model's logits alone
```

That is not an engineering gap; it is an information/verifiability limit.

## Ideal research program

Do not prematurely collapse RDE-R into one simple feature. Test its assumptions in order:

1. Can operator diversity produce complementary, not merely distant, branches?
2. Does a verifier stack correlate with correctness better than confidence?
3. Does rollback make best-so-far quality monotonic with compute?
4. Does a latent DAG beat a chain at equal total core evaluations?
5. Does the gain increase with model size, as the strong-model branch hypothesis predicts?
6. Does scratch state remain isolated for attention and SSM models?

The ideal is not “one magic recurrence formula”. It is a **closed-loop inference computer**:

```text
propose → diversify → verify → preserve evidence → revise → commit
```

OpenMythos supplies the recurrence substrate. RDE-R adds search, epistemic separation, evidence,
and reversible control. That is the path from a looped transformer to an inference-time reasoning
system rather than a clone.
