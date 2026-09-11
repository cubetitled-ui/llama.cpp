"""Small falsifiable simulations for OpenMythos inference-time ideas.

These are not model-quality results. They test controller mechanics and expose optimistic
assumptions before changing llama.cpp.
"""
import math
import random
from statistics import mean


def recurrence(x, target, a, gain, noise, steps):
    xs = [x]
    for _ in range(steps):
        # A toy nonlinear residual update: useful for detecting error amplification.
        x = a * x + gain * math.tanh(target - x) + noise
        xs.append(x)
    return xs


def experiment_dynamics():
    print("[dynamics]")
    for a, gain in [(0.9, 0.1), (0.9, 0.4), (0.5, 0.4)]:
        xs = recurrence(3.0, 0.0, a, gain, 0.0, 16)
        print(f"a={a} gain={gain} final={xs[-1]:+.4f} max_abs={max(map(abs, xs)):.4f}")


def experiment_branching(seed=7, n=10000):
    random.seed(seed)
    # Latent branches are useful only when their errors are not perfectly correlated.
    # q is probability that a branch independently corrects a hard binary decision.
    print("[branching: oracle selection]")
    for correlation in (0.0, 0.25, 0.5, 0.75, 1.0):
        one = two = 0
        for _ in range(n):
            shared = random.random() < correlation
            e1 = (shared and random.random() < 0.75) or random.random() < 0.25
            e2 = (shared and random.random() < 0.75) or random.random() < 0.25
            one += not e1
            # An oracle verifier succeeds if either branch is correct.
            two += not (e1 and e2)
        print(f"requested_corr={correlation:.2f} one={one/n:.3f} oracle_two={two/n:.3f}")


def experiment_rollback(seed=11, n=10000):
    random.seed(seed)
    # A verifier with false-positive rate fp can make rollback actively harmful.
    print("[rollback: noisy verifier]")
    for fp in (0.0, 0.1, 0.3, 0.5):
        kept = 0
        for _ in range(n):
            base = random.random() < 0.5
            improved = random.random() < 0.55
            score_base = base or random.random() < fp
            score_new = (improved) or random.random() < fp
            kept += (improved if score_new else base)
        print(f"false_positive={fp:.2f} kept_correct={kept/n:.3f}")


def experiment_budget():
    print("[budget accounting]")
    # Equal core calls do not imply equal memory: branches need isolated state.
    for branches, steps in [(1, 8), (2, 4), (4, 2), (8, 1)]:
        print(f"branches={branches} steps={steps} calls={branches*steps} scratch_units={branches}")


if __name__ == "__main__":
    experiment_dynamics()
    experiment_branching()
    experiment_rollback()
    experiment_budget()
