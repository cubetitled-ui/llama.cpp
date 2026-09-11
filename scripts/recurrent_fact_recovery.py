"""Toy test: can recurrent refinement recover a distributed fact after compression?

This is not an LLM result. It isolates information loss from iterative computation.
"""
import argparse
import numpy as np


def make_data(rng, n, dim, fact_dim):
    # A fact is distributed across many coordinates via a random codebook.
    facts = rng.integers(0, 2, size=(n, fact_dim)).astype(np.float32)
    code = rng.normal(size=(fact_dim, dim)).astype(np.float32)
    x = facts @ code + 0.25 * rng.normal(size=(n, dim)).astype(np.float32)
    return facts, x, code


def compress(x, out_dim, rng):
    proj = rng.normal(size=(x.shape[1], out_dim)).astype(np.float32) / np.sqrt(out_dim)
    return x @ proj, proj


def run(args):
    rng = np.random.default_rng(args.seed)
    facts, x, code = make_data(rng, args.samples, args.dim, args.fact_dim)
    z, proj = compress(x, args.bottleneck, rng)

    # Linear decoder is deliberately trained only on the compressed representation.
    train = np.arange(args.samples // 2)
    test = np.arange(args.samples // 2, args.samples)
    reg = args.reg * np.eye(args.bottleneck, dtype=np.float32)
    decoder = np.linalg.solve(z[train].T @ z[train] + reg, z[train].T @ facts[train])

    def accuracy(state):
        pred = (state @ decoder > 0.5).astype(np.float32)
        return float(np.mean(pred == facts[test]))

    # A recurrent map can only help if it has a useful learned/known operator. Here it is a
    # denoising fixed-point step using the same decoder and the compressed observation as anchor.
    state = z[test].copy()
    baseline = accuracy(state)
    print(f"bottleneck={args.bottleneck} information_ratio={args.bottleneck/args.dim:.3f} baseline={baseline:.4f}")
    for t in range(args.steps):
        logits = state @ decoder
        hard = (logits > 0.5).astype(np.float32)
        # Project the reconstructed fact back into the compressed latent space. This is a
        # legitimate iterative denoiser, not a claim that missing bits can be recreated.
        state = (1.0 - args.gate) * state + args.gate * (hard @ decoder.T)
        print(f"step={t+1} accuracy={accuracy(state):.4f} latent_norm={np.mean(np.linalg.norm(state, axis=1)):.4f}")

    # Upper bound: decode directly from uncompressed x with a least-squares decoder.
    full_reg = args.reg * np.eye(args.dim, dtype=np.float32)
    full_decoder = np.linalg.solve(x[train].T @ x[train] + full_reg, x[train].T @ facts[train])
    full_acc = float(np.mean(((x[test] @ full_decoder > 0.5).astype(np.float32)) == facts[test]))
    print(f"uncompressed_upper_bound={full_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--fact-dim", type=int, default=32)
    p.add_argument("--bottleneck", type=int, default=16)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--gate", type=float, default=0.25)
    p.add_argument("--reg", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=3)
    run(p.parse_args())
