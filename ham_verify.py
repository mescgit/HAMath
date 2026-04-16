"""
HAMesh Conjecture Verifier

Takes the output of ham_scholar.py (conjectures.json) and checks whether
the gap regions the mesh found correspond to real known mathematics.

Method:
  1. Re-diffract from each conjecture's seed to recover the gap vector
  2. Embed a "verification corpus" of bridge theorems NOT in the training mesh
  3. Find which verification theorems land closest to each gap vector
  4. If sim > VERIFY_THRESHOLD: the gap is real -- the mesh found its absence

The verification corpus is a curated set of theorems that are known to
bridge the domains represented in BUILTIN_CORPUS. These were deliberately
excluded from the training corpus so the verification is clean.

Usage:
    python ham_verify.py --mesh math_mesh_combined.pt --conjectures conjectures_combined.json
    python ham_verify.py --mesh math_mesh.pt --conjectures conjectures.json --top 5
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ham_core import HolographicMesh
from ham_embedder import Embedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERIFY_THRESHOLD = 0.38   # sim to gap vector that counts as "verified"


# ---------------------------------------------------------------------------
# Verification corpus: bridge theorems held out from training
# These are real, known theorems that connect the domains in BUILTIN_CORPUS.
# Deliberately NOT included in ham_corpus.py BUILTIN_CORPUS.
# ---------------------------------------------------------------------------

VERIFICATION_THEOREMS = [
    # --- Analysis <-> Number Theory bridges ---
    ("prime number theorem",
     "The number of primes up to n is asymptotically n divided by ln(n). "
     "Proved using complex analysis and the Riemann zeta function.",
     "analytic number theory"),

    ("riemann zeta function",
     "The Riemann zeta function zeta(s) is the analytic continuation of the "
     "sum 1/n^s. Its zeros control the distribution of prime numbers.",
     "analytic number theory"),

    ("dirichlet theorem on primes",
     "There are infinitely many primes in any arithmetic progression a + nd "
     "where gcd(a,d)=1. Proved using Dirichlet L-functions (complex analysis).",
     "analytic number theory"),

    ("p-adic numbers",
     "The p-adic numbers define a metric on the rationals where two numbers "
     "are close if their difference is divisible by a high power of p. "
     "They give a topology where divisibility defines distance.",
     "p-adic analysis"),

    ("p-adic absolute value",
     "The p-adic absolute value |n|_p = p^(-v) where v is the p-adic valuation "
     "bridges number theory and analysis via ultrametric topology.",
     "p-adic analysis"),

    ("ostrowski theorem",
     "Every non-trivial absolute value on the rationals is equivalent to either "
     "the standard absolute value or a p-adic absolute value.",
     "p-adic analysis"),

    # --- Topology <-> Algebra bridges ---
    ("fundamental group",
     "The fundamental group of a topological space captures its loop structure. "
     "It is an algebraic invariant: two homeomorphic spaces have isomorphic "
     "fundamental groups.",
     "algebraic topology"),

    ("seifert van kampen theorem",
     "The fundamental group of a space built from two overlapping pieces equals "
     "the free product of their fundamental groups amalgamated over the intersection.",
     "algebraic topology"),

    ("homology groups",
     "Homology assigns a sequence of abelian groups to a topological space, "
     "measuring n-dimensional holes. It is a functor from topology to algebra.",
     "algebraic topology"),

    ("de rham theorem",
     "The de Rham cohomology of a smooth manifold (defined using differential "
     "forms) is isomorphic to its singular cohomology (defined topologically).",
     "algebraic topology"),

    ("brouwer degree",
     "The degree of a continuous map between spheres is an integer topological "
     "invariant that generalises the winding number.",
     "algebraic topology"),

    # --- Analysis <-> Topology bridges ---
    ("ascoli theorem",
     "A sequence of equicontinuous, uniformly bounded functions on a compact "
     "space has a uniformly convergent subsequence. Bridges compactness and analysis.",
     "functional analysis"),

    ("baire category theorem",
     "A complete metric space cannot be written as a countable union of "
     "nowhere-dense sets. Compactness meets topology meets analysis.",
     "functional analysis"),

    ("open mapping theorem",
     "A surjective bounded linear operator between Banach spaces is an open map. "
     "Connects functional analysis and topology.",
     "functional analysis"),

    ("hahn banach theorem",
     "Every bounded linear functional on a subspace of a normed vector space "
     "can be extended to the whole space without increasing its norm.",
     "functional analysis"),

    ("spectral theorem functional calculus",
     "A normal operator on a Hilbert space is unitarily equivalent to "
     "multiplication by a function on an L2 space. Bridges linear algebra, "
     "topology, and measure theory.",
     "functional analysis"),

    # --- Analysis theorems that unify EVT / MVT / IVT ---
    ("extreme value theorem compact spaces",
     "A continuous real-valued function on a compact topological space attains "
     "its maximum and minimum. Generalises the extreme value theorem from "
     "closed intervals to arbitrary compact spaces.",
     "topology-analysis bridge"),

    ("fixed point theorem banach",
     "A contraction mapping on a complete metric space has a unique fixed point. "
     "Generalises and unifies Brouwer, intermediate value, and mean value theorems "
     "via the structure of complete metric spaces.",
     "fixed point theory"),

    ("intermediate value property",
     "A connected topological space has the intermediate value property: "
     "any continuous function takes all values between any two of its values. "
     "Unifies IVT with connectedness.",
     "topology-analysis bridge"),

    ("rolle theorem generalisation",
     "Between any two zeros of a differentiable function there is a zero of "
     "its derivative. Special case of the mean value theorem, which is itself "
     "a consequence of the extreme value theorem.",
     "analysis"),

    # --- Set Theory <-> Logic bridges ---
    ("lowenheim skolem theorem",
     "If a first-order theory has an infinite model, it has models of every "
     "infinite cardinality. Connects model theory and set theory.",
     "model theory"),

    ("completeness theorem godel",
     "A first-order sentence is provable if and only if it is true in every "
     "model. Connects syntactic provability and semantic truth.",
     "model theory"),

    ("compactness theorem logic",
     "A set of first-order sentences has a model if and only if every finite "
     "subset has a model. The logical analogue of topological compactness.",
     "model theory"),

    ("forcing cohen",
     "Cohen's forcing technique shows the continuum hypothesis is independent "
     "of ZFC. Constructs new set-theoretic universes by adding generic reals.",
     "set theory foundations"),

    ("godel constructible universe",
     "Godel's constructible universe L shows the consistency of the axiom of "
     "choice and the generalised continuum hypothesis with ZFC.",
     "set theory foundations"),

    # --- Linear Algebra <-> Number Theory bridges ---
    ("smith normal form",
     "Every integer matrix can be reduced to diagonal form over the integers "
     "by row and column operations. Bridges linear algebra and divisibility.",
     "algebraic number theory"),

    ("minkowski theorem",
     "A convex symmetric body in R^n with volume greater than 2^n contains "
     "a non-zero integer point. Bridges geometry, linear algebra, and number theory.",
     "geometry of numbers"),

    ("lattice basis reduction LLL",
     "The LLL algorithm finds a nearly-orthogonal basis for an integer lattice. "
     "Bridges linear algebra, geometry, and computational number theory.",
     "computational number theory"),

    ("representation theory",
     "A group representation is a homomorphism from a group to a group of "
     "linear transformations. Bridges group theory and linear algebra.",
     "representation theory"),

    ("character theory",
     "The characters of a finite group (traces of representations) completely "
     "determine its representation theory. Bridges algebra and analysis.",
     "representation theory"),

    # --- Information Theory <-> Computability bridges ---
    ("kolmogorov complexity",
     "The Kolmogorov complexity of a string is the length of its shortest "
     "description in a universal language. Bridges computability and information theory.",
     "algorithmic information theory"),

    ("algorithmic randomness",
     "A sequence is algorithmically random (Martin-Lof random) if it passes "
     "all effective statistical tests. Bridges probability, computability, and "
     "information theory.",
     "algorithmic information theory"),

    ("chaitin omega",
     "Chaitin's constant Omega is the halting probability of a universal "
     "prefix-free Turing machine -- an algorithmically random, uncomputable real.",
     "algorithmic information theory"),

    ("shannon source coding theorem",
     "The minimum average code length for lossless compression equals the "
     "entropy of the source. Bridges information theory and probability.",
     "information theory"),

    # --- Combinatorics <-> Algebra bridges ---
    ("polya enumeration theorem",
     "The number of distinct objects under a group of symmetries is given by "
     "the cycle index of the group. Bridges combinatorics and group theory.",
     "combinatorial algebra"),

    ("burnside lemma",
     "The number of distinct orbits of a group acting on a set equals the "
     "average number of fixed points. Connects group theory and combinatorics.",
     "combinatorial algebra"),

    ("mobius inversion",
     "The Mobius function inverts summation over divisors. Bridges "
     "combinatorics, number theory, and partially ordered sets.",
     "combinatorial number theory"),
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class ConjectureVerifier:
    """
    Tests whether a mesh's conjectures correspond to real known mathematics.

    For each conjecture:
      1. Re-diffract from its seed to recover the gap vector
      2. Compute cosine similarity to all verification theorems
      3. Report which verification theorems land near the gap
    """

    def __init__(self, mesh: HolographicMesh, embedder: Embedder):
        self.mesh = mesh
        self.embedder = embedder

        # Pre-embed all verification theorems
        print("  Embedding verification corpus...")
        self._verify_texts = [
            f"{name}: {desc}" for name, desc, _ in VERIFICATION_THEOREMS
        ]
        self._verify_domains = [d for _, _, d in VERIFICATION_THEOREMS]
        self._verify_vecs = embedder.embed_batch(self._verify_texts)
        print(f"  {len(self._verify_texts)} bridge theorems ready.")

    def verify_conjecture(self, conjecture: dict, hops: int = 2) -> dict:
        """
        Re-diffract from the conjecture's seed and find which verification
        theorems are nearest to the gap vector.
        """
        seed_text = conjecture["seed"]

        # Re-embed the seed and diffract
        seed_vec = self.embedder.embed(seed_text)
        gap_vec  = self.mesh.diffract(seed_vec, hops=hops)

        # Similarity to verification theorems
        sims = F.cosine_similarity(gap_vec.unsqueeze(0), self._verify_vecs)
        top_vals, top_idx = torch.topk(sims, min(5, len(self._verify_texts)))

        matches = []
        for i in range(len(top_vals)):
            sim  = top_vals[i].item()
            idx  = top_idx[i].item()
            matches.append({
                "sim":    round(sim, 4),
                "theorem": self._verify_texts[idx][:120],
                "domain":  self._verify_domains[idx],
                "verified": sim >= VERIFY_THRESHOLD,
            })

        best_sim     = matches[0]["sim"] if matches else 0.0
        verified     = best_sim >= VERIFY_THRESHOLD
        best_theorem = matches[0]["theorem"] if matches else ""
        best_domain  = matches[0]["domain"]  if matches else ""

        return {
            "seed":         seed_text[:80],
            "novelty":      conjecture["novelty"],
            "recurrence":   conjecture["recurrence"],
            "verified":     verified,
            "best_sim":     best_sim,
            "best_theorem": best_theorem,
            "best_domain":  best_domain,
            "top_matches":  matches,
            "nearest_in_mesh": conjecture.get("nearest", []),
        }

    def verify_all(self, conjectures: list[dict], hops: int = 2) -> list[dict]:
        results = []
        for c in conjectures:
            r = self.verify_conjecture(c, hops=hops)
            results.append(r)
        return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[dict], top_n: int = 10):
    verified   = [r for r in results if r["verified"]]
    unverified = [r for r in results if not r["verified"]]

    print(f"\n{'='*70}")
    print(f"  Conjecture Verification Report")
    print(f"  {len(verified)}/{len(results)} conjectures verified "
          f"(bridge theorem found with sim >= {VERIFY_THRESHOLD})")
    print(f"{'='*70}")

    if verified:
        print(f"\n  VERIFIED ({len(verified)} conjectures point at real known math)")
        print("-" * 70)
        for r in sorted(verified, key=lambda x: -x["best_sim"])[:top_n]:
            print(f"\n  [VERIFIED] novelty={r['novelty']:.4f}  "
                  f"recurrence={r['recurrence']}")
            print(f"  Seed : {r['seed'][:75]}")
            print(f"  Gap points at ({r['best_domain']}):")
            print(f"    sim={r['best_sim']:.4f}  {r['best_theorem'][:80]}")
            if len(r["top_matches"]) > 1:
                for m in r["top_matches"][1:3]:
                    marker = "[verified]" if m["verified"] else "          "
                    print(f"    sim={m['sim']:.4f}  {m['theorem'][:75]}")
            print(f"  Nearest in mesh: "
                  f"{r['nearest_in_mesh'][0][1][:60] if r['nearest_in_mesh'] else 'n/a'}")

    if unverified:
        print(f"\n  UNVERIFIED ({len(unverified)} conjectures -- gap may be novel "
              f"or verification corpus too small)")
        print("-" * 70)
        for r in sorted(unverified, key=lambda x: -x["best_sim"])[:5]:
            print(f"\n  [?] novelty={r['novelty']:.4f}  best_sim={r['best_sim']:.4f}")
            print(f"  Seed : {r['seed'][:75]}")
            print(f"  Closest bridge: {r['best_theorem'][:75]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify HAMesh conjectures against known bridge theorems"
    )
    parser.add_argument("--mesh",        default="math_mesh_combined.pt",
                        help="Mesh used to generate the conjectures")
    parser.add_argument("--conjectures", default="conjectures_combined.json",
                        help="Conjecture log from ham_scholar.py")
    parser.add_argument("--top",         type=int, default=10,
                        help="Top N conjectures to verify")
    parser.add_argument("--hops",        type=int, default=2,
                        help="Diffraction hops for gap recovery (default 2)")
    parser.add_argument("--out",         default=None,
                        help="Save verification results to JSON")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  HAMesh Conjecture Verifier")
    print("=" * 60)

    # Load
    print(f"\n  Loading mesh: {args.mesh}")
    mesh = HolographicMesh.load(args.mesh, device=DEVICE)
    print(f"  {mesh.stats()}")

    print(f"  Loading conjectures: {args.conjectures}")
    data = json.loads(Path(args.conjectures).read_text(encoding="utf-8"))
    conjectures = data.get("conjectures", data) if isinstance(data, dict) else data
    conjectures = conjectures[:args.top]
    print(f"  Verifying top {len(conjectures)} conjectures")

    embedder = Embedder()
    verifier = ConjectureVerifier(mesh, embedder)

    results = verifier.verify_all(conjectures, hops=args.hops)
    print_report(results, top_n=args.top)

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()
