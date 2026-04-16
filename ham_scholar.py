"""
HAMesh Scholar — Self-Teaching Math/Physics Reasoner (LLM-free)

The scholar loads a verified corpus into a HolographicMesh, then dreams on it
indefinitely. No LLM calls. No generation. No hallucination.

The mesh can only blend what it genuinely has. When it keeps pointing at a
region between known theorems that doesn't match anything stored, that is
a conjecture: the mesh is reaching toward something it doesn't yet know.

Two modes:
  Single-domain: one mesh dreams on itself
  Cross-domain:  two meshes (math + physics) cross-pollinate via diffraction.
                 A signal from the math mesh is diffracted through the physics
                 mesh (and vice versa). When the output is novel in BOTH
                 domains simultaneously, it is a cross-domain bridge conjecture.

Usage:
    # Build corpora
    python ham_corpus.py --builtin --save math_mesh.pt
    python ham_corpus.py --physics --save physics_mesh.pt
    python ham_corpus.py --combined --save combined_mesh.pt

    # Single-domain scholar
    python ham_scholar.py --mesh math_mesh.pt
    python ham_scholar.py --mesh math_mesh.pt --cycles 2000 --log conjectures.json

    # Cross-domain scholar (math <-> physics)
    python ham_scholar.py --mesh math_mesh.pt --mesh2 physics_mesh.pt --cycles 1000
    python ham_scholar.py --mesh math_mesh.pt --mesh2 physics_mesh.pt \\
                          --log xd_conjectures.json --cycles 2000
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ham_core import HolographicMesh

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Conjecture logger
# ---------------------------------------------------------------------------

class ConjectureLog:
    """
    Accumulates conjecture candidates found during dreaming.

    A conjecture is recorded when the mesh's diffraction output has
    novelty_score above threshold — meaning it points to a region
    between known theorems that hasn't been folded in.

    Each entry:
        cycle          : dream cycle when found
        novelty        : how far from all stored memories (0=known, 1=unknown)
        seed           : what theorem was used as the probe seed
        nearest        : list of (sim, theorem_text) closest known theorems
        recurrence     : how many times this region has been revisited
        first_seen     : cycle when first detected
    """

    def __init__(self, novelty_threshold=0.35):
        self.threshold = novelty_threshold
        self.entries = []
        self._region_map = {}   # signature -> entry index (dedup)

    def _signature(self, nearest_texts: list[str]) -> str:
        """A stable key for deduplicating similar conjecture regions."""
        return "|".join(sorted(t[:40] for t in nearest_texts[:2]))

    def record(self, cycle: int, novelty: float, seed_text: str,
               nearest: list[tuple]) -> bool:
        """
        Record a conjecture if novelty exceeds threshold.
        Returns True if a new conjecture was logged, False if updated existing.
        """
        if novelty < self.threshold:
            return False

        neighbor_texts = [t for _, t in nearest]
        sig = self._signature(neighbor_texts)

        if sig in self._region_map:
            # Revisit: increment recurrence, update novelty if higher
            idx = self._region_map[sig]
            self.entries[idx]["recurrence"] += 1
            self.entries[idx]["last_seen_cycle"] = cycle
            if novelty > self.entries[idx]["novelty"]:
                self.entries[idx]["novelty"] = novelty
                self.entries[idx]["seed"] = seed_text
            return False
        else:
            entry = {
                "cycle":           cycle,
                "novelty":         round(novelty, 4),
                "seed":            seed_text[:120],
                "nearest":         [(round(s, 4), t[:100]) for s, t in nearest],
                "recurrence":      1,
                "first_seen":      cycle,
                "last_seen_cycle": cycle,
            }
            self._region_map[sig] = len(self.entries)
            self.entries.append(entry)
            return True

    def top(self, n=10) -> list[dict]:
        """Return the most interesting conjectures: highest novelty * recurrence."""
        scored = sorted(
            self.entries,
            key=lambda e: e["novelty"] * (1 + 0.1 * e["recurrence"]),
            reverse=True,
        )
        return scored[:n]

    def save(self, path: str):
        out = {
            "generated":  datetime.now().isoformat(),
            "threshold":  self.threshold,
            "total":      len(self.entries),
            "conjectures": sorted(
                self.entries,
                key=lambda e: e["novelty"] * (1 + 0.1 * e["recurrence"]),
                reverse=True,
            ),
        }
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Conjecture log saved: {path} ({len(self.entries)} entries)")


# ---------------------------------------------------------------------------
# Scholar engine
# ---------------------------------------------------------------------------

class MathScholar:
    """
    A self-teaching reasoner that dreams on a verified math mesh.

    No LLM. No generation. Pure holographic dynamics.
    """

    # Hopfield capacity: ~0.14 * dim before interference noise dominates
    CAPACITY_FACTOR = 0.14

    def __init__(self, mesh: HolographicMesh, novelty_threshold=None):
        self.mesh = mesh
        n = len(mesh.memories)
        capacity = int(self.CAPACITY_FACTOR * mesh.dim)

        # Auto-set threshold based on mesh density.
        # Over-capacity meshes need a lower threshold because every output
        # vector will be close to *something* in the crowded space.
        if novelty_threshold is None:
            ratio = n / max(capacity, 1)
            if ratio <= 1.0:
                novelty_threshold = 0.35   # within capacity: standard
            elif ratio <= 3.0:
                novelty_threshold = 0.25   # moderately over: relax
            elif ratio <= 10.0:
                novelty_threshold = 0.15   # heavily over: use percentile mode
            else:
                novelty_threshold = 0.08   # saturated: very relaxed

        if n > capacity:
            print(f"  [Scholar] Mesh has {n} memories, capacity ~{capacity} "
                  f"({n/capacity:.1f}x over). Using adaptive threshold={novelty_threshold:.2f}.")
            print(f"  [Scholar] Tip: use --max 200 when building the corpus for best results.")

        self.log  = ConjectureLog(novelty_threshold=novelty_threshold)
        self.cycle = 0
        self.stats_history = []

    def _probe_for_conjectures(self, n_probes=30):
        """
        Use stored theorems as seeds; diffract and check novelty.
        Any high-novelty output is a potential conjecture.
        """
        new_count = 0
        indices = random.sample(range(len(self.mesh.memories)),
                                min(n_probes, len(self.mesh.memories)))

        stored = torch.stack([e for e, _ in self.mesh.memories]).to(self.mesh.device)
        # For over-capacity meshes also compute the mean max-sim as a baseline
        # so the threshold is percentile-relative rather than absolute.

        for idx in indices:
            seed_vec, seed_text = self.mesh.memories[idx]
            output = self.mesh.diffract(seed_vec, hops=2)
            novelty = self.mesh.novelty_score(output)

            if novelty >= self.log.threshold:
                sims = F.cosine_similarity(output.unsqueeze(0), stored)
                top_vals, top_idx = torch.topk(sims, min(3, len(self.mesh.memories)))
                nearest = [
                    (top_vals[i].item(), self.mesh.memories[top_idx[i].item()][1])
                    for i in range(len(top_vals))
                ]
                is_new = self.log.record(self.cycle, novelty, seed_text, nearest)
                if is_new:
                    new_count += 1

        return new_count

    def dream_and_discover(
        self,
        total_cycles: int = 1000,
        fold_strength: float = 0.05,
        decay: float = 0.02,
        decay_every: int = 20,
        probe_every: int = 10,
        reseed_every: int = 10,
        verbose: bool = True,
        print_every: int = 50,
    ):
        """
        Main scholar loop: dream on the math mesh and log conjectures.

        Each cycle:
          1. Diffract from current seed (or reseed from stored theorem)
          2. Fold output back in (self-modification)
          3. Every probe_every cycles: scan for novel output regions
          4. Every decay_every cycles: apply soft decay to prevent collapse

        No LLM is called at any point.
        """
        if not self.mesh.memories:
            print("  ERROR: mesh has no memories. Run ham_corpus.py first.")
            return

        start_time = time.time()
        signal = None
        total_new_conjectures = 0

        if verbose:
            print(f"\n  Starting scholar dream: {total_cycles} cycles")
            print(f"  Mesh: {len(self.mesh.memories)} theorems, "
                  f"dim={self.mesh.dim}, device={self.mesh.device}")
            print(f"  Novelty threshold: {self.log.threshold}")
            print("-" * 60)

        for i in range(total_cycles):
            self.cycle += 1

            # --- Seed ---
            if signal is None or i % reseed_every == 0:
                idx = random.randint(0, len(self.mesh.memories) - 1)
                signal = self.mesh.memories[idx][0]

            # --- Diffract ---
            output = self.mesh.diffract(signal, hops=1)

            # --- Self-modification (no LLM) ---
            self.mesh.fold(signal, output, strength=fold_strength)
            self.mesh.fold(output, output, strength=fold_strength * 0.3)

            # Cross-link top-2 activated memories
            _, activated = self.mesh.resonate(signal, hops=1, top_k=2)
            if len(activated) >= 2:
                e1 = self.mesh.memories[activated[0][1]][0]
                e2 = self.mesh.memories[activated[1][1]][0]
                self.mesh.fold(e1, e2, strength=fold_strength * 0.5)

            # --- Decay ---
            if decay > 0 and i > 0 and i % decay_every == 0:
                self.mesh.apply_decay(1.0 - decay)

            # --- Probe for conjectures ---
            if i % probe_every == 0:
                new = self._probe_for_conjectures(n_probes=20)
                total_new_conjectures += new

            # --- Progress report ---
            if verbose and i % print_every == 0 and i > 0:
                energy = torch.norm(self.mesh.mesh).item()
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  cycle={self.cycle:5d}  energy={energy:7.1f}  "
                      f"conjectures={len(self.log.entries):3d}  "
                      f"({rate:.0f} cycles/s)")

            signal = output

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 60)
            print(f"  Done. {total_cycles} cycles in {elapsed:.1f}s  "
                  f"({total_cycles/elapsed:.0f} cycles/s)")
            print(f"  Total conjectures found: {len(self.log.entries)}")

        return self.log

    def report(self, top_n=10):
        """Print the top conjecture candidates."""
        top = self.log.top(top_n)
        if not top:
            print("\n  No conjectures found above threshold.")
            print("  Try lowering --novelty-threshold or running more cycles.")
            return

        print(f"\n  Top {len(top)} Conjecture Candidates")
        print("  (regions the mesh kept pointing toward that don't match known theorems)")
        print("=" * 70)
        for i, e in enumerate(top, 1):
            print(f"\n  [{i}] novelty={e['novelty']:.4f}  "
                  f"recurrence={e['recurrence']}  "
                  f"first_seen=cycle {e['first_seen']}")
            print(f"  Seed: {e['seed'][:80]}")
            print(f"  Nearest known theorems:")
            for sim, text in e["nearest"]:
                print(f"    {sim:.4f}  {text[:70]}")


# ---------------------------------------------------------------------------
# Cross-domain scholar
# ---------------------------------------------------------------------------

class CrossDomainScholar:
    """
    Dreams on two separate meshes simultaneously and finds bridge conjectures.

    Each cycle:
      1. Pick a seed from mesh A (e.g., math)
      2. Diffract through mesh A  -> intermediate_A
      3. Diffract intermediate_A through mesh B (e.g., physics)
      4. Measure novelty in BOTH meshes
      5. If novel in both: record as cross-domain bridge conjecture

    Also runs the reverse direction: B -> A.

    Cross-domain conjectures are the most interesting: they are regions
    where mathematical structure and physical law are both pointing at
    something neither domain has explicitly encoded.

    Examples of what the mesh might find:
      - Noether's theorem gap (symmetry <-> conservation <-> group theory)
      - Statistical mechanics <-> information entropy bridge
      - Riemannian geometry <-> general relativity curvature
      - Spectral theorem <-> quantum observable structure
    """

    CAPACITY_FACTOR = 0.14

    def __init__(self, mesh_a: HolographicMesh, mesh_b: HolographicMesh,
                 label_a: str = "math", label_b: str = "physics",
                 novelty_threshold: float = None):
        self.mesh_a   = mesh_a
        self.mesh_b   = mesh_b
        self.label_a  = label_a
        self.label_b  = label_b
        self.cycle    = 0

        # Auto-threshold: use the more crowded mesh as the reference
        n_a = len(mesh_a.memories)
        n_b = len(mesh_b.memories)
        cap_a = int(self.CAPACITY_FACTOR * mesh_a.dim)
        cap_b = int(self.CAPACITY_FACTOR * mesh_b.dim)
        ratio = max(n_a / max(cap_a, 1), n_b / max(cap_b, 1))

        if novelty_threshold is None:
            if ratio <= 1.0:
                novelty_threshold = 0.30
            elif ratio <= 3.0:
                novelty_threshold = 0.22
            elif ratio <= 10.0:
                novelty_threshold = 0.12
            else:
                novelty_threshold = 0.08

        self.log = ConjectureLog(novelty_threshold=novelty_threshold)
        print(f"  [CrossDomainScholar] {label_a}({n_a}) <-> {label_b}({n_b}), "
              f"threshold={novelty_threshold:.2f}")

    def _cross_probe(self, source_mesh: HolographicMesh,
                     target_mesh: HolographicMesh,
                     source_label: str, target_label: str,
                     n_probes: int = 20) -> int:
        """
        Probe source_mesh seeds through target_mesh, log cross-domain novelties.
        Returns number of new conjectures found.
        """
        new_count = 0
        indices = random.sample(range(len(source_mesh.memories)),
                                min(n_probes, len(source_mesh.memories)))

        stored_a = torch.stack([e for e, _ in source_mesh.memories]).to(source_mesh.device)
        stored_b = torch.stack([e for e, _ in target_mesh.memories]).to(target_mesh.device)

        for idx in indices:
            seed_vec, seed_text = source_mesh.memories[idx]

            # Step 1: diffract through source
            mid = source_mesh.diffract(seed_vec, hops=1)

            # Step 2: cross-diffract through target
            cross = target_mesh.diffract(mid, hops=1)

            # Step 3: novelty in BOTH domains
            sims_a = F.cosine_similarity(cross.unsqueeze(0), stored_a)
            sims_b = F.cosine_similarity(cross.unsqueeze(0), stored_b)
            novelty_a = 1.0 - sims_a.max().item()
            novelty_b = 1.0 - sims_b.max().item()

            # Cross-domain novelty: geometric mean (both must be novel)
            novelty = (novelty_a * novelty_b) ** 0.5

            if novelty >= self.log.threshold:
                # Nearest in target domain
                top_vals, top_idx = torch.topk(sims_b, min(3, len(target_mesh.memories)))
                nearest = [
                    (top_vals[i].item(), target_mesh.memories[top_idx[i].item()][1])
                    for i in range(len(top_vals))
                ]
                # Annotate seed with source domain
                annotated_seed = f"[{source_label}] {seed_text}"
                is_new = self.log.record(self.cycle, novelty, annotated_seed, nearest)
                if is_new:
                    new_count += 1

        return new_count

    def dream_and_discover(
        self,
        total_cycles: int = 1000,
        fold_strength: float = 0.01,
        decay: float = 0.005,
        decay_every: int = 20,
        probe_every: int = 10,
        verbose: bool = True,
        print_every: int = 50,
    ) -> ConjectureLog:
        """
        Main cross-domain loop.

        Interleaves:
          - A->B probing (math seed through physics mesh)
          - B->A probing (physics seed through math mesh)
          - Self-dreaming on each mesh to keep attractors healthy
        """
        start_time = time.time()
        total_new = 0

        if verbose:
            print(f"\n  Starting cross-domain dream: {total_cycles} cycles")
            print(f"  {self.label_a}: {len(self.mesh_a.memories)} memories  "
                  f"{self.label_b}: {len(self.mesh_b.memories)} memories")
            print(f"  Novelty threshold: {self.log.threshold}")
            print("-" * 60)

        signal_a = None
        signal_b = None

        for i in range(total_cycles):
            self.cycle += 1

            # --- Reseed ---
            if signal_a is None or i % 10 == 0:
                idx = random.randint(0, len(self.mesh_a.memories) - 1)
                signal_a = self.mesh_a.memories[idx][0]
            if signal_b is None or i % 10 == 0:
                idx = random.randint(0, len(self.mesh_b.memories) - 1)
                signal_b = self.mesh_b.memories[idx][0]

            # --- Self-dream each mesh to maintain healthy attractors ---
            out_a = self.mesh_a.diffract(signal_a, hops=1)
            self.mesh_a.fold(signal_a, out_a, strength=fold_strength)

            out_b = self.mesh_b.diffract(signal_b, hops=1)
            self.mesh_b.fold(signal_b, out_b, strength=fold_strength)

            # --- Cross-pollinate: fold A output into B and vice versa ---
            # This is what creates genuine bridges: the physics mesh learns
            # to associate math patterns, and vice versa.
            self.mesh_b.fold(out_a, signal_b, strength=fold_strength * 0.3)
            self.mesh_a.fold(out_b, signal_a, strength=fold_strength * 0.3)

            # --- Decay ---
            if decay > 0 and i > 0 and i % decay_every == 0:
                self.mesh_a.apply_decay(1.0 - decay)
                self.mesh_b.apply_decay(1.0 - decay)

            # --- Probe for cross-domain conjectures ---
            if i % probe_every == 0:
                new = self._cross_probe(self.mesh_a, self.mesh_b,
                                        self.label_a, self.label_b, n_probes=15)
                new += self._cross_probe(self.mesh_b, self.mesh_a,
                                         self.label_b, self.label_a, n_probes=15)
                total_new += new

            # --- Progress ---
            if verbose and i % print_every == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  cycle={self.cycle:5d}  "
                      f"conjectures={len(self.log.entries):3d}  "
                      f"({rate:.0f} cycles/s)")

            signal_a = out_a
            signal_b = out_b

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 60)
            print(f"  Done. {total_cycles} cycles in {elapsed:.1f}s  "
                  f"({total_cycles/elapsed:.0f} cycles/s)")
            print(f"  Cross-domain conjectures found: {len(self.log.entries)}")

        return self.log

    def report(self, top_n: int = 10):
        """Print the top cross-domain bridge candidates."""
        top = self.log.top(top_n)
        if not top:
            print("\n  No cross-domain conjectures found above threshold.")
            return

        print(f"\n  Top {len(top)} Cross-Domain Bridge Candidates")
        print(f"  ({self.label_a} <-> {self.label_b})")
        print("=" * 70)
        for i, e in enumerate(top, 1):
            print(f"\n  [{i}] novelty={e['novelty']:.4f}  "
                  f"recurrence={e['recurrence']}  "
                  f"cycle {e['first_seen']}")
            print(f"  Seed: {e['seed'][:80]}")
            print(f"  Nearest in target domain:")
            for sim, text in e["nearest"]:
                print(f"    {sim:.4f}  {text[:70]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HAMesh Scholar: self-teaching math/physics reasoner (LLM-free)"
    )
    parser.add_argument("--mesh",    default="math_mesh.pt",
                        help="Primary mesh file (math or combined)")
    parser.add_argument("--mesh2",   default=None,
                        help="Second mesh for cross-domain mode (e.g. physics_mesh.pt)")
    parser.add_argument("--label",   default="math",
                        help="Label for primary mesh (default: math)")
    parser.add_argument("--label2",  default="physics",
                        help="Label for second mesh (default: physics)")
    parser.add_argument("--cycles",  type=int, default=1000,
                        help="Dream cycles to run (default 1000)")
    parser.add_argument("--novelty-threshold", type=float, default=None,
                        help="Min novelty score (auto-set if omitted)")
    parser.add_argument("--decay",   type=float, default=0.005,
                        help="Decay factor per decay_every cycles (default 0.005)")
    parser.add_argument("--fold-strength", type=float, default=0.01,
                        help="Fold strength per dream cycle (default 0.01)")
    parser.add_argument("--top",     type=int, default=10,
                        help="How many top conjectures to display")
    parser.add_argument("--log",     default="conjectures.json",
                        help="Output path for conjecture log (default conjectures.json)")
    parser.add_argument("--save-mesh", metavar="PATH",
                        help="Save the dreamed primary mesh after running")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  HAMesh Scholar — LLM-free reasoning")
    print("=" * 60)

    print(f"\n  Loading mesh from {args.mesh}...")
    mesh = HolographicMesh.load(args.mesh, device=DEVICE)
    print(f"  {mesh.stats()}")

    if args.mesh2:
        # Cross-domain mode
        print(f"\n  Loading second mesh from {args.mesh2}...")
        mesh2 = HolographicMesh.load(args.mesh2, device=DEVICE)
        print(f"  {mesh2.stats()}")

        scholar = CrossDomainScholar(
            mesh, mesh2,
            label_a=args.label,
            label_b=args.label2,
            novelty_threshold=args.novelty_threshold,
        )
        scholar.dream_and_discover(
            total_cycles=args.cycles,
            fold_strength=args.fold_strength,
            decay=args.decay,
        )
        scholar.report(top_n=args.top)
        scholar.log.save(args.log)

    else:
        # Single-domain mode
        scholar = MathScholar(mesh, novelty_threshold=args.novelty_threshold)
        scholar.dream_and_discover(
            total_cycles=args.cycles,
            fold_strength=args.fold_strength,
            decay=args.decay,
        )
        scholar.report(top_n=args.top)
        scholar.log.save(args.log)

    if args.save_mesh:
        mesh.save(args.save_mesh)
        print(f"  Dreamed mesh saved: {args.save_mesh}")


if __name__ == "__main__":
    main()
