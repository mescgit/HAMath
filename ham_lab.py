"""
HAMesh Research Lab — Autonomous Mathematical Discovery

Closes the full discovery loop:

  1. Scholar dreams on the mesh, finding high-novelty gap regions
  2. Conjectures are scored and ranked (novelty × stability × domain diversity)
  3. LLM formalizes the top conjectures as Lean 4
  4. Discovered theorem texts are embedded and folded back into the mesh
  5. The mesh grows from its own discoveries — it becomes its own corpus
  6. A research journal records every finding with provenance and similarity links

The feedback loop is the key architectural insight: every formalized statement
(even sorry-only) tells the mesh what it was thinking about. Sorry proofs fold
weakly (they are hypotheses). Verified proofs fold strongly (they are facts).
Over time the mesh develops expertise in the areas it keeps discovering.

Usage:
    python ham_lab.py                                        # start new lab
    python ham_lab.py --resume                               # resume from saved state
    python ham_lab.py --mesh math_mesh_advanced.pt           # seed from specific mesh
    python ham_lab.py --mesh math_mesh_advanced.pt \\
                      --mesh2 physics_mesh.pt                # cross-domain mode
    python ham_lab.py --dream-cycles 500 --formalize-top 8   # tune the cycle
    python ham_lab.py --report                               # print journal summary
    python ham_lab.py --max-cycles 10                        # run N cycles then stop
    python ham_lab.py --no-verify                            # skip lean check
"""

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError with math symbols)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

from ham_core import HolographicMesh
from ham_embedder import Embedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LM_STUDIO_URL = "http://192.168.0.183:1234/v1"

# Conjecture generation — creative, broad mathematical reasoning
DEFAULT_MODEL  = "qwen/qwen2.5-coder-32b"

# Lean proof repair — specialized formal verification model
PROVER_MODEL   = "deepseek-prover-v2-7b"

LAB_JOURNAL_PATH = Path("lab_journal.json")
LAB_MESH_PATH    = Path("lab_mesh.pt")
LAB_STATE_PATH   = Path("lab_state.json")
LAB_LEAN_DIR     = Path("lab_lean")


# ---------------------------------------------------------------------------
# Research Journal
# ---------------------------------------------------------------------------

class ResearchJournal:
    """
    Persistent record of all lab discoveries.

    Tracks:
      - Formal statement and Lean code for each discovery
      - Which prior discoveries each new one is related to (citation graph)
      - Lean verification status (verified / sorry / placeholder)
      - Mesh size at time of discovery (tracks growth)
    """

    def __init__(self, path: Path = LAB_JOURNAL_PATH):
        self.path    = path
        self.entries: list[dict] = []
        self._vecs:   list[torch.Tensor] = []   # in-memory only, rebuilt each session

        if path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = data.get("discoveries", [])
        print(f"  [Journal] Loaded {len(self.entries)} prior discoveries")

    def record(self, result: dict, cycle: int, embedder: Embedder,
               mesh_size: int) -> dict:
        """
        Add a discovery. Returns the new journal entry.
        Computes similarity to prior discoveries to build citation links.
        """
        conjecture = result.get("conjecture_text", "").strip()
        lean_code  = result.get("lean_code", "").strip()

        if not conjecture and not lean_code:
            return {}

        # Embed the discovery (conjecture + opening of lean code)
        embed_text = conjecture + "  " + lean_code[:200]
        vec = embedder.embed(embed_text)

        # Find related prior discoveries
        related: list[tuple[float, str]] = []
        for i, prev_vec in enumerate(self._vecs):
            sim = F.cosine_similarity(
                vec.unsqueeze(0), prev_vec.unsqueeze(0)
            ).item()
            if sim > 0.62:
                related.append((sim, self.entries[i].get("id", str(i))))
        related.sort(reverse=True)

        # Determine lean status
        if result.get("verified"):
            lean_status = "verified"
        elif lean_code and any(kw in lean_code for kw in
                               ("theorem ", "lemma ", "def ", "example ")):
            lean_status = "sorry" if "sorry" in lean_code.lower() else "proof_attempt"
        else:
            lean_status = "placeholder"

        entry = {
            "id":          f"disc_{len(self.entries) + 1:04d}",
            "cycle":       cycle,
            "timestamp":   datetime.now().isoformat(),
            "seed":        result.get("seed", "")[:100],
            "novelty":     round(result.get("novelty", 0), 4),
            "recurrence":  result.get("recurrence", 0),
            "conjecture":  conjecture[:200],
            "lean_status": lean_status,
            "verified":    result.get("verified", False),
            "attempts":    result.get("attempts", 1),
            "related_to":  [disc_id for _, disc_id in related[:3]],
            "mesh_size":   mesh_size,
        }

        self.entries.append(entry)
        self._vecs.append(vec)
        return entry

    def save(self):
        data = {
            "lab_version":       "1.0",
            "last_updated":      datetime.now().isoformat(),
            "total_discoveries": len(self.entries),
            "discoveries":       self.entries,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def report(self):
        """Print a human-readable summary of the research journal."""
        if not self.entries:
            print("  No discoveries yet.")
            return

        verified      = [e for e in self.entries if e["lean_status"] == "verified"]
        sorry_        = [e for e in self.entries if e["lean_status"] == "sorry"]
        proof_attempt = [e for e in self.entries if e["lean_status"] == "proof_attempt"]

        # Domain distribution from seed prefixes
        domains: dict[str, int] = {}
        for e in self.entries:
            seed = e.get("seed", "")
            d = seed.split(":")[0].strip() if ":" in seed else "unknown"
            domains[d] = domains.get(d, 0) + 1

        # Citation counts (how often each discovery is referenced by later ones)
        citations: dict[str, int] = {}
        for e in self.entries:
            for ref in e.get("related_to", []):
                citations[ref] = citations.get(ref, 0) + 1

        # Novelty over time (by cycle)
        cycle_novelty: dict[int, list[float]] = {}
        for e in self.entries:
            c = e.get("cycle", 0)
            cycle_novelty.setdefault(c, []).append(e.get("novelty", 0))

        print(f"\n{'='*60}")
        print(f"  HAMesh Research Lab — Journal Summary")
        print(f"{'='*60}")
        print(f"  Total discoveries   : {len(self.entries)}")
        print(f"  Verified (no sorry) : {len(verified)}")
        print(f"  Sorry proofs        : {len(sorry_)}")
        print(f"  Proof attempts      : {len(proof_attempt)}")
        print(f"  Placeholders        : "
              f"{len(self.entries) - len(verified) - len(sorry_) - len(proof_attempt)}")

        print(f"\n  Discoveries by area:")
        for d, n in sorted(domains.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * min(n, 20)
            print(f"    {d[:28]:<28} {bar} ({n})")

        if cycle_novelty:
            print(f"\n  Average novelty per cycle:")
            for cyc in sorted(cycle_novelty):
                avg = sum(cycle_novelty[cyc]) / len(cycle_novelty[cyc])
                bar = "█" * int(avg * 20)
                print(f"    Cycle {cyc:3d}  {bar}  {avg:.3f}")

        if citations:
            print(f"\n  Most cited discoveries:")
            for disc_id, cnt in sorted(citations.items(), key=lambda x: -x[1])[:5]:
                e = next((e for e in self.entries if e["id"] == disc_id), None)
                if e:
                    print(f"    [{disc_id}] cited {cnt}×  {e['conjecture'][:70]}")

        if verified:
            print(f"\n  Verified theorems:")
            for e in verified[:8]:
                print(f"    [{e['id']}] {e['conjecture'][:80]}")


# ---------------------------------------------------------------------------
# Research Lab
# ---------------------------------------------------------------------------

class MathResearchLab:
    """
    Autonomous research loop: dream → score → formalize → fold back → repeat.

    The mesh starts from a seed corpus (math_mesh_advanced.pt) and grows as
    the lab makes discoveries. Verified theorems fold in at 5× the strength
    of sorry proofs, so the mesh prioritises confirmed knowledge.
    """

    # Fold strengths by lean status
    FOLD_STRENGTH = {
        "verified":      0.05,
        "proof_attempt": 0.025,
        "sorry":         0.015,
        "placeholder":   0.003,
    }

    # Minimum novelty vs mesh before folding back (avoids redundant re-folding)
    MIN_FOLD_NOVELTY = 0.20

    def __init__(
        self,
        mesh:            HolographicMesh,
        embedder:        Embedder,
        lm_url:          str = LM_STUDIO_URL,
        model:           str = DEFAULT_MODEL,
        lean_bin:        str | None = None,
        max_attempts:    int = 2,
        verify:          bool = False,
        dream_cycles:    int = 300,
        formalize_top:   int = 5,
        mesh2:           HolographicMesh | None = None,
        mathlib_cfg:     dict | None = None,
        prover_model:    str | None = PROVER_MODEL,
    ):
        self.mesh          = mesh
        self.mesh2         = mesh2
        self.embedder      = embedder
        self.model         = model
        self.dream_cycles  = dream_cycles
        self.formalize_top = formalize_top
        self.cycle         = 0
        self.journal       = ResearchJournal()
        self.mathlib_cfg   = mathlib_cfg

        # Store scholar class/args so we can spin up a fresh instance each cycle
        # (fresh log = fresh attractor exploration on the evolved mesh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if mesh2:
                from ham_scholar import CrossDomainScholar as _ScholarCls
            else:
                from ham_scholar import MathScholar as _ScholarCls
        self._ScholarCls = _ScholarCls
        self._scholar_kwargs: dict = {}   # MathScholar takes no extra kwargs

        # Build autoformalization engine
        from ham_lean import AutoformalEngine, make_client, load_mathlib_config
        client = make_client(lm_url)
        # Auto-detect Mathlib config if not explicitly passed
        if mathlib_cfg is None:
            mathlib_cfg = load_mathlib_config()
        # Prover model uses the same endpoint but a different model name
        prover_client = make_client(lm_url) if prover_model else None
        self.engine = AutoformalEngine(
            client, model, lean_bin,
            max_attempts=max_attempts,
            verify=verify and (lean_bin is not None or mathlib_cfg is not None),
            mathlib_cfg=mathlib_cfg,
            prover_client=prover_client,
            prover_model=prover_model,
        )

        # Cooldown map: seed_key → last cycle it was formalized.
        # A seed can be re-formalized after SEED_COOLDOWN cycles (mesh may have
        # evolved and will produce a different conjecture from the same seed).
        self.SEED_COOLDOWN = 8
        self._seed_last_cycle: dict[str, int] = {
            e.get("seed", ""): 0          # journal seeds start with cooldown 0
            for e in self.journal.entries
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score(c: dict) -> float:
        """
        Priority score for deciding which conjectures to formalize.

        High novelty = unexplored territory.
        High recurrence = the mesh keeps returning here (stable attractor).
        The log dampens recurrence so extremely popular hubs don't monopolise.
        """
        novelty    = c.get("novelty", 0)
        recurrence = c.get("recurrence", 1)
        return novelty ** 0.8 * math.log1p(recurrence)

    # ------------------------------------------------------------------
    # Feedback loop
    # ------------------------------------------------------------------

    @staticmethod
    def _is_clean_conjecture(text: str) -> bool:
        """
        Return False if the text looks like an LLM prompt echo or code fragment
        rather than a genuine plain-English conjecture.  Such strings must not
        be folded into the mesh — they cause prompt-template contamination that
        surfaces as nonsense seeds in future cycles.
        """
        if len(text) < 20:
            return False
        # Code fragments: backticks, escaped newlines, triple-quotes
        if "`" in text or "\\n" in text or '"""' in text:
            return False
        # Looks like the LLM echoed the output-format hint verbatim
        if "```lean" in text or "```" in text:
            return False
        # Common prompt-echo patterns
        if text.endswith(('"', "```", "```.")) and "`" in text:
            return False
        # Very long lines with no spaces are probably code / base64 / hashes
        if len(text) > 50 and " " not in text:
            return False
        return True

    def _fold_back(self, result: dict) -> bool:
        """
        Embed the discovered conjecture and fold it into the growing mesh.
        Returns True if the fold happened (discovery was novel enough).
        """
        conjecture = result.get("conjecture_text", "").strip()
        if not self._is_clean_conjecture(conjecture):
            return False

        vec = self.embedder.embed(conjecture)

        # Check novelty vs current mesh — don't fold redundant knowledge
        _, acts = self.mesh.resonate(vec, hops=1, top_k=1)
        max_sim = acts[0][0] if acts else 0.0
        if (1.0 - max_sim) < self.MIN_FOLD_NOVELTY:
            return False

        # Also skip conjectures that are near-copies of corpus entries
        # (e.g. LLM echoed a known theorem rather than producing a bridge)
        if max_sim > 0.85:
            return False

        lean_status = result.get("lean_status", "placeholder")
        strength    = self.FOLD_STRENGTH.get(lean_status, 0.01)

        # key = seed concept, val = discovered theorem
        # This creates: "when thinking about X, recall this new theorem"
        seed = result.get("seed", "")
        seed_vec = self.embedder.embed(seed) if seed else vec

        self.mesh.fold(seed_vec, vec, strength=strength)
        self.mesh.remember(vec, f"[lab] {conjecture[:120]}")
        return True

    # ------------------------------------------------------------------
    # Research cycle
    # ------------------------------------------------------------------

    def research_cycle(self, out_dir: Path) -> int:
        """
        One full cycle: dream → collect → score → formalize → fold back → log.
        Returns the number of conjectures processed.
        """
        self.cycle += 1
        now = datetime.now().strftime("%H:%M:%S")
        s   = self.mesh.stats()

        print(f"\n{'='*60}")
        print(f"  Cycle {self.cycle:>3}  |  {now}  |  "
              f"{s['memories']} memories  energy={s['energy']:.1f}")
        print(f"{'='*60}")

        # --- 1. Dream — fresh scholar each cycle on the current mesh state ---
        # A fresh log means the scholar explores without being constrained by
        # what it found in previous cycles. The cooldown map (not the log) is
        # what prevents re-formalizing the same seed too soon.
        print(f"\n  [1/4] Dreaming {self.dream_cycles} cycles...", flush=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if self.mesh2 is not None:
                scholar = self._ScholarCls(self.mesh, self.mesh2)
            else:
                scholar = self._ScholarCls(self.mesh)
            scholar.dream_and_discover(
                total_cycles=self.dream_cycles,
                verbose=False,
            )

        # --- 2. Collect conjectures that are off cooldown ---
        all_conjectures = scholar.log.entries if hasattr(scholar, "log") else []

        new = [
            c for c in all_conjectures
            if (self.cycle - self._seed_last_cycle.get(c.get("seed", ""), -999))
               > self.SEED_COOLDOWN
        ]
        # Mark each as seen this cycle
        for c in new:
            self._seed_last_cycle[c.get("seed", "")] = self.cycle

        new.sort(key=self._score, reverse=True)

        # Diversity filter: cap how many times the same domain seed prefix can
        # appear in a single batch. Prevents one strong attractor (e.g. CLT)
        # from consuming all formalization slots every cycle.
        MAX_PER_DOMAIN = 2
        domain_counts: dict[str, int] = {}
        diverse_batch = []
        for c in new:
            # Domain key = first word of seed (e.g. "[math]" or "[physics]" +
            # the topic stem before the colon)
            seed = c.get("seed", "")
            parts = seed.split(":")
            domain_key = parts[0].strip()[:40] if parts else seed[:40]
            if domain_counts.get(domain_key, 0) < MAX_PER_DOMAIN:
                domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
                diverse_batch.append(c)
            if len(diverse_batch) >= self.formalize_top:
                break

        batch = diverse_batch

        print(f"  [2/4] {len(new)} new conjectures  ->  formalizing top {len(batch)}")
        if not batch:
            print("        (none — mesh may need more dream cycles or a fresh seed)")
            return 0

        # Show what we're about to formalize
        for c in batch:
            print(f"        score={self._score(c):.3f}  novelty={c['novelty']:.3f}  "
                  f"rec={c['recurrence']}  {c['seed'][:60]}")

        # --- 3. Formalize ---
        print(f"\n  [3/4] Formalizing...")
        out_dir.mkdir(exist_ok=True)
        folded = 0

        for c in batch:
            result = self.engine.process(c)

            # Compute lean_status for journal + fold-back
            if result.get("verified"):
                result["lean_status"] = "verified"
            elif result.get("lean_code") and any(
                kw in result["lean_code"]
                for kw in ("theorem ", "lemma ", "def ", "example ")
            ):
                result["lean_status"] = (
                    "sorry" if "sorry" in result["lean_code"].lower()
                    else "proof_attempt"
                )
            else:
                result["lean_status"] = "placeholder"

            # Save .lean file
            lean_path = self.engine.save_lean_file(result, out_dir)
            print(f"  -> {lean_path.name}")

            # --- 4. Fold back ---
            if self._fold_back(result):
                folded += 1
                print(f"     ↳ folded back [{result['lean_status']}  "
                      f"strength={self.FOLD_STRENGTH.get(result['lean_status'], 0):.3f}]")

            # Record in journal
            entry = self.journal.record(
                result, self.cycle, self.embedder,
                mesh_size=len(self.mesh.memories),
            )
            if entry.get("related_to"):
                print(f"     ↳ related to: {', '.join(entry['related_to'])}")

        print(f"\n  [4/4] {folded}/{len(batch)} discoveries folded into mesh")
        return len(batch)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, mesh_path: Path, state_path: Path):
        self.mesh.save(str(mesh_path))
        self.journal.save()
        # Keep only the 500 most recently seen seeds to avoid unbounded growth
        recent = sorted(self._seed_last_cycle.items(), key=lambda x: -x[1])[:500]
        state = {
            "cycle":            self.cycle,
            "seed_last_cycle":  dict(recent),
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"\n  [Saved] {mesh_path}  |  journal: {self.journal.path}"
              f"  |  cycle {self.cycle}")

    def load_state(self, state_path: Path):
        if not state_path.exists():
            return
        state = json.loads(state_path.read_text())
        self.cycle = state.get("cycle", 0)
        self._seed_last_cycle.update(state.get("seed_last_cycle", {}))
        print(f"  [Resumed] cycle={self.cycle}  "
              f"tracked_seeds={len(self._seed_last_cycle)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HAMesh Research Lab — autonomous mathematical discovery"
    )
    parser.add_argument("--mesh",          default="math_mesh_advanced.pt",
                        help="Seed mesh (used for first run; lab grows its own copy)")
    parser.add_argument("--mesh2",         default=None,
                        help="Second domain mesh for cross-domain mode (e.g. physics_mesh.pt)")
    parser.add_argument("--model",         default=DEFAULT_MODEL,
                        help=f"Conjecture generation model (default: {DEFAULT_MODEL})")
    parser.add_argument("--prover-model",  default=PROVER_MODEL,
                        help=f"Lean repair model (default: {PROVER_MODEL}). "
                             "Pass '' to use the same model as --model.")
    parser.add_argument("--url",           default=LM_STUDIO_URL)
    parser.add_argument("--dream-cycles",  type=int, default=1000,
                        help="Scholar dream cycles per lab cycle (default 1000)")
    parser.add_argument("--formalize-top", type=int, default=5,
                        help="Conjectures to formalize per cycle (default 5)")
    parser.add_argument("--attempts",      type=int, default=2,
                        help="LLM repair attempts per conjecture (default 2)")
    parser.add_argument("--cooldown",      type=int, default=8,
                        help="Cycles before a seed can be re-formalized (default 8)")
    parser.add_argument("--no-verify",     action="store_true",
                        help="Skip Lean verification")
    parser.add_argument("--no-mathlib",    action="store_true",
                        help="Disable Mathlib even if mathlib_config.json exists")
    parser.add_argument("--lab-mesh",      default=str(LAB_MESH_PATH),
                        help="Growing mesh path (accumulates lab discoveries)")
    parser.add_argument("--state",         default=str(LAB_STATE_PATH),
                        help="Lab state JSON path")
    parser.add_argument("--out",           default=str(LAB_LEAN_DIR),
                        help="Output directory for .lean files")
    parser.add_argument("--max-cycles",    type=int, default=0,
                        help="Stop after N cycles (0 = run forever)")
    parser.add_argument("--report",        action="store_true",
                        help="Print journal summary and exit")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from saved lab state and lab mesh")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  HAMesh Research Lab")
    print("="*60)

    # Report mode — no LLM needed
    if args.report:
        ResearchJournal().report()
        return

    embedder = Embedder()

    # Load mesh: prefer the growing lab mesh when resuming
    lab_mesh_path = Path(args.lab_mesh)
    if args.resume and lab_mesh_path.exists():
        print(f"\n  Loading lab mesh: {lab_mesh_path}")
        mesh = HolographicMesh.load(str(lab_mesh_path), device=DEVICE)
    else:
        print(f"\n  Seeding from corpus mesh: {args.mesh}")
        mesh = HolographicMesh.load(args.mesh, device=DEVICE)
    s = mesh.stats()
    print(f"  {s['memories']} memories  energy={s['energy']:.1f}")

    # Optional second mesh (cross-domain)
    mesh2 = None
    if args.mesh2:
        p2 = Path(args.mesh2)
        if p2.exists():
            print(f"  Cross-domain mesh: {args.mesh2}")
            mesh2 = HolographicMesh.load(str(p2), device=DEVICE)
            s2 = mesh2.stats()
            print(f"  {s2['memories']} memories  energy={s2['energy']:.1f}")
        else:
            print(f"  [Warning] --mesh2 path not found: {args.mesh2}")

    # Lean + optional Mathlib
    from ham_lean import find_lean, load_mathlib_config
    lean_bin    = None
    mathlib_cfg = None
    if not args.no_verify:
        lean_bin = find_lean()
        if lean_bin:
            r = subprocess.run([lean_bin, "--version"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            print(f"  Lean 4: {r.stdout.strip()}")
        else:
            print("  Lean 4: not found — running in --no-verify mode")
            args.no_verify = True

        # Mathlib: auto-detect config, or explicitly disabled
        if not getattr(args, 'no_mathlib', False):
            mathlib_cfg = load_mathlib_config()
            if mathlib_cfg:
                print(f"  Mathlib: {mathlib_cfg['mathlib_project_dir']}  [ENABLED]")
            else:
                print("  Mathlib: not configured  (run setup_mathlib.py to enable)")

    prover_model = args.prover_model or None   # treat empty string as None

    print(f"\n  Dream cycles / round : {args.dream_cycles}")
    print(f"  Formalize top        : {args.formalize_top}")
    print(f"  Max lab cycles       : {'inf' if not args.max_cycles else args.max_cycles}")
    print(f"  Cross-domain         : {'yes' if mesh2 else 'no'}")
    print(f"  Mathlib mode         : {'yes' if mathlib_cfg else 'no (standalone)'}")
    print(f"  Conjecture model     : {args.model}")
    print(f"  Repair model         : {prover_model or args.model}  "
          f"{'(dual-model)' if prover_model and prover_model != args.model else '(same model)'}")
    print(f"  Output               : {args.out}/")
    print(f"\n  Ctrl-C to stop cleanly.\n")
    print("="*60)

    lab = MathResearchLab(
        mesh=mesh, mesh2=mesh2, embedder=embedder,
        lm_url=args.url, model=args.model, lean_bin=lean_bin,
        max_attempts=args.attempts, verify=not args.no_verify,
        mathlib_cfg=mathlib_cfg,
        dream_cycles=args.dream_cycles, formalize_top=args.formalize_top,
        prover_model=prover_model,
    )
    lab.SEED_COOLDOWN = args.cooldown

    if args.resume:
        lab.load_state(Path(args.state))

    out_dir    = Path(args.out)
    state_path = Path(args.state)

    session_cycles = 0
    try:
        while True:
            n = lab.research_cycle(out_dir)
            session_cycles += 1
            lab.save(lab_mesh_path, state_path)

            if args.max_cycles > 0 and session_cycles >= args.max_cycles:
                print(f"\n  Ran {args.max_cycles} cycle(s) this session. Stopping.")
                break

            if n == 0:
                import time
                print("  No new conjectures this cycle. Waiting 3s...")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n  Interrupted — saving...")
        lab.save(lab_mesh_path, state_path)

    print("\n  Final report:")
    lab.journal.report()
    print("\n  Done.")


if __name__ == "__main__":
    main()
