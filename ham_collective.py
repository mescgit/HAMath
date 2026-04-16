"""
HAM Collective — Multiple Holographic Meshes + Curiosity Engine

Two (or more) specialized meshes think in parallel.
Their strongest attractor patterns cross-pollinate during dreaming.
The curiosity engine finds knowledge gaps and fills them autonomously.

Architecture:
  mesh_a  (science / math / technology)
  mesh_b  (philosophy / history / nature)
     ↓            ↓
     ├── blend intuitions ──→ unified signal
     ├── label by source ──→ "[science]  ..." / "[philosophy] ..."
     └── cross-pollinate ──→ shared attractor loops across domains

Setup:
  1. Start Bonsai (embedding + generation):
       ./llama-server.exe -m bonsai-8b.gguf --port 22334 --ctx-size 8192 --embedding --pooling mean

  2. Distill two specialised meshes:
       python ham_distill.py --topics science,math,technology   --save ham_science.pt
       python ham_distill.py --topics philosophy,history,nature --save ham_philosophy.pt

  3. Run the collective:
       python ham_collective.py

  Optional — add a third mesh:
       python ham_distill.py --topics art,music,literature --save ham_art.pt
       python ham_collective.py --meshes science:ham_science.pt,philosophy:ham_philosophy.pt,art:ham_art.pt

Commands:
  <question>         Both meshes diffract; blended answer
  deep <question>    2-hop transitive reasoning across meshes
  trace <question>   Show hop-by-hop from each mesh
  curious [N]        Curiosity engine: find gaps, generate questions, fill them
  dream [N]          Collective dream + cross-pollinate
  cross              Manual cross-pollination pass
  teach <text>       Fold into all meshes
  fold <A> -> <B>    Directed association in all meshes
  status             Per-mesh diagnostics
  save               Save all meshes
  exit               Quit
"""

import argparse
import os
import sys
import textwrap

import torch
import torch.nn.functional as F

from ham_core import HolographicMesh
from ham_brain  import embed, generate, MESH_SAVE, CONFIDENCE_FLOOR, SELF_TEACH_STRENGTH
from ham_brain  import ingest_directory, ingest_text, DATA_DIR
from ham_logger import HAMLogger

# ---------------------------------------------------------------------------
# Default mesh layout — override with --meshes on the command line
# ---------------------------------------------------------------------------
DEFAULT_MESHES = {
    "science":    "./ham_science.pt",
    "philosophy": "./ham_philosophy.pt",
}

# How strongly one mesh's dominant patterns bleed into another
CROSS_POLLINATION_STRENGTH = 0.04


# ---------------------------------------------------------------------------
# MeshCollective
# ---------------------------------------------------------------------------

class MeshCollective:
    """
    A network of collaborating HolographicMesh instances.

    Each mesh has a domain specialty.  Queries are sent to all meshes;
    results are labelled by source and blended into a single intuition.
    During collective dreaming, dominant attractor patterns cross-pollinate.
    """

    def __init__(self, meshes: dict):
        """meshes: {name: HolographicMesh}"""
        self.meshes = meshes

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def blend(self, query_emb, hops=1):
        """
        Weighted average of all mesh diffractions.
        Meshes with more energy have more influence.
        """
        parts   = []
        weights = []
        for ham in self.meshes.values():
            intuition = ham.diffract(query_emb, hops=hops)
            w = ham.stats()['energy']
            parts.append(intuition * w)
            weights.append(w)
        total_w = sum(weights)
        if total_w == 0:
            return F.normalize(parts[0], dim=0) if parts else query_emb
        blended = sum(parts) / total_w
        return F.normalize(blended, dim=0)

    def resonate_all(self, query_emb, hops=1, top_k=4):
        """
        Each mesh resonates independently.
        Returns {name: (intuition, activated_list)}.
        """
        return {
            name: ham.resonate(query_emb, hops=hops, top_k=top_k)
            for name, ham in self.meshes.items()
        }

    def collective_resonate(self, query_emb, hops=1, top_k=6):
        """
        Blend all mesh outputs, then find which memories across ALL
        meshes are most activated by the blended signal.

        Returns (blended_intuition, [(sim, text, mesh_name), ...])
        """
        blended = self.blend(query_emb, hops=hops)

        all_items = []  # (embedding, text, mesh_name)
        for name, ham in self.meshes.items():
            for emb, text in ham.memories:
                all_items.append((emb, text, name))

        if not all_items:
            return blended, []

        stored = torch.stack([e for e, _, _ in all_items])
        sims   = F.cosine_similarity(blended.unsqueeze(0), stored)
        k      = min(top_k, len(all_items))
        vals, idx = torch.topk(sims, k)

        activated = [
            (vals[i].item(), all_items[idx[i]][1], all_items[idx[i]][2])
            for i in range(k)
        ]
        return blended, activated

    # ------------------------------------------------------------------
    # Cross-pollination
    # ------------------------------------------------------------------

    def cross_pollinate(self, strength=CROSS_POLLINATION_STRENGTH, n=8):
        """
        Share each mesh's dominant attractor patterns with all others.
        Low strength keeps them distinct; too high homogenises everything.
        """
        dominants = {}
        for name, ham in self.meshes.items():
            dominants[name] = ham.dominant_memories(n=n)

        shared = 0
        for src_name, top_mems in dominants.items():
            for dst_name, dst_ham in self.meshes.items():
                if src_name == dst_name:
                    continue
                for sim, idx, text, emb in top_mems:
                    dst_ham.fold(emb, emb, strength=strength)
                    dst_ham.remember(emb, f"[from {src_name}] {text}")
                    shared += 1

        return shared

    # ------------------------------------------------------------------
    # Collective dreaming
    # ------------------------------------------------------------------

    def collective_dream(self, cycles=50, fold_strength=0.1, decay=0.02):
        """
        All meshes dream simultaneously, then cross-pollinate.
        Dominant patterns from each domain bleed into the others.
        decay: per-20-cycle decay factor (0.02 = 2% reduction every 20 cycles)
        """
        results = {}
        for name, ham in self.meshes.items():
            print(f"\n  [{name}] dreaming ({cycles} cycles)...")
            log, attractors, loops = ham.dream(
                cycles=cycles, fold_strength=fold_strength,
                decay=decay, decay_every=20
            )
            results[name] = {'attractors': attractors[:5], 'loops': loops[:3]}

        print("\n  Cross-pollinating...")
        shared = self.cross_pollinate()
        print(f"  {shared} attractor patterns shared across meshes.")

        return results

    # ------------------------------------------------------------------
    # Curiosity engine
    # ------------------------------------------------------------------

    def be_curious(self, n_gaps=4, questions_per_gap=2, fold_strength=0.25):
        """
        Autonomous self-expansion:
          1. Find isolated/weak memories in each mesh
          2. Identify universal gaps (weak in ALL meshes)
          3. Generate probing questions about each gap
          4. Let the collective answer them (2-hop diffraction)
          5. Fold the Q→A back into every mesh

        Returns the gaps explored and insights gained.
        """
        print("  Scanning for knowledge gaps...")

        # Collect isolation scores per memory across all meshes
        gap_catalog = {}  # text_key -> [(score, mesh_name)]
        for name, ham in self.meshes.items():
            for score, idx, text in ham.find_isolated(top_k=20):
                key = text[:80]
                gap_catalog.setdefault(key, []).append((score, name))

        # Universal gaps appear in multiple meshes — the real unknowns
        universal = sorted(
            [(t, e) for t, e in gap_catalog.items() if len(e) >= 2],
            key=lambda x: sum(s for s, _ in x[1])
        )
        domain_only = sorted(
            [(t, e) for t, e in gap_catalog.items() if len(e) == 1],
            key=lambda x: x[1][0][0]
        )

        # Prioritise universal gaps; fill with domain gaps if needed
        gaps = (universal + domain_only)[:n_gaps]

        if not gaps:
            print("  No significant gaps found (mesh may need more knowledge).")
            return []

        print(f"  Found {len(universal)} universal gaps, {len(domain_only)} domain gaps.")
        insights = []

        for gap_text, entries in gaps:
            mesh_names = [name for _, name in entries]
            print(f"\n  Gap [{', '.join(mesh_names)}]: '{gap_text[:60]}'")

            # Ask Bonsai to generate probing questions at this knowledge boundary
            q_prompt = (
                f"The concept '{gap_text}' is poorly connected in a knowledge network "
                f"covering {', '.join(self.meshes.keys())}.\n\n"
                f"Generate {questions_per_gap} specific, probing questions that would "
                f"deepen understanding of this topic and reveal connections to other "
                f"domains. One question per line, no numbering."
            )
            raw_questions = generate(
                q_prompt,
                system="Be intellectually curious. Ask about mechanisms, "
                       "surprising connections, and underlying principles.",
                max_tokens=250,
            )

            questions = [
                q.strip().lstrip('-').lstrip('•').strip()
                for q in raw_questions.strip().split('\n')
                if q.strip() and len(q.strip()) > 15
            ][:questions_per_gap]

            for question in questions:
                print(f"    ? {question[:75]}")

                q_emb = embed(question)
                blended, activated = self.collective_resonate(q_emb, hops=2, top_k=6)

                context = "\n".join(
                    f"[{mesh}|{sim:+.2f}] {text[:100]}"
                    for sim, text, mesh in activated
                    if sim > CONFIDENCE_FLOOR
                )

                sys_prompt = (
                    "You are the voice of a multi-domain Holographic Associative Memory. "
                    "These patterns have activated across specialised meshes:\n\n"
                    f"{context}\n\n"
                    "Synthesise an insightful answer from these cross-domain patterns. "
                    "Look for connections the individual meshes couldn't see alone."
                ) if context else (
                    "Answer with your own knowledge. Be specific and insightful."
                )

                answer = generate(question, system=sys_prompt, max_tokens=350)
                print(f"      → {answer[:120]}...")

                # Fold the Q→A into ALL meshes
                a_emb = embed(answer[:400])
                for ham in self.meshes.values():
                    ham.fold(q_emb, a_emb, strength=fold_strength)
                    ham.remember(a_emb, answer[:250])

                insights.append((question, answer, activated))

        return insights

    # ------------------------------------------------------------------
    # Knowledge ingestion
    # ------------------------------------------------------------------

    def teach_all(self, text):
        """Fold text into every mesh."""
        total = 0
        for ham in self.meshes.values():
            total += ingest_text(ham, text)
        return total

    def fold_all(self, a_emb, b_emb, a_text, b_text, strength=1.0):
        """Fold a directed association into every mesh."""
        for ham in self.meshes.values():
            ham.learn(a_emb, b_emb, a_text, b_text, strength=strength)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path_map):
        """path_map: {name: path}"""
        for name, ham in self.meshes.items():
            if name in path_map:
                ham.save(path_map[name])

    def stats(self):
        return {name: ham.stats() for name, ham in self.meshes.items()}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

# Colour codes for mesh labels (cycles through a few)
_MESH_COLORS = ['\033[96m', '\033[93m', '\033[92m', '\033[95m']
_RESET       = '\033[0m'

def mesh_color(name, mesh_names):
    idx = list(mesh_names).index(name) % len(_MESH_COLORS)
    return _MESH_COLORS[idx]


def show_collective_resonance(activated, mesh_names):
    if not activated:
        print("    (no memories activated)")
        return
    print()
    for sim, text, mesh in activated:
        col      = mesh_color(mesh, mesh_names)
        bar_len  = max(1, int(abs(sim) * 28))
        bar      = '█' * bar_len
        preview  = text[:80].replace('\n', ' ')
        if len(text) > 80:
            preview += '...'
        print(f"    {col}[{mesh:>10}]{_RESET} {sim:+.3f} {bar} {preview}")
    print()


def show_collective_trace(per_mesh_paths, mesh_names):
    for name, path in per_mesh_paths.items():
        col = mesh_color(name, mesh_names)
        print(f"  {col}[{name}]{_RESET}")
        for hop, activations in path:
            print(f"    --- hop {hop} ---")
            for sim, idx, text in activations:
                print(f"      {sim:+.3f}  {text[:75].replace(chr(10), ' ')}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HAM Collective — multi-mesh reasoning")
    parser.add_argument(
        "--meshes", type=str, default=None,
        help="Comma-separated name:path pairs, e.g. "
             "science:ham_science.pt,philosophy:ham_philosophy.pt"
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR,
        help=f"Folder of .txt knowledge files ingested into all meshes (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Session name for research logging (e.g. --log session_01). "
             "Logs saved to ./ham_logs/"
    )
    args = parser.parse_args()

    # Parse mesh spec
    if args.meshes:
        mesh_spec = {}
        for part in args.meshes.split(','):
            name, path = part.strip().split(':', 1)
            mesh_spec[name.strip()] = path.strip()
    else:
        mesh_spec = DEFAULT_MESHES

    print("=" * 66)
    print("   HAM Collective  —  Multi-Mesh Holographic Reasoning")
    print("=" * 66)

    # Connect to Bonsai
    print("\n  Connecting to Bonsai...", end=" ", flush=True)
    try:
        test_vec = embed("hello")
        dim = test_vec.shape[0]
        print(f"OK  (dim={dim})")
    except Exception as e:
        print(f"FAILED\n  {e}")
        print("\n  Start Bonsai:")
        print("  ./llama-server.exe -m bonsai-8b.gguf --port 22334 "
              "--ctx-size 8192 --embedding --pooling mean")
        sys.exit(1)

    # Load or create each mesh
    meshes = {}
    for name, path in mesh_spec.items():
        if os.path.exists(path):
            print(f"\n  Loading [{name}] from {path}...")
            ham = HolographicMesh.load(path)
            print(f"    {ham.n_folds} folds, {len(ham.memories)} memories, "
                  f"energy={ham.stats()['energy']:.0f}")
        else:
            print(f"\n  [{name}] not found at {path} — creating fresh mesh")
            ham = HolographicMesh(dim)
        meshes[name] = ham

    collective = MeshCollective(meshes)

    # Ingest shared data directory into all meshes
    data_dir = args.data_dir
    if os.path.exists(data_dir):
        import glob
        if glob.glob(os.path.join(data_dir, "*.txt")):
            print(f"\n  Ingesting {data_dir}/ into all meshes...")
            for name, ham in meshes.items():
                ingest_directory(ham, data_dir)

    # Commands
    mesh_names = list(meshes.keys())
    print("\n" + "-" * 66)
    print("  Commands:")
    print("    <question>         Both meshes diffract; blended answer")
    print("    deep <question>    2-hop transitive reasoning across meshes")
    print("    trace <question>   Hop-by-hop from each mesh")
    print("    curious [N]        Curiosity engine: find gaps and fill them")
    print("    dream [N]          Collective dream + cross-pollinate")
    print("    cross              Manual cross-pollination")
    print("    teach <text>       Fold into all meshes")
    print("    fold <A> -> <B>    Directed association in all meshes")
    print("    status             Per-mesh diagnostics")
    print("    save               Save all meshes")
    print("    exit               Quit")
    print("-" * 66)

    path_map = mesh_spec  # for saving

    # Optional research logger
    logger = HAMLogger(args.log) if args.log else None
    if logger:
        logger.print_path()
        for name, ham in meshes.items():
            logger.log_mesh_snapshot(name, ham.stats(), ham.dominant_memories(n=10))

    while True:
        try:
            raw = input("\nCOLLECTIVE> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        cmd = raw.lower()

        # --- exit ---
        if cmd == 'exit':
            break

        # --- status ---
        elif cmd == 'status':
            for name, s in collective.stats().items():
                col = mesh_color(name, mesh_names)
                print(f"  {col}[{name}]{_RESET}")
                print(f"    Folds    : {s['folds']}")
                print(f"    Memories : {s['memories']}")
                print(f"    Energy   : {s['energy']:.1f}")

        # --- save ---
        elif cmd == 'save':
            collective.save(path_map)
            for name, path in path_map.items():
                print(f"    [{name}] saved to {path}")

        # --- teach ---
        elif cmd.startswith('teach '):
            text = raw[6:].strip()
            if text:
                n = collective.teach_all(text)
                print(f"    Folded into all meshes ({n} chunks each).")

        # --- fold A -> B ---
        elif cmd.startswith('fold ') and ' -> ' in raw:
            parts = raw[5:].split(' -> ', 1)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                if a and b:
                    print(f"    Folding '{a}' -> '{b}' into all meshes...")
                    collective.fold_all(embed(a), embed(b), a, b)
                    print(f"    Done.")

        # --- cross-pollination ---
        elif cmd == 'cross':
            print("  Cross-pollinating...")
            n = collective.cross_pollinate()
            print(f"  {n} attractor patterns shared.")

        # --- normalize ---
        elif cmd == 'normalize':
            for name, ham in meshes.items():
                before = torch.norm(ham.mesh).item()
                ham.normalize_mesh()
                after = torch.norm(ham.mesh).item()
                print(f"    [{name}] energy {before:.1f} → {after:.1f}")
            print("    Mesh energy anchored. Multi-hop diversity restored.")

        # --- dream ---
        elif cmd.startswith('dream'):
            parts = raw.split()
            cycles = 50
            decay  = 0.02   # gentle decay by default during interactive dreams
            if len(parts) > 1:
                try:
                    cycles = int(parts[1])
                except ValueError:
                    pass
            if len(parts) > 2:
                try:
                    decay = float(parts[2])
                except ValueError:
                    pass

            energy_before = {n: h.stats()['energy'] for n, h in meshes.items()}
            folds_before  = {n: h.n_folds           for n, h in meshes.items()}
            results = collective.collective_dream(cycles=cycles, fold_strength=0.1)

            for name, data in results.items():
                col = mesh_color(name, mesh_names)
                print(f"\n  {col}[{name}]{_RESET} attractors:")
                for text, count in data['attractors']:
                    bar = '█' * min(count, 30)
                    print(f"    {count:4d}x {bar} {text[:50]}")
                if data['loops']:
                    print(f"  {col}[{name}]{_RESET} loops: {len(data['loops'])} detected")
                if logger:
                    logger.log_attractor_snapshot(
                        mesh_name=name,
                        attractors=data['attractors'],
                        loops=data['loops'],
                        energy_before=energy_before[name],
                        energy_after=meshes[name].stats()['energy'],
                        new_folds=meshes[name].n_folds - folds_before[name],
                        cycles=cycles,
                    )
            if logger:
                logger.log_cross_pollination(
                    n_shared=sum(len(d['attractors']) for d in results.values()),
                    mesh_pairs=[(a, b, 8) for a in meshes for b in meshes if a != b],
                )

        # --- curious ---
        elif cmd.startswith('curious'):
            parts = raw.split()
            n_gaps = 4
            if len(parts) > 1:
                try:
                    n_gaps = int(parts[1])
                except ValueError:
                    pass

            print(f"\n  Curiosity engine ({n_gaps} gaps)...\n")
            insights = collective.be_curious(n_gaps=n_gaps, questions_per_gap=2)
            print(f"\n  {len(insights)} new insights folded into the collective.")

        # --- trace ---
        elif cmd.startswith('trace '):
            question = raw[6:].strip()
            if not question:
                continue
            print("  Tracing across all meshes (2-hop)...")
            q_emb = embed(question)
            per_mesh = {
                name: ham.trace(q_emb, hops=2, top_k=3)
                for name, ham in meshes.items()
            }
            show_collective_trace(per_mesh, mesh_names)

        # --- deep ---
        elif cmd.startswith('deep '):
            question = raw[5:].strip()
            if not question:
                continue
            print("  Diffracting (2-hop across all meshes)...")
            q_emb    = embed(question)
            blended, activated = collective.collective_resonate(q_emb, hops=2, top_k=8)
            show_collective_resonance(activated, mesh_names)

            context = "\n".join(
                f"[{mesh}|{sim:+.2f}] {text[:120]}"
                for sim, text, mesh in activated if sim > CONFIDENCE_FLOOR
            )
            sys_prompt = (
                "You are the voice of a multi-domain Holographic Associative Memory. "
                "The following patterns activated across specialised meshes:\n\n"
                f"{context}\n\n"
                "Synthesise ONLY from these patterns. Highlight connections "
                "between different domains where you see them."
            ) if context else (
                "The collective had no strong resonance. Answer from your own knowledge."
            )
            response = generate(question, system=sys_prompt)
            print(textwrap.fill(response, width=80,
                                initial_indent="  ", subsequent_indent="  "))

            taught = False
            if activated and activated[0][0] > CONFIDENCE_FLOOR:
                r_emb = embed(response[:400])
                for ham in meshes.values():
                    ham.fold(q_emb, r_emb, strength=SELF_TEACH_STRENGTH)
                    ham.remember(r_emb, response[:250])
                taught = True
            if logger:
                logger.log_query(question, 2, activated, response,
                                 activated[0][0] if activated else 0.0, taught)

        # --- default: 1-hop question ---
        else:
            question = raw
            if cmd.startswith('ask '):
                question = raw[4:].strip()
            if not question:
                continue

            print("  Diffracting across all meshes...")
            q_emb    = embed(question)
            blended, activated = collective.collective_resonate(q_emb, hops=1, top_k=8)
            show_collective_resonance(activated, mesh_names)

            context = "\n".join(
                f"[{mesh}|{sim:+.2f}] {text[:120]}"
                for sim, text, mesh in activated if sim > CONFIDENCE_FLOOR
            )
            sys_prompt = (
                "You are the voice of a multi-domain Holographic Associative Memory. "
                "The following patterns activated across specialised meshes:\n\n"
                f"{context}\n\n"
                "Synthesise ONLY from these activated patterns. "
                "If patterns from different domains activated, find what connects them. "
                "If resonance is weak or tangential, say so honestly."
            ) if context else (
                "The collective had no strong resonance. Answer from your own knowledge."
            )
            response = generate(question, system=sys_prompt)
            print(textwrap.fill(response, width=80,
                                initial_indent="  ", subsequent_indent="  "))

            taught = False
            if activated and activated[0][0] > CONFIDENCE_FLOOR:
                r_emb = embed(response[:400])
                for ham in meshes.values():
                    ham.fold(q_emb, r_emb, strength=SELF_TEACH_STRENGTH)
                    ham.remember(r_emb, response[:250])
                taught = True
            if logger:
                logger.log_query(question, 1, activated, response,
                                 activated[0][0] if activated else 0.0, taught)


if __name__ == "__main__":
    main()
