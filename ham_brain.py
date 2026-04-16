"""
HAMesh Brain -- LLM + Holographic Memory

The LLM (Gemma via LM Studio) is the cortex: it reasons, generates, speaks.
The HAMesh is the hippocampus: it stores patterns, surfaces associations,
and is continuously reshaped by what the LLM thinks about.

Each cycle:
  1. User input is embedded
  2. HAMesh surfaces relevant associations (its "intuition")
  3. LLM reasons with those associations as context
  4. LLM response is embedded and folded back into the mesh
  5. Fold strength is proportional to surprise (novel responses fold harder)
  6. Self-mesh records what the brain was attending to each cycle

Over time the mesh develops a persistent internal state shaped by every
conversation. The brain becomes opinionated: it starts pulling toward
patterns it has reinforced, away from patterns it has forgotten.

Usage:
    python ham_brain.py
    python ham_brain.py --mesh combined_mesh.pt
    python ham_brain.py --mesh math_mesh.pt --mesh2 physics_mesh.pt
    python ham_brain.py --url http://192.168.0.183:1234 --model google/gemma-4-26b-a4b
    python ham_brain.py --list-models
    python ham_brain.py --resume    # pick up where you left off
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ham_core import HolographicMesh
from ham_embedder import Embedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LM_STUDIO_URL = "http://192.168.0.183:1234/v1"
DEFAULT_MODEL  = "google/gemma-4-26b-a4b"

SYSTEM_PROMPT = (
    "You are a reasoning system with holographic associative memory.\n\n"
    "Before each message you receive MEMORY ASSOCIATIONS -- concepts your long-term\n"
    "memory is currently activating in response to the conversation. These emerge\n"
    "from patterns the memory has built up over time. They are not random; they\n"
    "reflect deep structural connections the memory has learned to make.\n\n"
    "Treat the associations as your intuition:\n"
    "- They may point at analogies worth unpacking\n"
    "- They may reveal a gap or bridge between domains\n"
    "- They may simply be the memory warming up related concepts\n\n"
    "You do not need to mention every association. Let them shape your thinking\n"
    "the way background knowledge shapes an expert's reasoning -- implicitly.\n\n"
    "Think freely. Reason across domains. Be direct and intellectually honest.\n"
    "If you notice a surprising connection between the associations and the question,\n"
    "follow it. If the associations seem irrelevant, say so and reason from scratch.\n\n"
    "After each exchange your response is folded back into long-term memory. You\n"
    "will gradually develop a persistent internal state shaped by everything you\n"
    "have thought about."
)


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------

class HolographicBrain:
    """
    Integrates a local LLM with HAMesh associative memory.

    math_mesh  : primary knowledge mesh (updated each conversation turn)
    phys_mesh  : optional second domain mesh (queried but not modified)
    self_mesh  : grows during conversation; records attractor snapshots
    """

    def __init__(
        self,
        math_mesh,
        embedder,
        phys_mesh=None,
        lm_url=LM_STUDIO_URL,
        model=DEFAULT_MODEL,
        fold_base=0.05,
        top_k=5,
    ):
        self.math_mesh  = math_mesh
        self.phys_mesh  = phys_mesh
        self.embedder   = embedder
        self.model      = model
        self.fold_base  = fold_base
        self.top_k      = top_k
        self.cycle      = 0
        self.history    = []

        # Self-mesh: records what the brain was attending to each cycle
        self.self_mesh = HolographicMesh(dim=embedder.dim, device=DEVICE)

        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=lm_url, api_key="lm-studio")
        except ImportError:
            print("  ERROR: pip install openai")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Association retrieval
    # ------------------------------------------------------------------

    def _query_associations(self, query_vec):
        """Query all meshes. Returns list of (sim, text, domain) sorted by sim."""
        results = []

        _, math_acts = self.math_mesh.resonate(query_vec, hops=2, top_k=self.top_k)
        for sim, _idx, text in math_acts:
            results.append((sim, text, "math"))

        if self.phys_mesh:
            _, phys_acts = self.phys_mesh.resonate(query_vec, hops=2, top_k=3)
            for sim, _idx, text in phys_acts:
                results.append((sim, text, "physics"))

        # Only query self-mesh after enough history to be meaningful,
        # and cap sim at 0.85 so self-snapshots don't bury knowledge associations
        if len(self.self_mesh.memories) >= 5:
            _, self_acts = self.self_mesh.resonate(query_vec, hops=1, top_k=2)
            for sim, _idx, text in self_acts:
                results.append((min(sim, 0.85), text, "self"))

        results.sort(key=lambda x: -x[0])
        return results[:self.top_k + 4]

    MIN_SIM_TO_INJECT = 0.60   # only inject if at least one association clears this

    def _build_memory_block(self, associations):
        if not associations:
            return ""
        # Filter to associations that are genuinely relevant
        relevant = [(s, t, d) for s, t, d in associations
                    if s >= self.MIN_SIM_TO_INJECT or d == "self"]
        if not relevant:
            return ""   # nothing relevant -- don't pollute the prompt
        lines = ["MEMORY ASSOCIATIONS:"]
        for sim, text, domain in relevant:
            lines.append(f"  [{domain}] {text[:110]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Self-model update
    # ------------------------------------------------------------------

    def _update_self(self, associations, surprise):
        """Fold a snapshot of current attractor state into the self-mesh."""
        if not associations:
            return
        dominant = [text[:50] for _, text, _ in associations[:3]]
        snapshot = (
            f"cycle {self.cycle}: "
            f"attending [{' | '.join(dominant)}] "
            f"surprise={surprise:.2f}"
        )
        vec = self.embedder.embed(snapshot)
        self.self_mesh.fold(vec, vec, strength=0.1)
        self.self_mesh.remember(vec, snapshot)

    # ------------------------------------------------------------------
    # Core cognition cycle
    # ------------------------------------------------------------------

    def think(self, user_input):
        """
        One full cognition cycle. Streams LLM output to stdout.
        Returns the full response string.
        """
        self.cycle += 1

        # 1. Embed input
        input_vec = self.embedder.embed(user_input)

        # 2. Surface associations
        associations = self._query_associations(input_vec)
        memory_block = self._build_memory_block(associations)

        # 3. Mesh prediction (what does the mesh expect the answer to be?)
        predicted_vec = self.math_mesh.diffract(input_vec, hops=2)

        # 4. Print what memory is activating
        print("\n  [Holographic memory activating]")
        for sim, text, domain in associations[:6]:
            print(f"    [{domain}] {sim:.3f}  {text[:80]}")
        print()

        # 5. Build messages
        augmented = (
            memory_block + "\n\n---\n" + user_input
            if memory_block else user_input
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-6:]
        messages.append({"role": "user", "content": augmented})

        # 6. Stream LLM response
        print("  Brain: ", end="", flush=True)
        response_text = ""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=1024,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                response_text += delta
        except Exception as e:
            print("\n  [stream error: {}]".format(e))

        # Fallback: if streaming returned nothing, retry without streaming
        if not response_text.strip():
            print("\n  [empty stream — retrying without streaming...]  ", end="", flush=True)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    temperature=0.7,
                    max_tokens=1024,
                )
                response_text = resp.choices[0].message.content or ""
                print(response_text, end="")
            except Exception as e:
                print("\n  [LLM error: {}]".format(e))
                return ""
        print("\n")

        if not response_text.strip():
            return response_text

        # 7. Embed response
        response_vec = self.embedder.embed(response_text[:512])

        # 8. Surprise = how far from what the mesh predicted
        surprise = max(0.0, 1.0 - F.cosine_similarity(
            predicted_vec.unsqueeze(0), response_vec.unsqueeze(0)
        ).item())
        # Fold strength: always at least 30% base, up to 100% when fully surprising
        fold_strength = self.fold_base * (0.3 + 0.7 * surprise)

        # 9. Fold response into math mesh
        self.math_mesh.fold(input_vec, response_vec, strength=fold_strength)
        self.math_mesh.remember(response_vec, "[thought] " + response_text[:120])

        # 10. Update self-model
        self._update_self(associations, surprise)

        # 11. Update conversation history — truncate stored text to keep context lean
        self.history.append({"role": "user",      "content": user_input[:400]})
        self.history.append({"role": "assistant", "content": response_text[:600]})

        print(
            "  [surprise={:.3f}  fold={:.4f}  cycle={}  memories={}]".format(
                surprise, fold_strength, self.cycle, len(self.math_mesh.memories)
            )
        )

        return response_text

    # ------------------------------------------------------------------
    # Sleep / consolidation
    # ------------------------------------------------------------------

    def dream(self, cycles=200):
        """Offline consolidation: mesh dreams on accumulated memories."""
        print("\n  [Dreaming for {} cycles...]".format(cycles))
        import io, contextlib
        from ham_scholar import MathScholar
        # Suppress Scholar's capacity/tip warnings — they aren't relevant here
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            scholar = MathScholar(self.math_mesh)
            scholar.dream_and_discover(total_cycles=cycles, verbose=False)
        n = len(scholar.log.entries)
        print("  [Done. {} new attractor pattern{} found.  "
              "memories={}  energy={:.1f}]".format(
            n, "s" if n != 1 else "",
            len(self.math_mesh.memories),
            self.math_mesh.mesh.norm().item(),
        ))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path="brain_state.pt"):
        torch.save({
            "cycle":   self.cycle,
            "history": self.history[-20:],
            "math_mesh": {
                "mesh":     self.math_mesh.mesh.cpu(),
                "n_folds":  self.math_mesh.n_folds,
                "memories": [(e.cpu(), t) for e, t in self.math_mesh.memories],
                "dim":      self.math_mesh.dim,
            },
            "self_mesh": {
                "mesh":     self.self_mesh.mesh.cpu(),
                "n_folds":  self.self_mesh.n_folds,
                "memories": [(e.cpu(), t) for e, t in self.self_mesh.memories],
                "dim":      self.self_mesh.dim,
            },
        }, path)
        print("\n  [Saved -> {}  ({} math memories, {} self-model snapshots)]".format(
            path, len(self.math_mesh.memories), len(self.self_mesh.memories)
        ))

    @classmethod
    def load_state(cls, path, base_mesh, embedder, **kwargs):
        """Resume from a saved brain_state.pt."""
        data = torch.load(path, map_location="cpu", weights_only=False)

        mm = data["math_mesh"]
        base_mesh.mesh     = mm["mesh"].to(base_mesh.device)
        base_mesh.n_folds  = mm["n_folds"]
        base_mesh.memories = [(e.to(base_mesh.device), t) for e, t in mm["memories"]]

        brain = cls(base_mesh, embedder, **kwargs)
        brain.cycle   = data["cycle"]
        # Trim any previously saved oversized history entries
        brain.history = [
            {"role": m["role"], "content": m["content"][:600]}
            for m in data["history"]
        ]

        sm = data["self_mesh"]
        brain.self_mesh.mesh     = sm["mesh"].to(DEVICE)
        brain.self_mesh.n_folds  = sm["n_folds"]
        brain.self_mesh.memories = [(e.to(DEVICE), t) for e, t in sm["memories"]]

        print("  [Resumed: cycle={}, math_memories={}, self_memories={}]".format(
            brain.cycle, len(brain.math_mesh.memories), len(brain.self_mesh.memories)
        ))
        return brain


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HAMesh Brain: LLM + holographic memory REPL"
    )
    parser.add_argument("--mesh",    default="math_mesh.pt")
    parser.add_argument("--mesh2",   default="physics_mesh.pt",
                        help="Physics mesh -- loaded if the file exists")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--url",     default=LM_STUDIO_URL)
    parser.add_argument("--state",   default="brain_state.pt",
                        help="Save/load path for persistent brain state")
    parser.add_argument("--top-k",   type=int, default=5)
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from --state file if it exists")
    parser.add_argument("--list-models", action="store_true",
                        help="List models available in LM Studio and exit")
    args = parser.parse_args()

    if args.list_models:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=args.url, api_key="lm-studio")
            models = client.models.list()
            print("\nModels at {}:".format(args.url))
            for m in models.data:
                print("  {}".format(m.id))
        except Exception as e:
            print("  Error connecting to LM Studio: {}".format(e))
        return

    print("\n" + "=" * 60)
    print("  HAMesh Brain -- LLM + Holographic Memory")
    print("=" * 60)

    embedder = Embedder()

    print("\n  Loading math mesh: {}".format(args.mesh))
    math_mesh = HolographicMesh.load(args.mesh, device=DEVICE)
    s = math_mesh.stats()
    print("  {} memories, energy={:.1f}".format(s["memories"], s["energy"]))

    phys_mesh = None
    if Path(args.mesh2).exists():
        print("  Loading physics mesh: {}".format(args.mesh2))
        phys_mesh = HolographicMesh.load(args.mesh2, device=DEVICE)
        s2 = phys_mesh.stats()
        print("  {} memories, energy={:.1f}".format(s2["memories"], s2["energy"]))

    if args.resume and Path(args.state).exists():
        brain = HolographicBrain.load_state(
            args.state, math_mesh, embedder,
            phys_mesh=phys_mesh, lm_url=args.url,
            model=args.model, top_k=args.top_k,
        )
    else:
        brain = HolographicBrain(
            math_mesh=math_mesh, phys_mesh=phys_mesh, embedder=embedder,
            lm_url=args.url, model=args.model, top_k=args.top_k,
        )

    print("\n  Model  : {}".format(args.model))
    print("  Server : {}".format(args.url))
    print("\n  Commands: 'dream' | 'save' | 'state' | 'quit'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower().startswith("dream"):
            parts = user_input.split()
            n = 200
            for p in parts[1:]:
                if p.isdigit():
                    n = int(p)
                    break
            brain.dream(cycles=n)
            continue
        if user_input.lower() == "save":
            brain.save(args.state)
            continue
        if user_input.lower() == "state":
            s = brain.math_mesh.stats()
            print("\n  Math mesh  : {} memories, {} folds, energy={:.1f}".format(
                s["memories"], s["folds"], s["energy"]
            ))
            print("  Self mesh  : {} snapshots".format(len(brain.self_mesh.memories)))
            print("  Cycle      : {}".format(brain.cycle))
            continue

        brain.think(user_input)

    brain.save(args.state)
    print("  Goodbye.")


if __name__ == "__main__":
    main()
