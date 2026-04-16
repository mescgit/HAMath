"""
Holographic Associative Memory — Core Engine

The HAM stores knowledge as interference patterns in a dense matrix.
It doesn't search — it diffracts. Query vectors propagate through
the mesh like light through a hologram, and the answer constructively
interferes into existence.

This operates in a real LLM's embedding space. The LLM is the
sensory organ (eyes/mouth). The HAM is the brain.
"""

import torch
import torch.nn.functional as F


class HolographicMesh:
    """
    A holographic associative memory that operates in an LLM's
    native embedding space.

    Knowledge is stored via Hebbian superposition (outer products).
    Retrieval is done via diffraction (matrix-vector multiply + threshold).
    Multi-hop reasoning chains diffractions to follow transitive associations.
    """

    def __init__(self, dim, device='cuda'):
        self.dim = dim
        self.device = device
        self.mesh = torch.zeros((dim, dim), device=device, dtype=torch.float32)
        self.n_folds = 0

        # Stored embeddings + text for decoding HAM outputs back to language
        self.memories = []  # list of (embedding_tensor, text_string)

        # Target energy ceiling — normalize_mesh() rescales to this value.
        # Keeps dominant attractors from drowning out everything else.
        self.target_energy = None  # set automatically on first normalize call

    def _prepare(self, vec):
        """L2-normalize to the unit sphere — preserves semantic structure."""
        v = vec.float().to(self.device)
        return F.normalize(v, dim=0)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def fold(self, key_vec, val_vec, strength=1.0):
        """
        Fold a key->value association into the mesh via outer product.

        M += strength * (phase(key) ⊗ phase(val))

        Multiple folds superimpose in the same physical space.
        """
        k = self._prepare(key_vec)
        v = self._prepare(val_vec)
        self.mesh += strength * torch.outer(k, v)
        self.n_folds += 1

    def remember(self, embedding, text):
        """Register an embedding-text pair for later decoding."""
        self.memories.append((embedding.to(self.device).detach(), text))

    def learn(self, key_emb, val_emb, key_text, val_text, strength=1.0):
        """Store + fold in one shot. Registers both sides and their link."""
        self.remember(key_emb, key_text)
        self.remember(val_emb, val_text)
        self.fold(key_emb, val_emb, strength=strength)

    # ------------------------------------------------------------------
    # Retrieval / Reasoning
    # ------------------------------------------------------------------

    def diffract(self, query_vec, hops=1, temperature=1.0):
        """
        Shine a query through the mesh.  The answer emerges via
        constructive interference.

        hops > 1 enables transitive reasoning:
          hop 1: A -> B  (direct association)
          hop 2: B -> C  (the HAM follows the chain)

        Returns the raw continuous output so we can read magnitudes.
        """
        signal = self._prepare(query_vec)

        for _ in range(hops):
            raw = torch.matmul(signal, self.mesh)
            # L2 normalize to stay on the unit sphere — preserves direction,
            # prevents explosion, and keeps multi-hop stable
            signal = F.normalize(raw, dim=0)

        return signal

    def resonate(self, query_vec, hops=1, temperature=1.0, top_k=5):
        """
        Diffract and decode: returns which stored memories "lit up"
        and how strongly.

        Returns
        -------
        intuition : Tensor   — the HAM's synthesised output vector
        activated : list of (similarity_float, memory_index, text)
        """
        intuition = self.diffract(query_vec, hops=hops, temperature=temperature)

        if not self.memories:
            return intuition, []

        stored = torch.stack([e for e, _ in self.memories]).to(self.device)
        sims = F.cosine_similarity(intuition.unsqueeze(0), stored)

        # Rank by resonance strength
        k = min(top_k, len(self.memories))
        topk_vals, topk_idx = torch.topk(sims, k)

        activated = []
        for i in range(k):
            idx = topk_idx[i].item()
            sim = topk_vals[i].item()
            activated.append((sim, idx, self.memories[idx][1]))

        return intuition, activated

    # ------------------------------------------------------------------
    # Multi-hop trace  (shows the reasoning path)
    # ------------------------------------------------------------------

    def trace(self, query_vec, hops=2, temperature=1.0, top_k=3):
        """
        Run diffraction hop-by-hop and return the activation pattern
        at each step.  This lets you *see* the HAM's chain of thought.

        Returns list of (hop_number, [(sim, idx, text), ...])
        """
        signal = self._prepare(query_vec)
        path = []

        for hop in range(1, hops + 1):
            raw = torch.matmul(signal, self.mesh)
            signal = F.normalize(raw, dim=0)

            if self.memories:
                stored = torch.stack([e for e, _ in self.memories]).to(self.device)
                sims = F.cosine_similarity(signal.unsqueeze(0), stored)
                k = min(top_k, len(self.memories))
                topk_vals, topk_idx = torch.topk(sims, k)
                step = []
                for i in range(k):
                    idx = topk_idx[i].item()
                    step.append((topk_vals[i].item(), idx, self.memories[idx][1]))
                path.append((hop, step))

        return path

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def find_isolated(self, top_k=10):
        """
        Find memories that have few strong connections to others —
        knowledge islands that exist but haven't been integrated.

        Isolation is measured as the average cosine similarity to the
        3 nearest neighbours.  Low score = isolated = prime curiosity target.

        Returns list of (isolation_score, memory_idx, text), lowest first.
        """
        if len(self.memories) < 4:
            return []

        stored = torch.stack([e for e, _ in self.memories]).to(self.device)
        normed  = F.normalize(stored, dim=1)

        sim_matrix = torch.mm(normed, normed.t())
        sim_matrix.fill_diagonal_(float('-inf'))

        k_neighbors = min(3, len(self.memories) - 1)
        top_sims, _ = sim_matrix.topk(k_neighbors, dim=1)
        connectedness = top_sims.mean(dim=1)

        k = min(top_k, len(self.memories))
        scores, idx = connectedness.topk(k, largest=False)

        return [(scores[i].item(), idx[i].item(),
                 self.memories[idx[i]][1]) for i in range(k)]

    def novelty_score(self, output_vec) -> float:
        """
        Measure how far an output vector is from all stored memories.

        novelty = 1 - max_cosine_similarity(output, all_stored)

        0.0 = output perfectly matches a known memory (fully understood)
        1.0 = output is orthogonal to all stored memories (completely novel)

        Use this after diffraction to detect when the mesh is pointing
        at an unexplored region between known concepts — a conjecture.
        """
        if not self.memories:
            return 1.0
        stored = torch.stack([e for e, _ in self.memories]).to(self.device)
        sims = F.cosine_similarity(output_vec.unsqueeze(0), stored)
        return 1.0 - sims.max().item()

    def find_novel_regions(self, n_probes=50, threshold=0.35, hops=2):
        """
        Probe the mesh systematically to find high-novelty output regions.

        Each stored memory is used as a seed; the mesh diffracts it and
        measures how novel the output is. High novelty = the mesh is pointing
        somewhere between known concepts — a potential conjecture zone.

        Returns list of (novelty_score, seed_text, nearest_neighbors)
        sorted by novelty descending.
        """
        if not self.memories:
            return []

        results = []
        n_probes = min(n_probes, len(self.memories))
        probe_indices = list(range(len(self.memories)))

        stored = torch.stack([e for e, _ in self.memories]).to(self.device)

        for idx in probe_indices[:n_probes]:
            seed_vec, seed_text = self.memories[idx]
            output = self.diffract(seed_vec, hops=hops)
            score = self.novelty_score(output)

            if score >= threshold:
                # Find nearest known memories to this novel output
                sims = F.cosine_similarity(output.unsqueeze(0), stored)
                top_vals, top_idx = torch.topk(sims, min(3, len(self.memories)))
                neighbors = [
                    (top_vals[i].item(), self.memories[top_idx[i].item()][1])
                    for i in range(len(top_vals))
                ]
                results.append((score, seed_text, neighbors))

        results.sort(key=lambda x: -x[0])
        return results

    def dominant_memories(self, n=10):
        """
        Find memories most aligned with the mesh's dominant direction.
        These are the current attractor kings — what to share with other meshes.
        Uses power iteration to find the top eigenvector of the mesh.
        """
        if not self.memories:
            return []

        v = F.normalize(torch.randn(self.dim, device=self.device), dim=0)
        for _ in range(20):
            v = F.normalize(torch.matmul(self.mesh, v), dim=0)

        stored = torch.stack([e for e, _ in self.memories]).to(self.device)
        sims   = F.cosine_similarity(v.unsqueeze(0), stored)
        k      = min(n, len(self.memories))
        vals, idx = torch.topk(sims, k)

        return [(vals[i].item(), idx[i].item(),
                 self.memories[idx[i]][1],
                 self.memories[idx[i]][0]) for i in range(k)]

    # ------------------------------------------------------------------
    # Energy management
    # ------------------------------------------------------------------

    def normalize_mesh(self, target_energy=None):
        """
        Rescale the mesh so its Frobenius norm equals target_energy.

        Without this, repeated dreaming causes attractor collapse:
        dominant eigenvectors accumulate energy until every diffraction
        immediately falls into one of ~5 gravitational wells, killing
        multi-hop diversity and making the mesh nearly useless.

        Call this periodically (or pass decay_every to dream()) to keep
        the mesh in a healthy distributed state.

        If target_energy is None, uses self.target_energy (set on first
        call to the energy at that moment — a good baseline).
        """
        current = torch.norm(self.mesh).item()
        if current == 0:
            return current

        if target_energy is None:
            if self.target_energy is None:
                # First call: anchor to current energy as the ceiling
                self.target_energy = current
            target_energy = self.target_energy

        self.mesh = self.mesh * (target_energy / current)
        return current

    def apply_decay(self, decay=0.98):
        """
        Soft exponential decay: mesh *= decay.

        Older folds fade; recent ones remain strong.
        Makes the mesh more plastic — new learning can shift attractors
        instead of being swamped by the accumulated history.

        decay=0.98 → halves energy every ~34 dream cycles
        decay=0.99 → halves energy every ~69 dream cycles
        """
        self.mesh *= decay

    # ------------------------------------------------------------------
    # Dreaming — recursive self-modification
    # ------------------------------------------------------------------

    def dream(self, cycles=50, fold_strength=0.1, reseed_every=10,
              decay=0.0, decay_every=20):
        """
        The HAM queries itself, folds its own outputs back in,
        and develops self-perpetuating attractor loops.

        Each cycle:
          1. Seed: pick a memory (or reuse last cycle's output)
          2. Diffract through the mesh
          3. Fold the output back in — the mesh modifies itself
          4. Use the output as next cycle's seed — the loop

        Over many cycles, certain patterns keep activating each
        other in loops.  These are the "strange attractors" —
        self-reinforcing behavioral basins the mesh falls into.

        decay : float
            Per-application decay factor (0 = no decay, 0.98 = gentle).
            Applied every decay_every cycles to prevent attractor collapse.
            Recommended: 0.98 for long runs, 0.0 for short exploratory runs.
        decay_every : int
            How often (in cycles) to apply decay.

        Returns
        -------
        log : list of dicts (per-cycle records)
        attractors : list of (text, count) sorted by frequency
        loops : list of detected activation cycles
        """
        import random

        log = []
        activation_counts = {}  # which memories appear most
        recent_trail = []       # last N top activations for loop detection
        detected_loops = []

        signal = None

        for cycle in range(cycles):
            # --- Seed selection ---
            if signal is None or cycle % reseed_every == 0:
                idx = random.randint(0, len(self.memories) - 1)
                signal = self.memories[idx][0]
                seed_text = self.memories[idx][1][:60]
            else:
                seed_text = "(self)"

            # --- Diffract ---
            intuition, activated = self.resonate(signal, hops=1, top_k=5)

            if not activated:
                continue

            top_sim, top_idx, top_text = activated[0]
            top_key = top_text[:60]

            # --- Track activations for attractor detection ---
            for sim, idx, text in activated:
                key = text[:60]
                activation_counts[key] = activation_counts.get(key, 0) + 1

            # --- Loop detection ---
            recent_trail.append(top_key)
            if len(recent_trail) > 20:
                recent_trail.pop(0)
            # Check if the current top activation appeared recently
            trail_window = recent_trail[:-1]
            if top_key in trail_window:
                loop_start = len(trail_window) - 1 - trail_window[::-1].index(top_key)
                loop_seq = recent_trail[loop_start:]
                if len(loop_seq) >= 2 and loop_seq not in detected_loops:
                    detected_loops.append(loop_seq[:])

            # --- Self-modification: fold outputs back into the mesh ---

            # 1. Bridge: connect seed to what emerged
            self.fold(signal, intuition, strength=fold_strength)

            # 2. Self-reference: the output learns to recognize itself
            #    This is what creates the self-perpetuating aspect
            self.fold(intuition, intuition, strength=fold_strength * 0.3)

            # 3. Cross-pollinate top 2 activations — forces unexpected bridges
            if len(activated) >= 2:
                e1 = self.memories[activated[0][1]][0]
                e2 = self.memories[activated[1][1]][0]
                self.fold(e1, e2, strength=fold_strength * 0.5)

            # --- Decay: prevent attractor collapse on long runs ---
            if decay > 0 and cycle > 0 and cycle % decay_every == 0:
                self.apply_decay(1.0 - decay)

            # --- Record ---
            log.append({
                'cycle': cycle,
                'seed': seed_text,
                'top_3': [(s, t[:50]) for s, _, t in activated[:3]],
                'energy': torch.norm(self.mesh).item(),
            })

            # --- The loop: output becomes next input ---
            signal = intuition

        # Sort attractors by frequency
        attractors = sorted(activation_counts.items(), key=lambda x: -x[1])

        return log, attractors, detected_loops

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            'mesh': self.mesh.cpu(),
            'n_folds': self.n_folds,
            'memories': [(e.cpu(), t) for e, t in self.memories],
            'dim': self.dim,
        }, path)

    @classmethod
    def load(cls, path, device='cuda'):
        data = torch.load(path, map_location='cpu', weights_only=False)
        ham = cls(data['dim'], device=device)
        ham.mesh = data['mesh'].to(device)
        ham.n_folds = data['n_folds']
        ham.memories = [(e.to(device), t) for e, t in data['memories']]
        return ham

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self):
        energy = torch.norm(self.mesh).item()
        sparsity = (self.mesh == 0).float().mean().item()
        return {
            'dim': self.dim,
            'folds': self.n_folds,
            'memories': len(self.memories),
            'energy': energy,
            'sparsity': sparsity,
        }
