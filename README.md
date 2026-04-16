# HAMath — Autonomous Mathematical Research System

HAMath is a self-teaching mathematical research system that combines a
**holographic associative memory** with an **LLM-driven conjecture engine** and a
**Lean 4 / Mathlib formal verifier**. It explores the space of mathematics
autonomously: dreaming up conjectures in the gaps between known theorems,
formalizing them in Lean 4, verifying them with Mathlib, and folding each
discovery back into its own growing knowledge base.

Built as an evolution of [HAMesh](https://github.com/mescgit/HAMesh), which
introduced the holographic mesh and scholar loop. HAMath adds the full research
lab cycle with formal verification.

---

## What It Does

```
Known Math Corpus
      ↓
  Embed into 768-D space (all-mpnet-base-v2)
      ↓
  Weave into Holographic Mesh (outer-product superposition)
      ↓
  Scholar "Dreams" (random walks, finds high-novelty gaps)
      ↓
  LLM generates Lean 4 conjecture (bridging the gap)
      ↓
  Lean 4 / Mathlib verifies the statement
    ✓ Verified → fold back into mesh at full strength
    ~ Sorry     → fold back weakly (statement is typed, proof deferred)
    ✗ Failed    → repair loop → try again → sorry-rescue
      ↓
  Mesh grows from its own discoveries → repeat
```

The feedback loop is the key: every verified theorem becomes new knowledge that
shapes future exploration. The mesh develops its own research agenda.

---

## Architecture

| Module | Role |
|---|---|
| `ham_core.py` | `HolographicMesh` — outer-product superposition, Hopfield recall, resonance |
| `ham_embedder.py` | Sentence embedding via `all-mpnet-base-v2` (768-dim float32) |
| `ham_scholar.py` | `MathScholar` — dream loop, novelty scoring, conjecture log |
| `ham_corpus.py` | Corpus builder — Metamath, arXiv, textbooks → mesh |
| `ham_lab.py` | `MathResearchLab` — full autonomous research cycle |
| `ham_lean.py` | `AutoformalEngine` — LLM → Lean 4 → Mathlib verification |
| `setup_mathlib.py` | One-shot Mathlib project setup + cache download |
| `install_lean.py` | Lean 4 binary installer (handles Windows SSL issues) |
| `ham_brain.py` | Brain state — persistent mesh with session memory |
| `ham_analyze.py` | Analysis tools for discovered theorems |
| `ham_experiment.py` | Reproducible experiment runner |
| `ham_verify.py` | Batch verification pipeline |
| `ham_collective.py` | Multi-agent collective mesh |
| `ham_distill.py` | Mesh distillation and compression |
| `ham_logger.py` | Structured research logging |

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch sentence-transformers openai
```

### 2. Install Lean 4

```bash
python install_lean.py
```

### 3. Set up Mathlib (optional but recommended — takes ~15 min on first run)

```bash
python setup_mathlib.py
```

This clones Mathlib v4.29.0, downloads 7800+ prebuilt `.olean` files, and
writes `mathlib_config.json`. Requires Lean 4.29.0 (installed in step 2).

### 4. Build a math corpus mesh

```bash
python ham_corpus.py --builtin --save math_mesh.pt
# or with Metamath:
python ham_corpus.py --metamath --download --save math_mesh.pt
```

### 5. Run the research lab

```bash
# Continuous autonomous discovery (Ctrl-C to stop cleanly)
python ham_lab.py --mesh math_mesh.pt

# Resume a previous session
python ham_lab.py --resume

# Tune the cycle
python ham_lab.py --resume --dream-cycles 800 --formalize-top 5 --attempts 3

# Print the research journal
python ham_lab.py --report
```

### 6. Run the autoformalization engine standalone

```bash
# Generate conjectures first
python ham_scholar.py --mesh math_mesh.pt --log conjectures.json

# Then formalize with Lean verification
python ham_lean.py --conjectures conjectures.json --top 10 --attempts 3
```

---

## How the Holographic Mesh Works

The mesh stores mathematical knowledge as **associative memories** using outer-product
superposition — a technique from Hopfield networks. Every piece of knowledge is
embedded as a 768-dimensional vector. Connections between concepts are encoded as
outer products of their vectors, summed into a single weight matrix.

To recall: give the mesh a partial idea → it resonates through the matrix →
related concepts activate. High-novelty regions (places the mesh keeps pointing
toward but has nothing stored) are where conjectures get generated.

```python
from ham_core import HolographicMesh
from ham_embedder import Embedder

embedder = Embedder()
mesh = HolographicMesh(dim=768)

# Store knowledge
v1 = embedder.embed("The derivative of x^2 is 2x")
v2 = embedder.embed("Power rule for differentiation")
mesh.fold(v1, v2, strength=0.05)

# Recall
query = embedder.embed("derivative of a polynomial")
result, activations = mesh.resonate(query, hops=3)
```

---

## LLM + Lean Verification

HAMath uses any OpenAI-compatible LLM endpoint (LM Studio, Ollama, OpenAI, etc.)
to generate Lean 4 theorem statements. The verification loop:

1. LLM generates a Lean 4 theorem (with imports if in Mathlib mode)
2. `lean` or `lake env lean` checks it
3. Errors are fed back to the LLM for repair (up to N attempts)
4. Final fallback: `sorry-rescue` — replaces the proof body with `sorry`,
   verifying the *statement* is well-typed even if the proof is missing

```python
from ham_lean import AutoformalEngine, make_client, find_lean, load_mathlib_config

client     = make_client("http://localhost:1234/v1")  # LM Studio
lean_bin   = find_lean()
mathlib    = load_mathlib_config()   # None if not set up

engine = AutoformalEngine(client, "your-model-name", lean_bin,
                          max_attempts=3, mathlib_cfg=mathlib)

conjecture = {
    "seed": "triangle inequality in metric spaces",
    "nearest": [(0.92, "metric space axioms"), (0.87, "norm properties")],
    "novelty": 0.74,
    "recurrence": 12,
}

result = engine.process(conjecture)
print(result["verified"])       # True / False
print(result["lean_code"])      # The Lean 4 theorem
print(result["conjecture_text"]) # Plain English statement
```

---

## Mathlib Mode

When `mathlib_config.json` is present (created by `setup_mathlib.py`), the engine
switches to full Mathlib mode: real numbers (`ℝ`), topology, calculus, measure
theory, linear algebra, and 150,000+ lemmas are available.

**Verified working imports (Mathlib v4.29.0):**
```lean
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic  -- also import Data.Real.Basic
import Mathlib.AlgebraicTopology.FundamentalGroupoid.Basic
import Mathlib.NumberTheory.PrimeCounting
import Mathlib.Combinatorics.SimpleGraph.Basic
```

---

## Requirements

- Python 3.11+
- PyTorch (CPU is fine)
- `sentence-transformers`
- `openai` (for LLM client)
- Lean 4.29.0 (`install_lean.py` handles this)
- Mathlib v4.29.0 (optional, `setup_mathlib.py` handles this — ~2 GB disk)
- An OpenAI-compatible LLM endpoint running locally or remotely

**Recommended models:**
- Conjecture generation: `google/gemma-4-27b`, `Qwen2.5-32B-Instruct`
- Lean proof steps: `deepseek-ai/DeepSeek-Prover-V2` (when available locally)

---

## Lab Results (366 cycles)

After 366 autonomous research cycles on a math corpus:

| Metric | Value |
|---|---|
| Total discoveries | 366 |
| Lean-verified theorems | 130 (35%) |
| Sorry proofs (typed, unproven) | 148 (40%) |
| Strongest attractors | Brownian motion, CLT, quantum stochastic |
| Most cited discovery | "Hamiltonian is constant along trajectories" (10 citations) |

The mesh developed its own research agenda — gravitating toward stochastic
processes and quantum systems without being told to care about those areas.

---

## Relationship to HAMesh

[HAMesh](https://github.com/mescgit/HAMesh) is the base system:
holographic mesh, scholar loop, cross-domain discovery. HAMath extends it with:

- **Lean 4 formal verification** — conjectures are checked by a proof assistant
- **Mathlib integration** — access to 150,000+ real mathematical theorems
- **Autonomous research lab** — full feedback loop, journal, mesh growth
- **Sorry rescue** — last-chance statement verification when proofs fail
- **Dual-model architecture** — separate models for conjecture vs. proof steps

---

## License

MIT
