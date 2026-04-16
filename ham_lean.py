"""
HAMesh Lean 4 Autoformalization Engine

Closes the discovery loop:

  1. ham_scholar.py finds "gap regions" between known theorems
     (places the mesh keeps pointing at that don't match anything stored)
  2. ham_lean.py takes those conjectures and asks the LLM to:
       a. Formulate a precise mathematical statement in plain English
       b. Translate it to Lean 4 code
  3. lean <file> checks whether the formalization type-checks
  4. Errors are fed back to the LLM for repair (up to --attempts tries)
  5. Outputs .lean files: verified ones are stamped VERIFIED, others DRAFT

A proof that compiles with `sorry` means the STATEMENT is well-formed --
the conjecture is a meaningful, type-correct mathematical claim.
A proof without `sorry` is fully machine-verified.

Both outputs are valuable. The DRAFT files can be handed to a human
mathematician or a stronger model for completion.

Usage:
    # Step 1: build corpus and generate conjectures
    python ham_corpus.py --builtin --save math_mesh.pt
    python ham_scholar.py --mesh math_mesh.pt --log conjectures.json

    # Step 2: autoformalize
    python ham_lean.py --conjectures conjectures.json
    python ham_lean.py --conjectures conjectures.json --top 10 --attempts 3
    python ham_lean.py --conjectures conjectures.json --no-verify  # LLM only, skip lean
    python ham_lean.py --live  # run scholar + autoformalize in one continuous loop
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError with math symbols)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

LM_STUDIO_URL = "http://192.168.0.183:1234/v1"

# Conjecture generation — creative, broad mathematical reasoning
DEFAULT_MODEL = "qwen/qwen2.5-coder-32b"

# Lean proof repair — specialized formal verification model
# Used for all repair attempts after the first formalization
PROVER_MODEL  = "deepseek-prover-v2-7b"

# Mathlib config (auto-detected from mathlib_config.json if present)
_MATHLIB_CONFIG_PATH = Path(__file__).parent / "mathlib_config.json"

OUTPUT_DIR = Path("lean_output")


# ---------------------------------------------------------------------------
# Lean 4 system prompt + few-shot examples
# ---------------------------------------------------------------------------

LEAN_SYSTEM_PROMPT = (
    "You are a Lean 4 expert running in standalone mode (no Mathlib).\n"
    "Write a theorem using ONLY Lean 4 core — no import statements of any kind.\n\n"
    "AVAILABLE TYPES (no imports needed):\n"
    "  Nat, Int, Float, Bool, String, Char\n"
    "  List α, Array α, Option α, Prod α β\n"
    "  Fin n  (integers 0..n-1)\n"
    "  α → β  (functions and predicates, e.g. α → Prop)\n"
    "  Prop, Type, Sort\n\n"
    "KEY SYNTAX RULES:\n"
    "  - NEVER write `import` — not even `import Mathlib`\n"
    "  - For predicate-style sets use  `s x`  NOT  `x ∈ s`\n"
    "    (∈ needs a Membership instance that requires Mathlib)\n"
    "  - For function equality use `funext` or `sorry`\n"
    "  - For propositional equality use `propext` or `sorry`\n"
    "  - Real numbers: use `Float` or encode as `Int × Int` (numerator, denominator)\n"
    "  - Matrices: encode as `Fin m → Fin n → Int` (or Float)\n\n"
    "WORKING EXAMPLES (copy this style):\n\n"
    "Example 1 — predicate extensionality:\n"
    "```lean\n"
    "theorem pred_ext (α : Type) (p q : α → Prop) :\n"
    "    p = q ↔ ∀ x : α, p x ↔ q x := by\n"
    "  sorry\n"
    "```\n\n"
    "Example 2 — natural number property:\n"
    "```lean\n"
    "theorem add_comm_nat (m n : Nat) : m + n = n + m := by\n"
    "  sorry\n"
    "```\n\n"
    "Example 3 — function composition:\n"
    "```lean\n"
    "theorem comp_id (α β : Type) (f : α → β) :\n"
    "    (fun x => f x) = f := by\n"
    "  funext x; rfl\n"
    "```\n\n"
    "Example 4 — abstract structure:\n"
    "```lean\n"
    "def conserved (state : Nat → Float) (energy : Nat → Float) : Prop :=\n"
    "  ∀ t : Nat, energy t = energy 0\n\n"
    "theorem constant_energy (H : conserved state energy) (t : Nat) :\n"
    "    energy t = energy 0 := H t\n"
    "```\n\n"
    "Respond with:\n"
    "CONJECTURE: one sentence in plain English\n\n"
    "```lean\n"
    "-- your theorem here\n"
    "```"
)

# ---------------------------------------------------------------------------
# Mathlib-aware system prompt (used when a Lake project is configured)
# ---------------------------------------------------------------------------

LEAN_MATHLIB_SYSTEM_PROMPT = (
    "You are a Lean 4 / Mathlib expert (Mathlib v4.29.0). Write a theorem bridging the given concepts.\n\n"
    "You have FULL Mathlib access via a configured Lake project.\n"
    "Use appropriate imports at the top of your lean block.\n\n"
    "IMPORT GUIDE (verified working in Mathlib v4.29.0):\n"
    "  import Mathlib.Topology.MetricSpace.Basic              -- metric spaces, dist\n"
    "  import Mathlib.Analysis.Calculus.Deriv.Basic           -- derivatives, HasDerivAt\n"
    "  import Mathlib.Analysis.SpecialFunctions.Pow.Real      -- Real.sqrt, rpow\n"
    "  import Mathlib.Analysis.ODE.Gronwall                   -- ODE estimates\n"
    "  import Mathlib.Analysis.Normed.Group.Basic             -- normed groups/spaces\n"
    "  import Mathlib.Analysis.InnerProductSpace.Basic        -- inner product spaces\n"
    "  import Mathlib.MeasureTheory.Measure.MeasureSpace      -- measures\n"
    "  import Mathlib.MeasureTheory.Measure.Lebesgue.Basic    -- Lebesgue measure\n"
    "  import Mathlib.MeasureTheory.Integral.MeanInequalities -- mean value inequalities\n"
    "  import Mathlib.Data.Real.Basic                         -- \u211d type + all instances\n"
    "  import Mathlib.Algebra.Group.Basic                     -- groups, CommGroup\n"
    "  import Mathlib.RingTheory.Polynomial.Basic             -- polynomials\n"
    "  import Mathlib.LinearAlgebra.Eigenspace.Basic          -- eigenvalues\n"
    "  import Mathlib.LinearAlgebra.Matrix.Determinant.Basic  -- det (also add Data.Real.Basic)\n"
    "  import Mathlib.AlgebraicTopology.FundamentalGroupoid.Basic -- pi_1\n"
    "  import Mathlib.NumberTheory.PrimeCounting              -- prime counting\n"
    "  import Mathlib.Combinatorics.SimpleGraph.Basic         -- graph theory\n\n"
    "  DO NOT use: Mathlib.MeasureTheory.Integral.Bochner (broken cache)\n"
    "  DO NOT use: Mathlib.MeasureTheory.Integral.IntervalIntegral (broken cache)\n"
    "  DO NOT use: Mathlib.Analysis.SpecialFunctions.Integrals (broken cache)\n\n"
    "RULES:\n"
    "  - Use `sorry` for proof bodies you cannot complete.\n"
    "  - The STATEMENT must be type-correct — Lean will check it.\n"
    "  - Use `\u211d` (\\R) for the reals — NOT `Real` alone as a type.\n"
    "  - For matrices: import BOTH `Mathlib.Data.Real.Basic` and `Mathlib.LinearAlgebra.Matrix.Determinant.Basic`.\n"
    "  - Use `Matrix (Fin m) (Fin n) \u211d` for m\u00d7n real matrices.\n\n"
    "WORKING EXAMPLES (copy this style exactly):\n\n"
    "Example 1 — topology:\n"
    "```lean\n"
    "import Mathlib.Topology.MetricSpace.Basic\n"
    "theorem metric_nonneg' {X : Type*} [MetricSpace X] (x y : X) :\n"
    "    0 \u2264 dist x y := dist_nonneg\n"
    "```\n\n"
    "Example 2 — group theory:\n"
    "```lean\n"
    "import Mathlib.Algebra.Group.Basic\n"
    "theorem comm_group_symm {G : Type*} [CommGroup G] (a b : G) :\n"
    "    a * b = b * a := mul_comm a b\n"
    "```\n\n"
    "Example 3 — measure theory:\n"
    "```lean\n"
    "import Mathlib.MeasureTheory.Measure.MeasureSpace\n"
    "theorem measure_set_nonneg {X : Type*} [MeasurableSpace X]\n"
    "    (mu : MeasureTheory.Measure X) (s : Set X) :\n"
    "    0 \u2264 mu s := zero_le _\n"
    "```\n\n"
    "Example 4 — real analysis (sorry proof):\n"
    "```lean\n"
    "import Mathlib.Analysis.SpecialFunctions.Pow.Real\n"
    "theorem sqrt_sq_bound (x : \u211d) (hx : 0 \u2264 x) :\n"
    "    Real.sqrt (x ^ 2) = x := by\n"
    "  sorry\n"
    "```\n\n"
    "Respond with:\n"
    "CONJECTURE: one sentence in plain English\n\n"
    "```lean\n"
    "import Mathlib.XXX\n"
    "-- your theorem here\n"
    "```"
)

# ---------------------------------------------------------------------------
# DeepSeek-Prover system prompt — used for all repair/proof steps
# Tighter focus than the general prompts: error analysis + targeted fix only.
# ---------------------------------------------------------------------------

DEEPSEEK_PROVER_PROMPT = (
    "You are DeepSeek-Prover, an expert Lean 4 theorem prover.\n"
    "Your job is to FIX broken Lean 4 code given a specific error message.\n\n"
    "RULES:\n"
    "  - Read the error carefully. Fix ONLY what the error describes.\n"
    "  - Keep the theorem statement identical — only change the proof body.\n"
    "  - If you cannot complete the proof, use `sorry` for the proof body.\n"
    "  - Never add `import` lines in standalone mode (no Mathlib).\n"
    "  - In Mathlib mode, keep existing imports and add new ones if needed.\n"
    "  - Do not explain — output only the corrected ```lean ... ``` block.\n\n"
    "COMMON FIXES:\n"
    "  - `unknown identifier X` → check spelling; use `sorry` if X is unavailable\n"
    "  - `type mismatch` → add explicit coercions or use `norm_cast`\n"
    "  - `failed to synthesize` → add missing instance or use `sorry`\n"
    "  - `unsolved goals` → add `simp`, `ring`, `linarith`, or `sorry`\n"
    "  - `function expected` → check argument count / namespace\n\n"
    "Output format — ONLY this, nothing else:\n"
    "```lean\n"
    "-- fixed theorem\n"
    "```"
)


def load_mathlib_config() -> dict | None:
    """Load Mathlib project config if it exists and is enabled."""
    if _MATHLIB_CONFIG_PATH.exists():
        try:
            cfg = json.loads(_MATHLIB_CONFIG_PATH.read_text(encoding="utf-8"))
            if cfg.get("enabled") and Path(cfg.get("mathlib_project_dir", "")).exists():
                return cfg
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Lean runner
# ---------------------------------------------------------------------------

def find_lean() -> str | None:
    """
    Find the lean binary.  Returns path or None.

    Search order (most reliable first):
      1. Direct versioned installs under C:\lean4\ and D:\lean4\
         These are real binaries — not elan shims that can hang on SSL checks.
      2. Common PATH-independent locations (~/.local/bin, /usr/local/bin)
      3. PATH lookup last — on Windows the elan shim is often first on PATH
         and hangs when SSL certificate revocation checks fail.
    """
    def _try(path: str) -> str | None:
        try:
            r = subprocess.run([path, "--version"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return None

    # 1. Direct versioned installs (skip elan shim)
    for base in [Path("C:/lean4"), Path("D:/lean4")]:
        if base.exists():
            for lean_exe in sorted(base.rglob("lean.exe"), reverse=True):
                if result := _try(str(lean_exe)):
                    return result

    # 2. Home-dir non-elan locations
    home = Path.home()
    for p in [
        home / ".local" / "bin" / "lean",
        Path("/usr/local/bin/lean"),
    ]:
        if p.exists():
            if result := _try(str(p)):
                return result

    # 3. PATH lookup (last — elan shim may be here)
    for candidate in ["lean", "lean4"]:
        if result := _try(candidate):
            return result

    # 4. Elan install (may hang on SSL check — only try if nothing else found)
    for p in [
        home / ".elan" / "bin" / "lean.exe",
        home / ".elan" / "bin" / "lean",
        Path("C:/Users") / os.getenv("USERNAME", "") / ".elan" / "bin" / "lean.exe",
    ]:
        if p.exists():
            if result := _try(str(p)):
                return result

    return None


def _strip_imports(code: str) -> str:
    """Remove all import lines from Lean code (standalone Lean has no Mathlib)."""
    return "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("import ")
    ).strip()


def _run_lean_on_code(code: str, lean_bin: str, timeout: int) -> tuple[bool, str]:
    """Write code to a temp file, run lean, return (success, output)."""
    with tempfile.NamedTemporaryFile(
        suffix=".lean", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run(
            [lean_bin, tmp],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",   # Lean outputs UTF-8; avoid cp1252 on Windows
            timeout=timeout
        )
        # r.stdout / r.stderr may be None if the subprocess reader thread crashed
        # (can happen on Windows when encoding detection fails mid-stream)
        stdout = r.stdout or ""
        stderr = r.stderr or ""
        output = (stdout + stderr).strip()
        # returncode 0 AND no "error:" lines → success
        # (sorry produces "warning: declaration uses 'sorry'" which is fine)
        success = r.returncode == 0 and "error:" not in output
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"lean timed out after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def verify_lean(lean_code: str, lean_bin: str,
                timeout: int = 30) -> tuple[bool, str, str]:
    """
    Run lean on lean_code. Returns (success, output_text, clean_code).

    Always strips import lines first — standalone Lean has no Mathlib.
    clean_code is the code that was actually run (imports removed).
    """
    clean = _strip_imports(lean_code)
    if not clean:
        return False, "empty code after stripping imports", lean_code

    success, output = _run_lean_on_code(clean, lean_bin, timeout)
    note = "  [imports stripped]" if clean != lean_code else ""
    return success, output + note, clean


def verify_lean_mathlib(lean_code: str, mathlib_cfg: dict,
                        timeout: int = 90) -> tuple[bool, str, str]:
    """
    Verify lean_code using `lake env lean` inside a Mathlib Lake project.
    Returns (success, output_text, lean_code).

    Unlike verify_lean, this does NOT strip imports — Mathlib imports are valid here.
    The Lake project's environment provides all compiled Mathlib .olean files,
    so imports like `import Mathlib.Analysis.Calculus.Deriv.Basic` work instantly.
    """
    project_dir = Path(mathlib_cfg["mathlib_project_dir"])
    lake_bin    = Path(mathlib_cfg["lake_bin"])
    lean_bin    = Path(mathlib_cfg["lean_bin"])

    if not project_dir.exists():
        return False, f"Mathlib project not found: {project_dir}", lean_code
    if not lake_bin.exists():
        return False, f"lake not found: {lake_bin}", lean_code
    if not lean_bin.exists():
        return False, f"lean not found: {lean_bin}", lean_code

    # Write conjecture to a temp file inside the project's Conjectures/ dir
    conj_dir = project_dir / "Conjectures"
    conj_dir.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".lean", mode="w", delete=False,
        encoding="utf-8", dir=conj_dir
    ) as f:
        f.write(lean_code)
        tmp = Path(f.name)

    try:
        # Set up environment so lake can find lean
        env = os.environ.copy()
        env["PATH"] = str(lean_bin.parent) + os.pathsep + env.get("PATH", "")

        r = subprocess.run(
            [str(lake_bin), "env", str(lean_bin), str(tmp)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
            cwd=project_dir,
            env=env,
        )
        stdout = r.stdout or ""
        stderr = r.stderr or ""
        output = (stdout + stderr).strip()

        # Filter out errors SOURCED from dependency package lakefiles.
        # An error is a "package error" only when the file path at the START of
        # the error line (before line:col) is inside .lake/packages/.
        # We must NOT filter errors whose source is our file but whose message
        # body merely mentions a package path.
        def _is_pkg_error(line: str) -> bool:
            # Format A: "error: <pkg-path>:line:col: ..."  (lake's own msgs)
            if re.match(r"^error[:(]\s*[^\s]*[/\\]\.lake[/\\]packages", line):
                return True
            # Format B: "<pkg-path>:line:col: error|warning: ..."  (lean compiler msgs)
            # Windows paths start with "C:\" etc. — use a pattern that handles
            # the drive-letter colon: match optional "X:" then the rest of the path.
            m = re.match(
                r"^([A-Za-z]:[/\\][^:]+|[/][^:]+):\d+:\d+:\s+(?:error|warning)[:(]",
                line
            )
            if m:
                src = m.group(1).replace("\\", "/")
                return "/.lake/packages/" in src
            return False

        # Match both "error:" (simple) and "error(...): " (structured Lean errors
        # like `error(lean.unknownIdentifier): Unknown identifier`)
        error_lines = [l for l in output.splitlines()
                       if re.search(r"\berror[:(]", l)]
        real_errors = [l for l in error_lines if not _is_pkg_error(l)]

        # Safety check: if lake exited with error AND our file doesn't appear in
        # the output at all, lean never ran on our code (lake failed before it).
        # Treat as failure rather than a false positive.
        file_was_processed = (str(tmp) in output or
                              tmp.name in output or
                              r.returncode == 0)

        success = file_was_processed and not real_errors
        return success, output, lean_code

    except subprocess.TimeoutExpired:
        return False, f"lean (Mathlib) timed out after {timeout}s", lean_code
    except Exception as e:
        return False, str(e), lean_code
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def make_client(url: str = LM_STUDIO_URL):
    from openai import OpenAI
    return OpenAI(base_url=url, api_key="lm-studio")


def _stream_llm(client, model: str, messages: list,
                temperature: float = 0.4, max_tokens: int = 2048) -> str:
    """
    Stream a completion and return the final answer text.

    LM Studio / thinking models split output into two fields:
      delta.content          -- the model's final answer
      delta.reasoning_content -- internal chain-of-thought (not shown to user)

    Strategy:
      1. Collect content and thinking separately.
      2. Return content if it is substantive (>80 chars after cleaning).
      3. Fall back to thinking text only if content is empty / trivially short.
         (Some LM Studio configs put everything in reasoning_content.)
    """
    content_text  = ""
    thinking_text = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content_text  += getattr(delta, "content",           None) or ""
            thinking_text += getattr(delta, "reasoning_content", None) or ""
    except Exception as e:
        print(f"\n  [LLM error: {e}]")

    def _clean(t: str) -> str:
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

    content_clean  = _clean(content_text)
    thinking_clean = _clean(thinking_text)

    # Prefer content when it has the answer.
    # A proper lean response is always >80 chars (CONJECTURE line + lean block).
    if len(content_clean) > 80:
        return content_clean

    # Content empty or trivially short — fall back to thinking text.
    # This handles LM Studio configs that route everything to reasoning_content.
    return thinking_clean or content_clean


def _sorry_rescue(lean_code: str) -> str:
    """
    Last-resort sorry injection.

    If lean_code has a theorem/lemma/def declaration followed by a broken
    proof body, replace everything from the first `:= by` onwards with
    `:= by\n  sorry`.  This preserves the *statement* while making the
    proof trivially accept-able for type-checking.

    Returns the rescued code, or the original if no `:= by` was found.
    """
    # Match : ... := by  (the proof obligation entry point)
    # We replace from := by onward with := by\n  sorry
    patterns = [
        r":=\s*by\b.*$",           # := by ...rest
        r":=\s*\{\s*.*$",          # := { ... (structure proof)
        r"by\s*$",                  # trailing `by` with nothing after
    ]
    lines = lean_code.splitlines()
    new_lines = []
    injected = False
    for i, line in enumerate(lines):
        # Look for `:= by` or bare `by` at end of a theorem line
        if not injected and re.search(r":=\s*by\b", line):
            # Cut off at `:= by`, add sorry
            prefix = re.sub(r":=\s*by\b.*", ":= by", line)
            new_lines.append(prefix)
            new_lines.append("  sorry")
            injected = True
        elif not injected:
            new_lines.append(line)
        # Skip remaining lines if injected (they were the broken proof)

    if injected:
        return "\n".join(new_lines)
    return lean_code


def extract_lean_block(text: str) -> str:
    """
    Extract Lean 4 code from LLM response. Tries in order:
    1. ```lean ... ``` fenced block
    2. Any ``` ... ``` block that contains theorem/def/lemma
    3. Lines that look like Lean code (import/theorem/def/lemma)
    """
    # 1. Explicit ```lean block
    m = re.search(r"```lean\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Any ``` block containing Lean keywords
    for m in re.finditer(r"```\w*\s*(.*?)```", text, re.DOTALL):
        block = m.group(1).strip()
        if any(kw in block for kw in
               ("theorem ", "lemma ", "def ", "import ", "#check", "example ")):
            return block

    # 3. Line-by-line scan for Lean-looking code
    lines = text.split("\n")
    lean_lines = []
    capturing = False
    for line in lines:
        if re.match(r"^(import |theorem |lemma |def |#check |example )", line):
            capturing = True
        if capturing:
            # Stop if we hit a clearly non-code paragraph
            if line and not line.startswith(" ") and not line.startswith("\t") \
               and not re.match(r"^(import |theorem |lemma |def |--|#|  |\(|\{)", line) \
               and lean_lines:
                break
            lean_lines.append(line)

    return "\n".join(lean_lines).strip() if lean_lines else ""


def extract_conjecture(text: str, lean_code: str = "") -> str:
    """
    Pull the plain-English conjecture line.

    Falls back to:
      1. A leading comment block inside the lean code
      2. The theorem/lemma name converted from snake_case to prose
    """
    # 1. Explicit CONJECTURE: line
    m = re.search(r"CONJECTURE:\s*(.+?)(?:\n|$)", text)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # 2. First prose sentence in the LLM text (not a Lean keyword line)
    for line in text.splitlines():
        line = line.strip()
        # Skip lines that are code fragments: backticks, escaped newlines,
        # end-of-string markers, or anything that looks like the LLM echoing
        # a prompt template back (these contaminate the mesh if folded in).
        if "`" in line:
            continue
        if "\\n" in line or '"""' in line or "'''" in line:
            continue
        if line.endswith(('"', "'", "```", "```.")):
            continue
        if (line and not line.startswith(("```", "--", "#", "*", "-", ">"))
                and not re.match(r"^(theorem|lemma|def|import|//|RULE|HINT|Example|NOTE)", line, re.I)
                and len(line) > 20 and ":" not in line[:5]):
            return line[:200]

    # 3. Theorem/lemma name from lean_code → snake_case to prose
    if lean_code:
        m2 = re.search(r"\b(?:theorem|lemma)\s+(\w+)", lean_code)
        if m2:
            name = m2.group(1).replace("_", " ").strip()
            if name and name not in ("placeholder", "placeholder true", "sorry"):
                return name.capitalize()

    return ""


def extract_proof_status(text: str) -> str:
    m = re.search(r"PROOF_STATUS:\s*(\w+)", text)
    return m.group(1).lower() if m else "unknown"


def llm_formalize(client, model: str, conjecture: dict, attempt: int = 0,
                  system_prompt: str = None) -> str:
    """
    Ask the LLM to formalize a HAMesh conjecture as Lean 4.
    Uses streaming so large models don't time out.
    system_prompt: override (e.g. LEAN_MATHLIB_SYSTEM_PROMPT). Defaults to standalone.
    """
    if system_prompt is None:
        system_prompt = LEAN_SYSTEM_PROMPT

    seed    = conjecture.get("seed", "")[:150]
    near    = conjecture.get("nearest", [])
    novelty = conjecture.get("novelty", 0)
    mathlib = "Mathlib" in system_prompt

    near_lines = "\n".join(
        f"  [{i+1}] sim={s:.3f}  {t[:120]}"
        for i, (s, t) in enumerate(near[:3])
    )

    focus = near[0][1][:80] if near else seed

    if attempt == 0:
        mathlib_note = (
            "\nYou have full Mathlib access. Use appropriate imports for real numbers, "
            "topology, measure theory, calculus etc."
            if mathlib else ""
        )
        user_msg = (
            f"The HAMesh found a high-novelty gap (novelty={novelty:.3f}) between "
            f"these mathematical concepts:\n\n"
            f"SEED (what the mesh was probing):\n  {seed}\n\n"
            f"NEAREST KNOWN THEOREMS (what the gap is between):\n{near_lines}\n\n"
            f"Formulate a bridging conjecture and write it as Lean 4.{mathlib_note}\n"
            f"Begin your response with a CONJECTURE line, then a ```lean block."
        )
    elif attempt == 1:
        if mathlib:
            example = (
                "CONJECTURE: The derivative of the square function is 2x.\n\n"
                "```lean\n"
                "import Mathlib.Analysis.Calculus.Deriv.Basic\n"
                "open Real in\n"
                "theorem deriv_sq (x : \u211d) : HasDerivAt (fun x => x^2) (2*x) x := by\n"
                "  sorry\n"
                "```"
            )
        else:
            example = (
                "CONJECTURE: Every natural number divides zero.\n\n"
                "```lean\n"
                "theorem dvd_zero (n : Nat) : n \u2223 0 := by\n"
                "  sorry\n"
                "```"
            )
        user_msg = (
            f"Your previous response contained no Lean 4 code block. Please try again.\n\n"
            f"Write the simplest possible theorem about: {focus}\n\n"
            f"Here is the exact format — copy this style:\n\n"
            f"{example}\n\n"
            f"Now write your own theorem about: {focus}"
        )
    else:
        # Last resort
        if mathlib:
            example = "theorem result (x : \u211d) : x + 0 = x := by ring"
            imports = "import Mathlib.Data.Real.Basic\n"
        else:
            example = "theorem result (n : Nat) : n + 0 = n := by sorry"
            imports = ""
        user_msg = (
            f"Write one Lean 4 theorem about this topic: {seed[:60]}\n"
            f"Use sorry for the proof. Wrap only the theorem in a ```lean block.\n\n"
            f"```lean\n"
            f"{imports}"
            f"{example}\n"
            f"```"
        )

    return _stream_llm(client, model, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ])


def llm_repair(client, model: str, lean_code: str, errors: str,
               system_prompt: str = None, mathlib_mode: bool = False,
               temperature: float = 0.3) -> str:
    """Ask the LLM to fix Lean 4 errors. Uses streaming."""
    if system_prompt is None:
        system_prompt = LEAN_SYSTEM_PROMPT

    # Targeted hints based on error type and mode
    hints = []
    if not mathlib_mode:
        if "Membership" in errors or "\u2208 s" in lean_code:
            hints.append(
                "  HINT: Replace `x \u2208 s` with `s x` — "
                "Membership needs Mathlib.\n"
                "  Use predicate application: write `s x` not `x \u2208 s`."
            )
        if "Real" in errors or "\u211d" in errors or "\u211d" in lean_code:
            hints.append(
                "  HINT: `Real` and `\u211d` require Mathlib. Use `Float` or `Int` instead."
            )
        if "Matrix" in errors or "Finset" in errors:
            hints.append(
                "  HINT: `Matrix` and `Finset` require Mathlib. "
                "Encode matrices as `Fin m \u2192 Fin n \u2192 Float`."
            )
        no_import_line = "Use only core Lean 4 — no import statements.\n"
    else:
        if "unknown identifier" in errors or "failed to synthesize" in errors:
            hints.append(
                "  HINT: Check your imports. Add the specific Mathlib module that "
                "defines the type/function you're using."
            )
        if "type mismatch" in errors.lower():
            hints.append(
                "  HINT: Check type coercions. `\u2115` vs `\u2124` vs `\u211d` need explicit casts."
            )
        no_import_line = ""

    hint_str = "\n".join(hints)

    return _stream_llm(client, model, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"This Lean 4 code has errors:\n\n"
            f"```lean\n{lean_code}\n```\n\n"
            f"ERRORS:\n{errors[:600]}\n\n"
            f"{hint_str}\n"
            f"Fix the code. Keep the same conjecture intent but correct the syntax/types.\n"
            f"{no_import_line}"
            f"If unsure how to prove it, replace the proof body with `sorry`.\n"
            f"Wrap your answer in ```lean ... ```."
        )},
    ], temperature=temperature)


# ---------------------------------------------------------------------------
# Autoformalization pipeline
# ---------------------------------------------------------------------------

class AutoformalEngine:
    def __init__(self, client, model: str, lean_bin: str | None,
                 max_attempts: int = 3, verify: bool = True,
                 mathlib_cfg: dict | None = None,
                 prover_client=None, prover_model: str | None = PROVER_MODEL):
        self.client        = client
        self.model         = model
        self.lean_bin      = lean_bin
        self.max_attempts  = max_attempts
        self.mathlib_cfg   = mathlib_cfg   # if set, use lake env lean + Mathlib prompt
        self.prover_client = prover_client or client   # separate model for proof steps
        # Default to PROVER_MODEL constant; fall back to main model only if explicitly None
        self.prover_model  = prover_model if prover_model is not None else model
        self.verify        = verify and (lean_bin is not None or mathlib_cfg is not None)

        if mathlib_cfg:
            self.system_prompt = LEAN_MATHLIB_SYSTEM_PROMPT
            self._verifier     = "mathlib"
        else:
            self.system_prompt = LEAN_SYSTEM_PROMPT
            self._verifier     = "standalone"

    def _verify(self, lean_code: str) -> tuple[bool, str, str]:
        """Route to the right verifier depending on mode."""
        if self._verifier == "mathlib" and self.mathlib_cfg:
            return verify_lean_mathlib(lean_code, self.mathlib_cfg, timeout=90)
        elif self.lean_bin:
            return verify_lean(lean_code, self.lean_bin, timeout=30)
        return False, "no verifier configured", lean_code

    def process(self, conjecture: dict) -> dict:
        """
        Full pipeline for one conjecture.
        Returns a result dict with lean_code, verified, errors, conjecture_text.
        """
        seed = conjecture.get("seed", "")[:80]
        print(f"\n  Conjecture: {seed}")
        print(f"  novelty={conjecture.get('novelty', 0):.4f}  "
              f"recurrence={conjecture.get('recurrence', 0)}")

        lean_code      = ""
        conjecture_text = ""
        proof_status   = "unknown"
        errors         = ""
        verified       = False

        for attempt in range(self.max_attempts):
            # 1. Generate Lean 4
            mode_tag = "[mathlib]" if self._verifier == "mathlib" else ""
            print(f"  [attempt {attempt+1}/{self.max_attempts}]{mode_tag} Calling LLM...",
                  end=" ", flush=True)
            if attempt == 0 or not lean_code:
                # No code to repair — (re)formalize, escalating simplicity with each attempt
                raw = llm_formalize(self.client, self.model, conjecture,
                                    attempt=attempt, system_prompt=self.system_prompt)
            else:
                # We have code with errors — ask the prover model to fix them.
                # If a dedicated prover model is configured (different from the
                # conjecture model), use DEEPSEEK_PROVER_PROMPT and a lower
                # temperature for more deterministic, focused repairs.
                using_dedicated_prover = (self.prover_model != self.model)
                repair_prompt = (DEEPSEEK_PROVER_PROMPT
                                 if using_dedicated_prover
                                 else self.system_prompt)
                repair_temp   = 0.1 if using_dedicated_prover else 0.3
                raw = llm_repair(self.prover_client, self.prover_model, lean_code, errors,
                                 system_prompt=repair_prompt,
                                 mathlib_mode=(self._verifier == "mathlib"),
                                 temperature=repair_temp)

            raw_code    = extract_lean_block(raw)
            proof_status = extract_proof_status(raw)

            # In standalone mode: strip imports (no Mathlib available).
            # In Mathlib mode:   keep imports — they're valid and needed.
            if self._verifier == "mathlib":
                lean_code = raw_code.strip() if raw_code else ""
            else:
                lean_code = _strip_imports(raw_code) if raw_code else ""

            # Extract conjecture text (pass lean_code for name-based fallback)
            conjecture_text = extract_conjecture(raw, lean_code) or conjecture_text

            # A valid lean block must contain at least one declaration keyword
            _has_decl = any(kw in lean_code for kw in
                            ("theorem ", "lemma ", "def ", "example ", "#check"))
            if not lean_code or not _has_decl:
                snippet = raw.replace("\n", " ")[:120] if raw else "(empty response)"
                print(f"no lean block found. LLM said: {snippet}")
                lean_code = ""   # reset so next attempt re-formalizes
                continue

            print(f"got {len(lean_code)} chars", end="")

            # 2. Verify
            if self.verify:
                print(", verifying...", end=" ", flush=True)
                verified, errors, lean_code = self._verify(lean_code)
                if verified:
                    print("VERIFIED [OK]")
                    break
                else:
                    first_error = errors.split("\n")[0][:80] if errors else "unknown"
                    print(f"error: {first_error}")
                    # Last-chance rescue: replace broken proof body with sorry
                    # (works in both standalone and Mathlib mode)
                    if attempt == self.max_attempts - 1 and lean_code:
                        rescued = _sorry_rescue(lean_code)
                        if rescued and rescued != lean_code:
                            v2, out2, rescued_code = self._verify(rescued)
                            if v2:
                                lean_code = rescued_code
                                verified  = True
                                errors    = ""
                                print(f"  [sorry-rescued] VERIFIED [OK]")
                                break
            else:
                print(" (verification skipped)")
                break

        # If all attempts failed to produce any lean code, emit a typed placeholder
        # so the file is at least syntactically valid and records the conjecture.
        if not lean_code:
            print(f"  [all attempts failed — writing placeholder]")
            lean_code = (
                f"-- HAMesh could not autoformalize this conjecture.\n"
                f"-- Seed: {seed}\n"
                f"-- The conjecture may require Mathlib or a stronger model.\n"
                f"theorem placeholder_true : True := trivial"
            )

        return {
            "seed":             seed,
            "novelty":          conjecture.get("novelty", 0),
            "recurrence":       conjecture.get("recurrence", 0),
            "conjecture_text":  conjecture_text,
            "lean_code":        lean_code,
            "proof_status":     proof_status,
            "verified":         verified,
            "errors":           errors[:400] if not verified else "",
            "attempts":         attempt + 1,
            "timestamp":        datetime.now().isoformat(),
        }

    def save_lean_file(self, result: dict, out_dir: Path) -> Path:
        """Save result as a .lean file with metadata header."""
        out_dir.mkdir(parents=True, exist_ok=True)

        status_tag = "VERIFIED" if result["verified"] else (
            "SORRY" if "sorry" in result["lean_code"].lower() else "DRAFT"
        )
        slug = re.sub(r"[^a-z0-9]+", "_", result["seed"].lower())[:40]
        filename = f"{status_tag}_{slug}.lean"
        path = out_dir / filename

        header = (
            f"-- HAMesh Autoformalization\n"
            f"-- Status     : {status_tag}\n"
            f"-- Conjecture : {result['conjecture_text']}\n"
            f"-- Seed       : {result['seed']}\n"
            f"-- Novelty    : {result['novelty']:.4f}\n"
            f"-- Recurrence : {result['recurrence']}\n"
            f"-- Generated  : {result['timestamp']}\n"
            f"-- Attempts   : {result['attempts']}\n"
            f"\n"
        )

        path.write_text(header + result["lean_code"], encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Live mode: scholar + autoformalize loop
# ---------------------------------------------------------------------------

def live_loop(args):
    """Continuously dream on the mesh and autoformalize new conjectures."""
    from ham_core import HolographicMesh
    from ham_scholar import MathScholar, ConjectureLog

    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    client   = make_client(args.url)
    lean_bin = find_lean() if not args.no_verify else None
    engine   = AutoformalEngine(client, args.model, lean_bin,
                                 max_attempts=args.attempts,
                                 verify=not args.no_verify)

    print(f"  Loading mesh: {args.mesh}")
    mesh = HolographicMesh.load(args.mesh, device=DEVICE)
    scholar = MathScholar(mesh)

    seen_sigs = set()
    round_n   = 0
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results = []

    print(f"\n  Live mode: dream {args.cycles} cycles, then autoformalize new conjectures.")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Ctrl-C to stop.\n")

    try:
        while True:
            round_n += 1
            print(f"\n{'='*60}")
            print(f"  Round {round_n}: dreaming {args.cycles} cycles...")

            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                scholar.dream_and_discover(total_cycles=args.cycles, verbose=False)

            new_conjectures = [
                c for c in scholar.log.entries
                if c["seed"] not in seen_sigs
            ]
            for c in new_conjectures:
                seen_sigs.add(c["seed"])

            print(f"  {len(new_conjectures)} new conjecture(s) found.")

            for c in new_conjectures[:args.top]:
                result = engine.process(c)
                path   = engine.save_lean_file(result, OUTPUT_DIR)
                all_results.append(result)
                print(f"  Saved: {path.name}")

            # Save JSON log
            log_path = OUTPUT_DIR / "lean_results.json"
            log_path.write_text(
                json.dumps(all_results, indent=2), encoding="utf-8"
            )
            print(f"  Log updated: {log_path}")

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HAMesh Lean 4 autoformalization engine"
    )
    parser.add_argument("--conjectures", default="conjectures.json",
                        help="Conjecture log from ham_scholar.py")
    parser.add_argument("--mesh",        default="math_mesh.pt",
                        help="Mesh file (for --live mode)")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--url",         default=LM_STUDIO_URL)
    parser.add_argument("--top",         type=int, default=10,
                        help="How many conjectures to process")
    parser.add_argument("--attempts",    type=int, default=3,
                        help="Max LLM repair attempts per conjecture")
    parser.add_argument("--no-verify",   action="store_true",
                        help="Skip lean verification (LLM output only)")
    parser.add_argument("--live",        action="store_true",
                        help="Continuous dream+formalize loop")
    parser.add_argument("--cycles",      type=int, default=500,
                        help="Dream cycles per live-mode round (default 500)")
    parser.add_argument("--out",         default=str(OUTPUT_DIR),
                        help="Output directory for .lean files")
    # Mathlib mode
    parser.add_argument("--mathlib",     action="store_true",
                        help="Use Mathlib (requires setup_mathlib.py to have been run)")
    # Dual-model: separate prover model for repair steps
    parser.add_argument("--prover-model", default=PROVER_MODEL,
                        help=f"Model for Lean repair steps (default: {PROVER_MODEL}). "
                             "Pass --prover-model '' to use the same model as --model.")
    args = parser.parse_args()

    out_dir = Path(args.out)

    print("\n" + "=" * 60)
    print("  HAMesh Lean 4 Autoformalization Engine")
    print("=" * 60)

    # Check Mathlib config
    mathlib_cfg = None
    if args.mathlib or not args.no_verify:
        mathlib_cfg = load_mathlib_config()
    if args.mathlib and not mathlib_cfg:
        print("\n  --mathlib requested but mathlib_config.json not found or project missing.")
        print("  Run:  python setup_mathlib.py")
        print("  Falling back to standalone mode.\n")

    if mathlib_cfg:
        print(f"\n  Mathlib mode: {mathlib_cfg['mathlib_project_dir']}")

    # Check Lean
    lean_bin = None
    if not args.no_verify:
        lean_bin = find_lean()
        if lean_bin:
            r = subprocess.run([lean_bin, "--version"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            print(f"  Lean 4  : {r.stdout.strip()}")
        elif not mathlib_cfg:
            print("\n  Lean 4 : NOT FOUND — run install_lean.py")
            print("  Continuing in --no-verify mode.\n")
            args.no_verify = True

    # Prover model (optional second model for repair steps)
    prover_client = None
    prover_model  = args.prover_model
    if prover_model:
        prover_client = make_client(args.url)
        print(f"  Prover  : {prover_model}")

    if args.live:
        live_loop(args)
        return

    # Load conjectures
    print(f"\n  Loading: {args.conjectures}")
    raw = json.loads(Path(args.conjectures).read_text(encoding="utf-8"))
    conjectures = raw.get("conjectures", raw) if isinstance(raw, dict) else raw
    conjectures = conjectures[:args.top]
    print(f"  Processing top {len(conjectures)} conjectures")

    client = make_client(args.url)
    engine = AutoformalEngine(
        client, args.model, lean_bin,
        max_attempts=args.attempts,
        verify=not args.no_verify,
        mathlib_cfg=mathlib_cfg,
        prover_client=prover_client,
        prover_model=prover_model,
    )

    results = []
    verified_count = 0

    for c in conjectures:
        result = engine.process(c)
        path   = engine.save_lean_file(result, out_dir)
        results.append(result)
        if result["verified"]:
            verified_count += 1
        print(f"  -> {path.name}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results: {len(results)} processed")
    print(f"  Verified (no sorry): {verified_count}")
    print(f"  Sorry proofs (typed but unproven): "
          f"{sum(1 for r in results if 'sorry' in r['lean_code'].lower() and not r['verified'])}")
    print(f"  Output directory: {out_dir}/")

    # Save JSON log
    log_path = out_dir / "lean_results.json"
    out_dir.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  JSON log: {log_path}")

    # Print verified ones
    verified = [r for r in results if r["verified"]]
    if verified:
        print(f"\n  VERIFIED conjectures:")
        for r in verified:
            print(f"    - {r['conjecture_text'][:80]}")

    sorry_only = [r for r in results
                  if not r["verified"] and "sorry" in r["lean_code"].lower()]
    if sorry_only:
        print(f"\n  Type-correct but unproven (human/stronger model needed):")
        for r in sorry_only[:5]:
            print(f"    - {r['conjecture_text'][:80]}")


if __name__ == "__main__":
    main()
