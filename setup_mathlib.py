"""
Setup a Mathlib Lake project for HAMesh verification.

This creates a minimal Lake project at D:/lean4/hamesh_mathlib that:
  - Depends on Mathlib4
  - Downloads prebuilt .olean caches (avoids the 2-hour full compile)
  - Exposes a verify function that HAMesh can call

Usage:
    python setup_mathlib.py               # create project + download cache
    python setup_mathlib.py --test        # test the setup with a sample theorem
    python setup_mathlib.py --path D:/lean4/hamesh_mathlib  # custom location

After setup, ham_lean.py will auto-detect the project and enable Mathlib mode.
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Where we install the Mathlib project
DEFAULT_PROJECT_DIR = Path("D:/lean4/hamesh_mathlib")

# Lean 4 toolchain — must match Mathlib's expected version
# We let elan/lake manage this via the lean-toolchain file
LEAN_TOOLCHAIN = "leanprover/lean4:v4.29.0"

# Which version of Mathlib to pin (latest stable compatible with Lean 4.29.x)
MATHLIB_TAG = "v4.29.0"

# Both lake and lean must come from the same install that matches MATHLIB_TAG.
# The .olean cache is keyed by lean version; mixing versions causes
# 'failed to read file' errors because lake sets LEAN_SYSROOT to its own stdlib.
# Mathlib v4.29.0 requires leanprover/lean4:v4.29.0
LAKE_BIN = Path("C:/lean4/lean-4.29.0-windows/bin/lake.exe")
LEAN_BIN = Path("C:/lean4/lean-4.29.0-windows/bin/lean.exe")


def find_lake() -> Path | None:
    """Find lake.exe."""
    if LAKE_BIN.exists():
        return LAKE_BIN
    for base in [Path("C:/lean4"), Path("D:/lean4")]:
        if base.exists():
            for p in sorted(base.rglob("lake.exe"), reverse=True):
                return p
    # Try PATH
    which = shutil.which("lake")
    return Path(which) if which else None


def find_lean() -> Path | None:
    """Find lean.exe (the direct binary, not elan shim)."""
    if LEAN_BIN.exists():
        return LEAN_BIN
    for base in [Path("C:/lean4"), Path("D:/lean4")]:
        if base.exists():
            for p in sorted(base.rglob("lean.exe"), reverse=True):
                return p
    return None


LAKEFILE = """\
import Lake
open Lake DSL

package hamesh_mathlib where
  -- Increase heartbeat limit for complex conjectures
  moreServerOptions := #[\u27e8`maxHeartbeats, .ofNat 400000\u27e9]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.29.0"

lean_lib HameshMathlib where
  -- All .lean files under HameshMathlib/ are part of this library
"""

LEAN_TOOLCHAIN_CONTENT = "leanprover/lean4:v4.29.0\n"

GITIGNORE = """\
/.lake
/build
*.olean
*.ilean
"""

# A small test theorem that uses real Mathlib content
TEST_THEOREM = """\
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Topology.MetricSpace.Basic

-- Test: sqrt is nonneg (uses Mathlib.Analysis)
theorem sqrt_nonneg_mathlib (x : \u211d) : 0 \u2264 Real.sqrt x := Real.sqrt_nonneg x
"""

TEST_THEOREM_SIMPLE = """\
import Mathlib.Data.Nat.Defs
import Mathlib.Data.Int.Basic

-- Test: basic Mathlib access
theorem nat_pos_of_ne_zero (n : \u2115) (h : n \u2260 0) : 0 < n := Nat.pos_of_ne_zero h
"""


def run(cmd: list, cwd: Path = None, timeout: int = 600, env: dict = None,
        warn_ok: bool = False) -> tuple[bool, str]:
    """
    Run a command, return (success, output).
    warn_ok=True: treat exit code != 0 as success if output only has warnings (no errors).
    """
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        out = (r.stdout + r.stderr).strip()
        if r.returncode == 0:
            return True, out
        # Lake often returns non-zero for deprecation warnings — treat as success
        # if there are no actual "error:" lines (only warnings/info)
        if warn_ok:
            lines = out.splitlines()
            has_error = any(
                l.strip().startswith("error:") or ": error:" in l
                for l in lines
            )
            if not has_error:
                return True, out
        return False, out
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def create_project(project_dir: Path, lake: Path) -> bool:
    """Create the Lake project structure."""
    print(f"\n[1/4] Creating project at {project_dir}")
    project_dir.mkdir(parents=True, exist_ok=True)

    # Write lakefile.lean
    lakefile = project_dir / "lakefile.lean"
    lakefile.write_text(LAKEFILE, encoding="utf-8")
    print(f"  Wrote {lakefile}")

    # Write lean-toolchain
    toolchain = project_dir / "lean-toolchain"
    toolchain.write_text(LEAN_TOOLCHAIN_CONTENT, encoding="utf-8")
    print(f"  Wrote {toolchain}")

    # Write .gitignore
    (project_dir / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    # Create source directory for conjecture files
    src_dir = project_dir / "Conjectures"
    src_dir.mkdir(exist_ok=True)
    (src_dir / ".gitkeep").touch()

    # Create a placeholder HameshMathlib.lean
    placeholder = project_dir / "HameshMathlib.lean"
    placeholder.write_text(
        "-- HAMesh Mathlib bridge\n-- Auto-generated conjecture files go in Conjectures/\n",
        encoding="utf-8",
    )

    print("  Project structure created.")
    return True


def update_packages(project_dir: Path, lake: Path) -> bool:
    """Run lake update to fetch Mathlib."""
    print("\n[2/4] Fetching Mathlib (lake update)...")
    print("  (Clones Mathlib4 + dependencies — 1-5 min on first run)")

    env = os.environ.copy()
    lean_bin_dir = str(LEAN_BIN.parent) if LEAN_BIN.exists() else ""
    if lean_bin_dir:
        env["PATH"] = lean_bin_dir + os.pathsep + env.get("PATH", "")

    # warn_ok=True: lake returns non-zero for deprecation warnings, treat as success
    ok, out = run([str(lake), "update"], cwd=project_dir,
                  timeout=300, env=env, warn_ok=True)
    if not ok:
        print(f"  lake update error:\n{out[:600]}")
        return False

    # Check that packages were actually cloned
    pkgs_dir = project_dir / ".lake" / "packages"
    if pkgs_dir.exists() and any(pkgs_dir.iterdir()):
        names = [p.name for p in pkgs_dir.iterdir()]
        print(f"  Packages fetched: {', '.join(names)}")
        return True

    print(f"  lake update output:\n{out[:400]}")
    return True  # continue anyway — cache step will fail if truly broken


def get_mathlib_cache(project_dir: Path, lake: Path) -> bool:
    """
    Download prebuilt .olean files for Mathlib.
    This avoids a 2-hour compile from source.
    """
    print("\n[3/4] Downloading Mathlib prebuilt cache (~1.5 GB)...")
    print("  This is the big download — allow 5-15 minutes depending on connection.")
    print("  Without this, `lake build` would take 2+ hours from source.\n")

    env = os.environ.copy()
    lean_bin_dir = str(LEAN_BIN.parent) if LEAN_BIN.exists() else ""
    if lean_bin_dir:
        env["PATH"] = lean_bin_dir + os.pathsep + env.get("PATH", "")

    # lake exe cache get -- downloads prebuilt oleans from the Mathlib cache server
    ok, out = run(
        [str(lake), "exe", "cache", "get"],
        cwd=project_dir,
        timeout=1200,  # 20 minutes
        env=env,
    )
    if not ok:
        print(f"  Cache download output:\n{out[:800]}")
        print("\n  If cache download failed, try manually:")
        print(f'    cd "{project_dir}"')
        print(f'    "{lake}" exe cache get')
        print("\n  Alternatively, HAMesh can still use Mathlib but individual")
        print("  files will take 30-60s to compile on first use.")
        # Non-fatal — Mathlib still works, just slower
        return False
    print(f"  Cache download complete.")
    return True


def build_project(project_dir: Path, lake: Path) -> bool:
    """Run lake build to verify the setup works."""
    print("\n[4/4] Verifying build (lake build HameshMathlib)...")

    env = os.environ.copy()
    lean_bin_dir = str(LEAN_BIN.parent) if LEAN_BIN.exists() else ""
    if lean_bin_dir:
        env["PATH"] = lean_bin_dir + os.pathsep + env.get("PATH", "")

    ok, out = run(
        [str(lake), "build", "HameshMathlib"],
        cwd=project_dir,
        timeout=600,
        env=env,
    )
    if not ok:
        print(f"  Build output:\n{out[:500]}")
        return False
    print("  Build successful.")
    return True


def test_conjecture(project_dir: Path, lake: Path) -> bool:
    """Test that we can verify a Mathlib-dependent theorem.

    Uses LEAN_BIN (the version that matches MATHLIB_TAG) so that the
    prebuilt .olean cache is compatible.  Mismatched versions produce
    'failed to read object file' errors even for trivially valid theorems.
    """
    print("\n[Test] Verifying a sample Mathlib theorem...")

    lean = LEAN_BIN
    if not lean.exists():
        lean = Path(find_lean() or "")
    if not lean.exists():
        print(f"  lean.exe not found (expected: {LEAN_BIN})")
        return False

    env = os.environ.copy()
    env["PATH"] = str(lean.parent) + os.pathsep + env.get("PATH", "")

    with tempfile.NamedTemporaryFile(
        suffix=".lean", mode="w", delete=False,
        encoding="utf-8", dir=project_dir / "Conjectures"
    ) as f:
        f.write(TEST_THEOREM_SIMPLE)
        tmp = Path(f.name)

    try:
        r_ok, out = run(
            [str(lake), "env", str(lean), str(tmp)],
            cwd=project_dir,
            timeout=120,
            env=env,
        )
        # Real success: returncode=0 AND no "error:" lines in our file.
        # Do NOT use `"warning" in out` as a proxy — deprecation warnings
        # appear even when lean never ran on our file.
        our_file = tmp.name
        error_lines = [l for l in out.splitlines()
                       if "error:" in l or " error(" in l]
        our_errors  = [l for l in error_lines if our_file in l or ".lake/packages" not in l]
        success = r_ok and not our_errors

        if success:
            print(f"  SUCCESS: Mathlib theorem verified.")
            print(f"  Output: {out[:200] or '(clean)'}")
            return True
        else:
            print(f"  FAILED (our errors: {len(our_errors)}, returncode={0 if r_ok else 1})")
            for l in (our_errors or error_lines)[:5]:
                print(f"    {l[:120]}")
            return False
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def write_hamesh_config(project_dir: Path, lake: Path, lean: Path):
    """Write mathlib_config.json so ham_lean.py can find the project.

    IMPORTANT: lean_bin and lake_bin must be from the SAME install that
    matches MATHLIB_TAG.  Mismatched versions cause 'failed to read object
    file' errors because:
      - lake sets LEAN_SYSROOT to its own stdlib
      - lean refuses .olean files built by a different version

    We always use the LAKE_BIN / LEAN_BIN constants (which are pinned to
    the matching version) rather than the passed-in `lake`/`lean` args
    (which may come from find_lake/find_lean and return the newest binary).
    """
    import json
    # Always use the version-pinned constants, not the auto-detected binaries
    mathlib_lake = LAKE_BIN if LAKE_BIN.exists() else lake
    mathlib_lean = LEAN_BIN if LEAN_BIN.exists() else lean
    config = {
        "mathlib_project_dir": str(project_dir).replace("\\", "/"),
        "lake_bin": str(mathlib_lake).replace("\\", "/"),
        "lean_bin": str(mathlib_lean).replace("\\", "/"),
        # lean_bin_standalone: reference to the standalone (newest) lean
        "lean_bin_standalone": str(lean).replace("\\", "/"),
        "enabled": True,
    }
    config_path = Path(__file__).parent / "mathlib_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n  HAMesh config written to: {config_path}")
    print(f"  lean_bin (Mathlib)      : {mathlib_lean}")
    print(f"  NOTE: lean_bin must match Mathlib version ({MATHLIB_TAG}).")
    print(f"        Prebuilt .olean cache is version-specific.")


def main():
    parser = argparse.ArgumentParser(description="Setup Mathlib for HAMesh")
    parser.add_argument("--path", default=str(DEFAULT_PROJECT_DIR),
                        help=f"Project directory (default: {DEFAULT_PROJECT_DIR})")
    parser.add_argument("--test", action="store_true",
                        help="Only run the test theorem, skip setup")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip the large cache download (Mathlib will be slow on first use)")
    args = parser.parse_args()

    project_dir = Path(args.path)

    print("\n" + "="*60)
    print("  HAMesh Mathlib Setup")
    print("="*60)

    lake = find_lake()
    lean = find_lean()

    if not lake:
        print("\n  ERROR: lake.exe not found.")
        print("  It should be at: C:/lean4/lean-4.29.1-windows/bin/lake.exe")
        print("  Run install_lean.py first.")
        sys.exit(1)

    if not lean:
        print("\n  ERROR: lean.exe not found.")
        sys.exit(1)

    print(f"  lake: {lake}")
    print(f"  lean: {lean}")

    if args.test:
        if not project_dir.exists():
            print(f"\n  Project not found at {project_dir}")
            print("  Run setup_mathlib.py without --test first.")
            sys.exit(1)
        test_conjecture(project_dir, lake)
        return

    # Full setup
    create_project(project_dir, lake)
    update_packages(project_dir, lake)
    if not args.skip_cache:
        get_mathlib_cache(project_dir, lake)
    build_project(project_dir, lake)
    test_conjecture(project_dir, lake)
    write_hamesh_config(project_dir, lake, lean)

    print("\n" + "="*60)
    print("  Mathlib setup complete!")
    print("="*60)
    print(f"\n  Project location : {project_dir}")
    print(f"  HAMesh config    : D:/AI_tools/HAMesh/mathlib_config.json")
    print(f"\n  To use Mathlib in ham_lean.py:")
    print(f"    python ham_lean.py --conjectures conjectures.json --mathlib")
    print(f"\n  To use Mathlib in ham_lab.py:")
    print(f"    python ham_lab.py --resume --mathlib")
    print(f"\n  Now the LLM can write:")
    print(f"    import Mathlib.Analysis.Calculus.Deriv.Basic")
    print(f"    import Mathlib.Topology.MetricSpace.Basic")
    print(f"    import Mathlib.MeasureTheory.Measure.MeasureSpace")
    print(f"    ...and Lean will actually check them.")


if __name__ == "__main__":
    main()
