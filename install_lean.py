"""
Install Lean 4 on Windows without elan.

Downloads the prebuilt lean4 Windows binary directly from GitHub releases.
No elan, no SSL certificate revocation check, no PowerShell tricks needed.

Usage:
    python install_lean.py                  # install to C:\lean4
    python install_lean.py --dest D:\lean4  # install to a custom path
    python install_lean.py --check          # just check if lean is already installed
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

GITHUB_API  = "https://api.github.com/repos/leanprover/lean4/releases/latest"
GITHUB_REPO = "https://github.com/leanprover/lean4/releases"


def find_existing_lean() -> str | None:
    """
    Check if lean is already installed.  Returns path or None.

    Searches direct installs first to avoid the elan SSL shim on Windows.
    """
    def _try(path: str) -> str | None:
        try:
            r = subprocess.run([path, "--version"],
                               capture_output=True, text=True, timeout=5)
            return path if r.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None

    # 1. Direct versioned installs (most reliable — not an elan shim)
    for base in [Path("C:/lean4"), Path("D:/lean4")]:
        if base.exists():
            for lean_exe in sorted(base.rglob("lean.exe"), reverse=True):
                if result := _try(str(lean_exe)):
                    return result

    # 2. PATH lookup (elan shim may be here and can hang)
    for cmd in ["lean", "lean4"]:
        if result := _try(cmd):
            return result

    # 3. Elan install last (may hang on SSL check)
    home = Path.home()
    for p in [
        home / ".elan" / "bin" / "lean.exe",
        home / ".elan" / "bin" / "lean",
    ]:
        if p.exists():
            if result := _try(str(p)):
                return result

    return None


def get_latest_release() -> tuple[str, str]:
    """
    Query GitHub API for the latest lean4 release.
    Returns (version_tag, windows_zip_url).
    """
    print("  Querying GitHub for latest Lean 4 release...")
    req = urllib.request.Request(
        GITHUB_API,
        headers={"User-Agent": "HAMesh-lean-installer/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    tag = data["tag_name"]   # e.g. "v4.14.0"

    # Find the Windows zip asset
    windows_asset = None
    for asset in data.get("assets", []):
        name = asset["name"].lower()
        if "windows" in name and name.endswith(".zip"):
            windows_asset = asset["browser_download_url"]
            break

    if not windows_asset:
        # Fallback: construct URL from tag
        version = tag.lstrip("v")
        windows_asset = (
            f"https://github.com/leanprover/lean4/releases/download/{tag}/"
            f"lean4-{tag}-windows.zip"
        )
        print(f"  (No asset found in API response — trying constructed URL)")

    return tag, windows_asset


def download_with_progress(url: str, dest: Path):
    """Download a file with a simple progress indicator."""
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")

    def _reporthook(count, block_size, total):
        if total > 0:
            pct = min(count * block_size * 100 // total, 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()   # newline after progress bar


def install_lean(dest_dir: Path):
    """Download and extract lean4 to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing install
    existing = find_existing_lean()
    if existing:
        try:
            r = subprocess.run([existing, "--version"],
                               capture_output=True, text=True)
            print(f"\n  Lean 4 already installed: {r.stdout.strip()}")
            print(f"  Path: {existing}")
            return existing
        except Exception:
            pass

    # Get latest release info
    try:
        tag, zip_url = get_latest_release()
        print(f"  Latest release: {tag}")
    except Exception as e:
        print(f"  Could not reach GitHub API: {e}")
        print(f"\n  Manual install:")
        print(f"  1. Visit: {GITHUB_REPO}")
        print(f"  2. Download the Windows .zip for the latest release")
        print(f"  3. Extract to {dest_dir}")
        print(f"  4. Add {dest_dir / 'bin'} to your PATH")
        return None

    # Download zip
    zip_path = dest_dir / f"lean4-{tag}-windows.zip"
    try:
        download_with_progress(zip_url, zip_path)
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print(f"\n  Try manually:")
        print(f"  1. Visit: {GITHUB_REPO}/tag/{tag}")
        print(f"  2. Download the Windows .zip")
        print(f"  3. Extract to: {dest_dir}")
        return None

    # Extract
    print(f"  Extracting to {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()

    # Find lean.exe in extracted content
    lean_exe = None
    for p in dest_dir.rglob("lean.exe"):
        lean_exe = p
        break
    if not lean_exe:
        for p in dest_dir.rglob("lean"):
            if p.is_file():
                lean_exe = p
                break

    if not lean_exe:
        print(f"  Extraction done but lean.exe not found in {dest_dir}")
        print(f"  Contents: {list(dest_dir.iterdir())[:10]}")
        return None

    lean_bin_dir = lean_exe.parent
    print(f"  lean found at: {lean_exe}")

    # Verify it runs
    try:
        r = subprocess.run([str(lean_exe), "--version"],
                           capture_output=True, text=True, timeout=10)
        print(f"  {r.stdout.strip() or r.stderr.strip()}")
        ok = r.returncode == 0
    except Exception as e:
        print(f"  Could not run lean: {e}")
        ok = False

    # Add to PATH for this session
    current_path = os.environ.get("PATH", "")
    if str(lean_bin_dir) not in current_path:
        os.environ["PATH"] = str(lean_bin_dir) + os.pathsep + current_path
        print(f"  Added to PATH for this session: {lean_bin_dir}")

    print(f"\n  To make this permanent, add to your system PATH:")
    print(f"    {lean_bin_dir}")
    print(f"\n  In PowerShell (run as Administrator):")
    print(f'    [Environment]::SetEnvironmentVariable("PATH",')
    print(f'      $env:PATH + ";{lean_bin_dir}",')
    print(f'      "Machine")')

    return str(lean_exe) if ok else None


def main():
    parser = argparse.ArgumentParser(
        description="Install Lean 4 on Windows (no elan required)"
    )
    parser.add_argument("--dest",  default="C:\\lean4",
                        help="Install directory (default: C:\\lean4)")
    parser.add_argument("--check", action="store_true",
                        help="Check if Lean is already installed and exit")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  Lean 4 Installer (direct GitHub download)")
    print("="*50)

    if args.check:
        lean = find_existing_lean()
        if lean:
            try:
                r = subprocess.run([lean, "--version"],
                                   capture_output=True, text=True)
                print(f"\n  Found: {lean}")
                print(f"  {r.stdout.strip()}")
            except Exception:
                print(f"\n  Found at: {lean}")
        else:
            print("\n  Lean 4 not found.")
            print(f"  Run without --check to install.")
        return

    lean_path = install_lean(Path(args.dest))

    if lean_path:
        print(f"\n  Installation complete.")
        print(f"\n  Test it:")
        print(f'    "{lean_path}" --version')
        print(f"\n  Run ham_lean.py with verification:")
        print(f"    python ham_lean.py --conjectures lab_lean\\lean_results.json "
              f"--top 5")
        print(f"\n  ham_lean will auto-detect lean at: {lean_path}")
    else:
        print(f"\n  Installation failed — see errors above.")
        print(f"  Run with --no-verify in ham_lean.py/ham_lab.py to skip lean.")


if __name__ == "__main__":
    main()
