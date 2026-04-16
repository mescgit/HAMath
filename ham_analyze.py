"""
HAM Analyzer -- Read experiment JSON logs and report results clearly

Usage:
  python ham_analyze.py                    # summarize all experiments in ham_logs/
  python ham_analyze.py --claim C4         # show latest C4 result in detail
  python ham_analyze.py --file ham_logs/experiment_C4_20260413_202200.json
  python ham_analyze.py --sessions         # show all logged interactive sessions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path("./ham_logs")

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def latest_for_claim(claim: str) -> Path | None:
    pattern = f"experiment_{claim}_*.json"
    files = sorted(LOGS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Claim reporters
# ---------------------------------------------------------------------------

def report_c1(data: dict):
    print("\n  +======================================================+")
    print("  |  C1 -- Attractor Convergence Determinism              |")
    print("  +======================================================+")
    print(f"  Hypothesis: {data['hypothesis']}")
    print(f"  Dream cycles per run: {data['dream_cycles_per_run']}")
    print(f"  Independent runs: {data['n_runs']}")

    for mesh_name, m in data["meshes"].items():
        print(f"\n  [{mesh_name}]")
        for run in m["runs"]:
            top = [t for t, _ in run["top_5_attractors"][:3]]
            print(f"    Run {run['run']}: coherence={run['coherence_score']:.2f}  "
                  f"top={top[0][:45]!r}")
        avg_ov  = m["avg_pairwise_overlap"]
        avg_coh = m["avg_coherence"]
        verdict = m["verdict"]
        print(f"    Avg pairwise overlap : {avg_ov:.3f}  ({'perfect' if avg_ov == 1.0 else 'partial'})")
        print(f"    Avg coherence score  : {avg_coh:.3f}")
        print(f"    Verdict              : {verdict}")

    print(f"\n  Overall: {data['conclusion']}")

    # Interpretation
    if all(m["avg_pairwise_overlap"] == 1.0 for m in data["meshes"].values()):
        print("\n  [ok] Perfect overlap (1.000) across all run pairs.")
        print("  The attractor basins are fully deterministic given the mesh state.")
        print("  This is a strong result for the paper -- same mesh always converges")
        print("  to the same concepts regardless of random seed order.")


def report_c2(data: dict):
    print("\n  +======================================================+")
    print("  |  C2 -- Semantic Coherence of Attractor Clusters       |")
    print("  +======================================================+")
    print(f"  Hypothesis: {data['hypothesis']}")

    for mesh_name, m in data["meshes"].items():
        print(f"\n  [{mesh_name}]")
        print(f"    Attractor coherence  : {m['attractor_coherence']:.2f}")
        print(f"    Random baseline avg  : {m['random_baseline_avg']:.2f}")
        print(f"    Delta                : {m['delta']:+.2f}")
        print(f"    Verdict              : {m['verdict']}")
        top = [t for t, _ in m["top_5_attractors"][:5]]
        print(f"    Top attractors       : {', '.join(t[:30] for t in top)}")

    print(f"\n  Overall: {data['conclusion']}")


def report_c3(data: dict):
    print("\n  +======================================================+")
    print("  |  C3 -- Cross-Pollination Creates Cross-Domain Curiosity|")
    print("  +======================================================+")
    print(f"  Hypothesis: {data['hypothesis']}")

    for mesh_name in data.get("after", {}):
        after = data["after"][mesh_name]
        before = data["before"].get(mesh_name, {})
        before_gaps = before.get("gaps", [])
        pre_xd = before.get("pre_existing_cross_domain_isolated", [])
        print(f"\n  [{mesh_name}]")
        print(f"    Isolated gaps before         : {len(before_gaps)}")
        print(f"    Pre-existing cross-domain    : {len(pre_xd)}")
        xd_total    = after.get("xd_total", 0)
        xd_iso      = after.get("xd_isolated", [])
        xd_con      = after.get("xd_connected", [])
        xd_iso_frac = after.get("xd_isolated_frac", 0)
        print(f"    Cross-domain memories added  : {xd_total}")
        print(f"    -- isolated (curiosity gaps) : {len(xd_iso)} ({xd_iso_frac:.0%})")
        print(f"    -- integrated (bridges)      : {len(xd_con)}")
        for s, t in xd_iso[:3]:
            print(f"      [iso] {s:.3f}  {str(t)[:65]}")
        for s, t in xd_con[:2]:
            print(f"      [int] {s:.3f}  {str(t)[:65]}")

    note = data.get("note", "")
    if note:
        print(f"\n  Note: {note}")
    print(f"\n  Overall: {data['conclusion']}")


def report_c4(data: dict):
    print("\n  +======================================================+")
    print("  |  C4 -- Curiosity Synthesis Is Genuinely Cross-Domain  |")
    print("  +======================================================+")
    print(f"  Hypothesis: {data['hypothesis']}")
    print(f"  Gaps tested: {data['n_gaps_tested']}")

    insights = data.get("insights", [])
    total = len(insights)

    if total == 0:
        print("\n  [!] No insights recorded -- experiment may have failed silently.")
        print("  Check that Bonsai server is running and responding.")
        print("  Try: curl http://localhost:22334/v1/models")
        return

    cross = data["cross_domain_count"]
    frac  = data["cross_domain_fraction"]

    print(f"\n  Insights generated : {total}")
    print(f"  Cross-domain       : {cross}/{total}  ({frac:.1%})")
    print(f"  Verdict            : {data['conclusion']}")

    print("\n  Sample insights:")
    shown = 0
    for ins in insights:
        if ins["cross_domain"] and shown < 3:
            print(f"\n    Gap: {ins['gap'][:60]}")
            print(f"    Q:   {ins['question'][:80]}")
            meshes = ', '.join(ins['meshes_activated'])
            print(f"    Meshes activated: [{meshes}]")
            print(f"    A:   {ins['answer_excerpt'][:180]}...")
            shown += 1

    if shown == 0:
        print("\n  [!] No cross-domain insights found.")
        print("  All insights activated only one mesh domain.")
        print("  Possible causes:")
        print("    - Meshes need more cross-pollination (run 'dream 1000' then 'cross')")
        print("    - CONFIDENCE_FLOOR too high (currently 0.15)")
        print("    - Mesh energy too low -- try more distillation")

    # Integrity check
    print(f"\n  Integrity:")
    gaps_with_q = sum(1 for i in insights if len(i.get("question","")) > 10)
    gaps_with_a = sum(1 for i in insights if len(i.get("answer_excerpt","")) > 20)
    print(f"    Insights with real questions : {gaps_with_q}/{total}")
    print(f"    Insights with real answers   : {gaps_with_a}/{total}")
    if gaps_with_q < total or gaps_with_a < total:
        print("    [!] Some insights appear empty -- possible silent API errors")


def report_c5(data: dict):
    print("\n  +======================================================+")
    print("  |  C5 -- Multi-Hop Follows Transitive Chains            |")
    print("  +======================================================+")
    print(f"  Hypothesis: {data['hypothesis']}")
    print(f"  Queries tested: {data['queries_tested']}")

    for mesh_name, m in data["meshes"].items():
        print(f"\n  [{mesh_name}]")
        diff_frac = m["different_fraction"]
        print(f"    1-hop vs 2-hop differ : {diff_frac:.1%} of queries")
        print(f"    Avg 1-hop similarity  : {m['avg_1hop_sim']:.4f}")
        print(f"    Avg 2-hop similarity  : {m['avg_2hop_sim']:.4f}")
        print(f"    Verdict               : {m['verdict']}")

        print(f"\n    Sample comparisons:")
        different_queries = [q for q in m["query_results"] if q["different"]]
        same_queries      = [q for q in m["query_results"] if not q["different"]]
        for q in different_queries[:2]:
            print(f"      Q: {q['question'][:55]}")
            print(f"         1-hop -> {q['1hop_top'][:45]!r}")
            print(f"         2-hop -> {q['2hop_top'][:45]!r}")
            print()

        if not different_queries:
            print("      [!] No queries produced different 1-hop vs 2-hop results.")
            print("      The mesh may have collapsed to few dominant attractors.")
            print("      Try running 'dream 500' then retry C5.")

    print(f"\n  Overall: {data['conclusion']}")

    # Interpretation
    for mesh_name, m in data["meshes"].items():
        delta = m["avg_1hop_sim"] - m["avg_2hop_sim"]
        if abs(delta) < 0.001:
            print(f"\n  [!] [{mesh_name}] 1-hop and 2-hop similarities are identical ({delta:+.4f}).")
            print("  This suggests the mesh may be fully collapsed to a single attractor.")
            print("  The diffraction is cycling back to the same point regardless of hops.")


# ---------------------------------------------------------------------------
# Session log reporter
# ---------------------------------------------------------------------------

def report_session(path: Path):
    records = load_jsonl(path)
    if not records:
        print(f"  Empty log: {path.name}")
        return

    event_counts = {}
    for r in records:
        t = r.get("type", "unknown")
        event_counts[t] = event_counts.get(t, 0) + 1

    print(f"\n  Session: {path.name}")
    print(f"  Events : {len(records)}")
    for etype, count in sorted(event_counts.items()):
        print(f"    {etype:<25} {count}")

    # Attractor snapshots
    snapshots = [r for r in records if r["type"] == "attractor_snapshot"]
    if snapshots:
        print(f"\n  Attractor snapshots ({len(snapshots)}):")
        for s in snapshots[-3:]:  # last 3
            mesh  = s["mesh"]
            top   = [a["text"][:40] for a in s["attractors"][:3]]
            print(f"    [{mesh}] {s['cycles']} cycles | energy delta{s['energy_delta']:+.0f}")
            for t in top:
                print(f"      * {t}")

    # Curiosity insights
    insights = [r for r in records if r["type"] == "curiosity_insight"]
    cross    = [i for i in insights if i.get("cross_domain")]
    if insights:
        print(f"\n  Curiosity insights: {len(insights)} total, {len(cross)} cross-domain")
        for i in cross[:2]:
            print(f"    [{', '.join(i['meshes_activated'])}] {i['question'][:70]}")
            print(f"    -> {i['answer'][:100]}...")

    # Queries
    queries = [r for r in records if r["type"] == "query"]
    if queries:
        cross_q = [q for q in queries if q.get("cross_domain")]
        print(f"\n  Queries logged: {len(queries)}, {len(cross_q)} cross-domain")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate_c5_phase():
    """
    Reads ALL C5_PHASE result files and computes mean +/- std dev
    across runs for each dream stage. This is what goes in the paper.
    """
    files = sorted(LOGS_DIR.glob("experiment_C5_PHASE_*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        print("  No C5_PHASE results found.")
        return

    # Collect per-stage diversity values across runs
    from collections import defaultdict
    import statistics

    stage_data = defaultdict(list)   # stage -> [diversity_fraction, ...]
    stage_energy = defaultdict(list) # stage -> [energy, ...]

    for f in files:
        try:
            data = load_json(f)
            for stage_str, s in data.get("stages", {}).items():
                stage = int(stage_str)
                stage_data[stage].append(s["different_fraction"])
                stage_energy[stage].append(s["mesh_energy"])
        except Exception:
            pass

    if not stage_data:
        print("  Could not parse C5_PHASE files.")
        return

    n_runs = max(len(v) for v in stage_data.values())
    print(f"\n  C5-PHASE Aggregate ({n_runs} runs across {len(files)} files)")
    print(f"\n  {'Cycles':>8}  {'Mean Div':>10}  {'Std Dev':>8}  "
          f"{'Min':>6}  {'Max':>6}  {'State'}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*20}")

    for stage in sorted(stage_data.keys()):
        vals = stage_data[stage]
        mean = sum(vals) / len(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mn   = min(vals)
        mx   = max(vals)
        n    = len(vals)

        if std < 0.05 and mean > 0.9:
            state = "[ok] deterministic distributed"
        elif std < 0.05 and mean < 0.1:
            state = "[x] deterministic collapsed"
        elif mean > 0.6:
            state = "~ stochastic (mostly distributed)"
        elif mean > 0.3:
            state = "~ stochastic (bistable)"
        else:
            state = "~ stochastic (mostly collapsed)"

        print(f"  {stage:>8}  {mean:>10.1%}  {std:>8.1%}  "
              f"{mn:>6.0%}  {mx:>6.0%}  {state}  (n={n})")

    print(f"\n  Key findings:")
    for stage in sorted(stage_data.keys()):
        vals = stage_data[stage]
        mean = sum(vals) / len(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if std < 0.05 and mean > 0.9:
            print(f"    {stage} cycles -> always 100% (deterministic distributed) <- USE THIS")
        elif std < 0.05 and mean < 0.1:
            print(f"    {stage} cycles -> always   0% (deterministic collapsed)")


def main():
    parser = argparse.ArgumentParser(description="HAMesh experiment analyzer")
    parser.add_argument("--claim",     type=str, choices=["C1","C2","C3","C4","C5","C5_PHASE"])
    parser.add_argument("--file",      type=str)
    parser.add_argument("--sessions",  action="store_true")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate all C5_PHASE runs into mean +/- std table")
    parser.add_argument("--all",       action="store_true", default=True)
    args = parser.parse_args()

    if not LOGS_DIR.exists():
        print("No ham_logs/ directory found.")
        print("Run an experiment first: python ham_experiment.py --claim C1")
        sys.exit(1)

    def report_c5_phase(data: dict):
        print("\n  +======================================================+")
        print("  |  C5-PHASE: Multi-Hop Phase Transition                |")
        print("  +======================================================+")
        print(f"  Hypothesis: {data['hypothesis']}")
        print()
        print(f"  {'Cycles':>8}  {'Energy':>8}  {'Diversity':>10}  {'State'}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*15}")
        for stage, s in sorted(data["stages"].items(), key=lambda x: int(x[0])):
            df = s["different_fraction"]
            state = "[ok] distributed" if df > 0.4 else "~ partial" if df > 0.1 else "[x] collapsed"
            print(f"  {stage:>8}  {s['mesh_energy']:>8.1f}  {df:>10.0%}  {state}")
        print()
        print(f"  Collapse: {data.get('conclusion', '?')}")

    reporters = {"C1": report_c1, "C2": report_c2,
                 "C3": report_c3, "C4": report_c4, "C5": report_c5,
                 "C5-PHASE": report_c5_phase, "C5_PHASE": report_c5_phase}

    print("=" * 58)
    print("  HAMesh Analyzer")
    print("=" * 58)

    # Specific file
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        data = load_json(path)
        claim = data.get("claim")
        if claim and claim in reporters:
            reporters[claim](data)
        else:
            print(json.dumps(data, indent=2))
        return

    # Specific claim
    if args.claim:
        path = latest_for_claim(args.claim)
        if not path:
            print(f"No results found for {args.claim}. Run: python ham_experiment.py --claim {args.claim}")
            sys.exit(1)
        print(f"  File: {path.name}")
        reporters[args.claim](load_json(path))
        return

    # Aggregate C5_PHASE
    if args.aggregate:
        aggregate_c5_phase()
        return

    # Sessions
    if args.sessions:
        session_files = sorted(LOGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not session_files:
            print("  No session logs found.")
            print("  Run: python ham_collective.py --log my_session")
        for f in session_files[:5]:
            report_session(f)
        return

    # Default: summarize everything
    found_any = False
    for claim in ["C1", "C2", "C3", "C4", "C5"]:
        path = latest_for_claim(claim)
        if path:
            found_any = True
            print(f"\n  {claim}: {path.name}")
            try:
                data = load_json(path)
                conclusion = data.get("conclusion", "?")
                # Quick one-liner summary
                if claim == "C1":
                    meshes = data.get("meshes", {})
                    overlaps = [m["avg_pairwise_overlap"] for m in meshes.values()]
                    cohs = [m["avg_coherence"] for m in meshes.values()]
                    print(f"    Overlap: {overlaps}  Coherence: {cohs}")
                elif claim == "C4":
                    frac = data.get("cross_domain_fraction", "?")
                    total = len(data.get("insights", []))
                    print(f"    Cross-domain: {frac:.1%} of {total} insights")
                elif claim == "C5":
                    for mname, m in data.get("meshes", {}).items():
                        print(f"    [{mname}] different: {m['different_fraction']:.1%}")
                print(f"    -> {conclusion}")
            except Exception as e:
                print(f"    Error reading: {e}")

    if not found_any:
        print("\n  No experiment results found in ham_logs/")
        print("  Run: python ham_experiment.py --claim C1")
        return

    print("\n\n  For detail on any claim:")
    print("    python ham_analyze.py --claim C4")
    print("\n  To check interactive session logs:")
    print("    python ham_analyze.py --sessions")
    print("\n  To log future sessions:")
    print("    python ham_collective.py --log session_name")


if __name__ == "__main__":
    main()
