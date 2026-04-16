"""
HAM Logger — Structured research logging for HAMesh experiments

Captures every significant event as a JSON record:
  - dream cycles (per-cycle attractor activations, energy, loops)
  - curiosity events (gap found, question generated, answer, insight folded)
  - query events (question, activated patterns, response, mesh state)
  - cross-pollination events (what moved from where to where)
  - mesh snapshots (attractor topology at a point in time)

All records go to a JSONL file (one JSON object per line) so you can
stream, grep, and analyze without loading everything into memory.

Usage:
  from ham_logger import HAMLogger
  logger = HAMLogger("experiment_01")

  # Attach to collective or brain
  logger.log_dream_start("science", cycles=1000)
  logger.log_attractor_snapshot("science", attractors, loops)
  logger.log_curiosity_event(gap, question, answer, activated)
  logger.log_query(question, activated, response, hops)
  logger.log_cross_pollination(shared_patterns)
"""

import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path


class HAMLogger:
    """
    Append-only structured event log for HAMesh research.

    Records go to:
      ./ham_logs/<session_name>_<timestamp>.jsonl   — raw event stream
      ./ham_logs/<session_name>_summary.json        — rolling summary (overwritten)
    """

    def __init__(self, session_name: str = "session", log_dir: str = "./ham_logs"):
        self.session_name = session_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path    = self.log_dir / f"{session_name}_{ts}.jsonl"
        self.summary_path = self.log_dir / f"{session_name}_summary.json"

        self.session_start = time.time()
        self.event_count = 0

        # Rolling counters for summary
        self.summary = {
            "session_name":    session_name,
            "started_at":      datetime.now(timezone.utc).isoformat(),
            "total_events":    0,
            "dream_runs":      [],   # list of attractor snapshots
            "curiosity_runs":  [],   # list of curiosity sessions
            "query_count":     0,
            "total_insights":  0,
            "cross_pollinations": 0,
        }

        self._write_event("session_start", {
            "session": session_name,
            "log_file": str(self.log_path),
        })

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def _write_event(self, event_type: str, data: dict):
        record = {
            "t":    round(time.time() - self.session_start, 3),
            "ts":   datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.event_count += 1
        self.summary["total_events"] = self.event_count
        self._save_summary()
        return record

    def _save_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Dream events
    # ------------------------------------------------------------------

    def log_dream_start(self, mesh_name: str, cycles: int):
        self._write_event("dream_start", {
            "mesh": mesh_name,
            "cycles": cycles,
        })

    def log_attractor_snapshot(
        self,
        mesh_name: str,
        attractors: list,      # [(text, count), ...]
        loops: list,           # [[node, node, ...], ...]
        energy_before: float,
        energy_after: float,
        new_folds: int,
        cycles: int,
    ):
        """Record the attractor state at the end of a dream run."""
        record = {
            "mesh":          mesh_name,
            "cycles":        cycles,
            "energy_before": round(energy_before, 2),
            "energy_after":  round(energy_after, 2),
            "energy_delta":  round(energy_after - energy_before, 2),
            "new_folds":     new_folds,
            "attractors": [
                {"text": text[:120], "count": count}
                for text, count in attractors[:10]
            ],
            "loops": [
                {"nodes": [n[:80] for n in loop[:10]]}
                for loop in loops[:5]
            ],
            "n_loops":     len(loops),
            "n_attractors": len(attractors),
        }
        self._write_event("attractor_snapshot", record)

        # Append to summary
        self.summary["dream_runs"].append({
            "mesh":       mesh_name,
            "cycles":     cycles,
            "top_5":      [t for t, _ in attractors[:5]],
            "n_loops":    len(loops),
            "energy_delta": round(energy_after - energy_before, 2),
        })
        self._save_summary()

    def log_cross_pollination(self, n_shared: int, mesh_pairs: list):
        """mesh_pairs: [(src_name, dst_name, n_patterns), ...]"""
        self._write_event("cross_pollination", {
            "total_shared": n_shared,
            "transfers": [
                {"from": src, "to": dst, "n": n}
                for src, dst, n in mesh_pairs
            ],
        })
        self.summary["cross_pollinations"] += 1
        self._save_summary()

    # ------------------------------------------------------------------
    # Curiosity events
    # ------------------------------------------------------------------

    def log_curiosity_start(self, n_gaps: int):
        self._write_event("curiosity_start", {"n_gaps_requested": n_gaps})

    def log_gap_found(
        self,
        gap_text: str,
        mesh_names: list,
        isolation_score: float,
        gap_type: str,  # "universal" | "domain"
    ):
        self._write_event("gap_found", {
            "text":            gap_text[:120],
            "meshes":          mesh_names,
            "isolation_score": round(isolation_score, 4),
            "gap_type":        gap_type,
        })

    def log_curiosity_insight(
        self,
        gap_text: str,
        question: str,
        answer: str,
        activated: list,   # [(sim, text, mesh_name), ...]
        cross_domain: bool,
    ):
        """
        A single Q→A generated by the curiosity engine.
        cross_domain=True if activated patterns came from multiple meshes.
        """
        meshes_activated = list({m for _, _, m in activated})
        record = {
            "gap":           gap_text[:120],
            "question":      question[:200],
            "answer":        answer[:500],
            "answer_length": len(answer),
            "top_activations": [
                {"sim": round(s, 4), "text": t[:80], "mesh": m}
                for s, t, m in activated[:5]
            ],
            "meshes_activated": meshes_activated,
            "cross_domain":     cross_domain,
            "n_domains":        len(meshes_activated),
        }
        self._write_event("curiosity_insight", record)
        self.summary["total_insights"] += 1
        self._save_summary()

    def log_curiosity_end(self, n_insights: int, n_gaps: int):
        self._write_event("curiosity_end", {
            "insights_generated": n_insights,
            "gaps_explored":      n_gaps,
        })
        self.summary["curiosity_runs"].append({
            "insights": n_insights,
            "gaps":     n_gaps,
        })
        self._save_summary()

    # ------------------------------------------------------------------
    # Query events
    # ------------------------------------------------------------------

    def log_query(
        self,
        question: str,
        hops: int,
        activated: list,   # [(sim, text, mesh_name), ...]
        response: str,
        confidence: float,
        self_taught: bool,
    ):
        meshes = list({m for _, _, m in activated}) if activated else []
        record = {
            "question":       question[:200],
            "hops":           hops,
            "confidence":     round(confidence, 4),
            "cross_domain":   len(meshes) > 1,
            "meshes_active":  meshes,
            "response_len":   len(response),
            "top_activations": [
                {"sim": round(s, 4), "text": t[:80], "mesh": m}
                for s, t, m in activated[:3]
            ],
            "self_taught":    self_taught,
        }
        self._write_event("query", record)
        self.summary["query_count"] += 1
        self._save_summary()

    # ------------------------------------------------------------------
    # Mesh snapshot  (call periodically to capture topology)
    # ------------------------------------------------------------------

    def log_mesh_snapshot(self, mesh_name: str, stats: dict, dominant: list):
        """
        stats: from ham.stats()
        dominant: from ham.dominant_memories(n=10)
        """
        self._write_event("mesh_snapshot", {
            "mesh":     mesh_name,
            "folds":    stats["folds"],
            "memories": stats["memories"],
            "energy":   round(stats["energy"], 2),
            "dominant": [
                {"sim": round(s, 4), "text": t[:80]}
                for s, _, t, _ in dominant[:10]
            ],
        })

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def print_path(self):
        print(f"  Logging to: {self.log_path}")

    @property
    def path(self):
        return str(self.log_path)
