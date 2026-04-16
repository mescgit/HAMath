"""
HAM Distiller — Extract LLM knowledge into Holographic Memory

The LLM's knowledge is compressed in its weights. This script
decompresses it: Bonsai generates explanations, we embed them,
and fold them into the HAM as holographic interference patterns.

The result is a mesh that contains the LLM's knowledge in a form
that supports multi-hop diffraction — reasoning the LLM alone
can't do because it has no persistent associative memory.

Usage:
  1. Start Bonsai:
     ./llama-server.exe -m bonsai-8b.gguf --port 22334 --ctx-size 8192 --embedding --pooling mean

  2. python ham_distill.py [--topics science,history,math] [--depth 2] [--save ham_state.pt]
"""

import argparse
import time
import sys
import os

import torch

from ham_core import HolographicMesh
from ham_brain import embed, generate, MESH_SAVE, API_BASE

# ---------------------------------------------------------------------------
# Knowledge taxonomy — broad enough to cover general knowledge
# ---------------------------------------------------------------------------

TOPIC_SEEDS = {
    "science": [
        "physics", "chemistry", "biology", "astronomy", "geology",
        "quantum mechanics", "thermodynamics", "evolution", "genetics",
        "electromagnetism",
    ],
    "math": [
        "algebra", "calculus", "geometry", "probability", "statistics",
        "linear algebra", "number theory", "topology", "set theory",
        "graph theory",
    ],
    "history": [
        "ancient civilizations", "Roman Empire", "Renaissance",
        "Industrial Revolution", "World War I", "World War II",
        "Cold War", "ancient Greece", "medieval Europe", "Age of Exploration",
    ],
    "technology": [
        "artificial intelligence", "machine learning", "neural networks",
        "computer architecture", "cryptography", "internet",
        "operating systems", "databases", "programming languages",
        "quantum computing",
    ],
    "philosophy": [
        "epistemology", "ethics", "logic", "metaphysics", "existentialism",
        "stoicism", "utilitarianism", "empiricism", "rationalism",
        "philosophy of mind",
    ],
    "nature": [
        "ecology", "climate", "oceans", "forests", "deserts",
        "photosynthesis", "food chains", "biodiversity", "plate tectonics",
        "water cycle",
    ],
}


# ---------------------------------------------------------------------------
# Distillation engine
# ---------------------------------------------------------------------------

def distill_topic(ham, topic, depth=1, breadth=5):
    """
    Extract knowledge about a topic from the LLM and fold it into the HAM.

    depth=1: just the topic explanation
    depth=2: also expand into subtopics
    """
    print(f"  [{topic}]")

    # Phase 1: Get the LLM's explanation of this topic
    explanation = generate(
        f"Explain {topic} clearly and concisely in 2-3 paragraphs. "
        f"Cover the key concepts, principles, and significance.",
        system="You are a knowledgeable teacher. Give clear, factual explanations.",
        max_tokens=400,
    )

    # Embed both sides
    topic_emb = embed(topic)
    explanation_emb = embed(explanation[:500])

    # Fold the association: topic -> explanation
    ham.learn(topic_emb, explanation_emb, topic, explanation[:300])

    # Self-reinforce the topic embedding
    ham.fold(topic_emb, topic_emb, strength=0.3)

    count = 1
    print(f"    core concept folded")

    # Phase 2: Expand into subtopics if depth > 1
    if depth >= 2:
        subtopics_text = generate(
            f"List exactly {breadth} important subtopics or key concepts within {topic}. "
            f"One per line, no numbering, no explanation, just the concept name.",
            system="Be precise. One concept per line.",
            max_tokens=200,
        )

        subtopics = [
            s.strip().strip('-').strip('•').strip()
            for s in subtopics_text.strip().split('\n')
            if s.strip() and len(s.strip()) > 2
        ][:breadth]

        for sub in subtopics:
            sub_explanation = generate(
                f"Explain {sub} in the context of {topic}. Be concise (1-2 paragraphs).",
                system="You are a knowledgeable teacher. Be factual and clear.",
                max_tokens=300,
            )

            sub_emb = embed(sub)
            sub_exp_emb = embed(sub_explanation[:400])

            # Store the subtopic knowledge
            ham.learn(sub_emb, sub_exp_emb, sub, sub_explanation[:300])

            # Cross-link: topic <-> subtopic (bidirectional)
            ham.fold(topic_emb, sub_emb, strength=0.7)
            ham.fold(sub_emb, topic_emb, strength=0.5)

            count += 1
            print(f"    + {sub}")

    return count


def distill_cross_links(ham, all_topics):
    """
    Ask the LLM to identify connections between topics across domains,
    then fold those bridges into the mesh.
    """
    if len(all_topics) < 4:
        return 0

    print("\n  Cross-linking domains...")
    links = 0

    # Sample pairs from different domains
    pairs_text = generate(
        f"Given these topics: {', '.join(all_topics[:30])}\n\n"
        f"List 10 meaningful connections between pairs of topics from DIFFERENT domains. "
        f"Format: TopicA | TopicB | one-sentence explanation of connection\n"
        f"Only list connections that are genuinely insightful.",
        system="You are finding interdisciplinary connections. Be specific and accurate.",
        max_tokens=600,
    )

    for line in pairs_text.strip().split('\n'):
        parts = line.split('|')
        if len(parts) >= 2:
            a = parts[0].strip().strip('-').strip()
            b = parts[1].strip()
            explanation = parts[2].strip() if len(parts) > 2 else f"{a} relates to {b}"

            if a and b and len(a) > 2 and len(b) > 2:
                a_emb = embed(a)
                b_emb = embed(b)
                link_emb = embed(explanation[:200])

                # Fold the bridge: A -> B, B -> A, and the explanation
                ham.fold(a_emb, b_emb, strength=0.6)
                ham.fold(b_emb, a_emb, strength=0.6)
                ham.remember(link_emb, explanation[:200])
                ham.fold(a_emb, link_emb, strength=0.4)

                links += 1
                print(f"    {a} <-> {b}")

    return links


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distill LLM knowledge into HAM")
    parser.add_argument("--topics", type=str, default="science,technology",
                        help="Comma-separated domain names from: " +
                             ", ".join(TOPIC_SEEDS.keys()))
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2],
                        help="1=core concepts only, 2=expand subtopics")
    parser.add_argument("--breadth", type=int, default=5,
                        help="Subtopics per topic (depth=2)")
    parser.add_argument("--save", type=str, default=MESH_SAVE,
                        help="Where to save the enriched HAM")
    parser.add_argument("--load", type=str, default=None,
                        help="Load existing HAM to add knowledge to")
    parser.add_argument("--cross-link", action="store_true", default=True,
                        help="Find and fold cross-domain connections")
    args = parser.parse_args()

    print("=" * 62)
    print("   HAM Distiller — LLM Knowledge -> Holographic Memory")
    print("=" * 62)

    # Connect to Bonsai
    print("\n  Connecting to Bonsai...", end=" ", flush=True)
    try:
        test_vec = embed("hello")
        dim = test_vec.shape[0]
        print(f"OK (dim={dim})")
    except Exception as e:
        print(f"FAILED\n  {e}")
        sys.exit(1)

    # Load or create HAM
    if args.load and os.path.exists(args.load):
        print(f"  Loading existing HAM from {args.load}...")
        ham = HolographicMesh.load(args.load)
        print(f"    {ham.n_folds} folds, {len(ham.memories)} memories")
    else:
        print(f"  Creating fresh HAM (dim={dim})...")
        ham = HolographicMesh(dim)

    # Parse requested domains
    domains = [d.strip() for d in args.topics.split(',')]
    all_topics = []

    print(f"\n  Distilling: {', '.join(domains)}  (depth={args.depth}, breadth={args.breadth})")
    print("-" * 62)

    t0 = time.time()
    total_concepts = 0

    for domain in domains:
        if domain not in TOPIC_SEEDS:
            print(f"\n  Unknown domain '{domain}', skipping. "
                  f"Available: {', '.join(TOPIC_SEEDS.keys())}")
            continue

        topics = TOPIC_SEEDS[domain]
        print(f"\n  === {domain.upper()} ({len(topics)} seed topics) ===\n")

        for topic in topics:
            n = distill_topic(ham, topic, depth=args.depth, breadth=args.breadth)
            total_concepts += n
            all_topics.append(topic)

    # Cross-domain linking
    if args.cross_link and len(domains) > 1:
        links = distill_cross_links(ham, all_topics)
        print(f"    {links} cross-domain bridges folded")

    elapsed = time.time() - t0

    # Save
    ham.save(args.save)

    print("\n" + "=" * 62)
    print(f"  Distillation complete!")
    print(f"    Concepts distilled : {total_concepts}")
    print(f"    Total folds        : {ham.n_folds}")
    print(f"    Total memories     : {len(ham.memories)}")
    print(f"    Mesh energy        : {ham.stats()['energy']:.1f}")
    print(f"    Time               : {elapsed:.1f}s")
    print(f"    Saved to           : {args.save}")
    print("=" * 62)
    print(f"\n  Now run: python ham_brain.py")
    print(f"  The HAM will use its distilled knowledge to reason.\n")


if __name__ == "__main__":
    main()
