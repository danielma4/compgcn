#!/usr/bin/env python3
"""Analyse name-token similarity between head and tail entities per relation type.

Hypothesis: Stage-1 uses name-similarity features. If MORTAR training pairs have
much higher intra-pair name similarity than GEAR test pairs, the feature is poorly
calibrated for transfer.

Reads the pre-converted TSV triple files directly (no rdflib needed).

Usage:
    python scripts/name_similarity_analysis.py
"""

import math
import re
from collections import defaultdict
from pathlib import Path

# ── Config ───────────────────────────────────────────────────
MORTAR_FILES = [
    "data/brick_mortar/train.txt",
    "data/brick_mortar/valid.txt",
    "data/brick_mortar/test.txt",
]
GEAR_FILES = [
    "data/ttl_converted/test.txt",
    "data/ttl_converted/test_graph.txt",
]
GEAR_TARGET_FILES = [
    "data/ttl_converted/test.txt",
]
TARGET_RELS = {"brick:feeds", "brick:hasPart", "brick:hasPoint"}
# ─────────────────────────────────────────────────────────────


def local_name(entity: str) -> str:
    """Strip namespace prefix, return the local identifier."""
    return entity.split(":", 1)[1] if ":" in entity else entity


def tokenise(name: str) -> list[str]:
    """Split on non-alphanumeric runs, lowercase, drop empties and pure-number tokens."""
    raw = re.split(r"[^a-zA-Z0-9]+", name)
    return [t.lower() for t in raw if t and not t.isdigit()]


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenise(a)), set(tokenise(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_bigrams(s: str) -> set:
    s = s.lower()
    return {s[i:i+2] for i in range(len(s) - 1)}


def bigram_jaccard(a: str, b: str) -> float:
    ba, bb = char_bigrams(a), char_bigrams(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def common_prefix_ratio(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k / max(len(a), len(b), 1)


def shared_token_count(a: str, b: str) -> int:
    return len(set(tokenise(a)) & set(tokenise(b)))


def load_triples(paths):
    triples = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [SKIP] {p} not found")
            continue
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
    return triples


def analyse(triples, label):
    """Per-relation name similarity stats."""
    by_rel = defaultdict(list)
    for s, p, o in triples:
        # Skip schema-to-schema or schema-to-instance (e.g. rdf:type rows)
        s_ns = s.split(":", 1)[0] if ":" in s else ""
        o_ns = o.split(":", 1)[0] if ":" in o else ""
        if o_ns in ("brick", "rdf", "rdfs", "owl", "xsd"):
            continue
        sn = local_name(s)
        on = local_name(o)
        by_rel[p].append((sn, on))

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  {'Relation':<22} {'N':>6}  {'TokJacc':>8}  {'BigramJ':>8}  "
          f"{'PfxRatio':>9}  {'SharedTok':>9}  {'AnyShared%':>10}")
    print(f"  {'-'*22} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}")

    results = {}
    all_pairs = []
    for rel in sorted(by_rel):
        pairs = by_rel[rel]
        all_pairs.extend(pairs)
        tj  = [token_jaccard(a, b)       for a, b in pairs]
        bj  = [bigram_jaccard(a, b)      for a, b in pairs]
        pfx = [common_prefix_ratio(a, b) for a, b in pairs]
        stk = [shared_token_count(a, b)  for a, b in pairs]
        any_shared_pct = 100 * sum(1 for v in tj if v > 0) / len(tj)

        results[rel] = {
            "n":           len(pairs),
            "tok_jaccard": round(sum(tj)/len(tj), 4),
            "bigram_jacc": round(sum(bj)/len(bj), 4),
            "prefix_ratio":round(sum(pfx)/len(pfx), 4),
            "shared_tok":  round(sum(stk)/len(stk), 3),
            "any_shared_pct": round(any_shared_pct, 1),
        }
        print(f"  {rel:<22} {len(pairs):>6}  "
              f"{results[rel]['tok_jaccard']:>8.4f}  "
              f"{results[rel]['bigram_jacc']:>8.4f}  "
              f"{results[rel]['prefix_ratio']:>9.4f}  "
              f"{results[rel]['shared_tok']:>9.3f}  "
              f"{results[rel]['any_shared_pct']:>9.1f}%")

    # All relations combined
    if all_pairs:
        tj  = [token_jaccard(a, b)       for a, b in all_pairs]
        bj  = [bigram_jaccard(a, b)      for a, b in all_pairs]
        pfx = [common_prefix_ratio(a, b) for a, b in all_pairs]
        stk = [shared_token_count(a, b)  for a, b in all_pairs]
        any_shared_pct = 100 * sum(1 for v in tj if v > 0) / len(tj)
        print(f"  {'[ALL TARGET RELS]':<22} {len(all_pairs):>6}  "
              f"{sum(tj)/len(tj):>8.4f}  "
              f"{sum(bj)/len(bj):>8.4f}  "
              f"{sum(pfx)/len(pfx):>9.4f}  "
              f"{sum(stk)/len(stk):>9.3f}  "
              f"{any_shared_pct:>9.1f}%")
        results["ALL"] = {
            "n": len(all_pairs),
            "tok_jaccard": round(sum(tj)/len(tj), 4),
            "any_shared_pct": round(any_shared_pct, 1),
        }

    return results


def null_baseline(triples, label, n_samples=5000, seed=42):
    """Random-pair similarity — what chance similarity looks like."""
    import random
    rng = random.Random(seed)
    names = list({local_name(s) for s, _, _ in triples}
               | {local_name(o) for _, _, o in triples
                  if o.split(":",1)[0] not in ("brick","rdf","rdfs","owl","xsd")})
    if len(names) < 2:
        return
    samples = [(rng.choice(names), rng.choice(names)) for _ in range(n_samples)]
    tj  = [token_jaccard(a, b)  for a, b in samples]
    bj  = [bigram_jaccard(a, b) for a, b in samples]
    print(f"\n  NULL BASELINE ({label} — {n_samples} random pairs):")
    print(f"    TokJacc {sum(tj)/len(tj):.4f}   BigramJacc {sum(bj)/len(bj):.4f}")


def token_overlap_examples(triples, rel, n=8):
    """Print representative pairs sorted by similarity."""
    pairs = [(local_name(s), local_name(o))
             for s, p, o in triples
             if p == rel and o.split(":",1)[0] not in ("brick","rdf","rdfs","owl","xsd")]
    if not pairs:
        return
    scored = sorted([(token_jaccard(a, b), a, b) for a, b in pairs], reverse=True)
    hi = scored[:n//2]
    lo = scored[-(n//2):]
    print(f"\n  {rel} — high-similarity pairs:")
    for sim, a, b in hi:
        print(f"    {sim:.3f}  {a[:45]:<45}  {b[:45]}")
    print(f"  {rel} — low-similarity pairs:")
    for sim, a, b in lo:
        print(f"    {sim:.3f}  {a[:45]:<45}  {b[:45]}")


def compare(m_results, g_results):
    print(f"\n{'='*72}")
    print("  MORTAR vs GEAR — Token Jaccard comparison")
    print(f"{'='*72}")
    print(f"  {'Relation':<22} {'MORTAR TJ':>10}  {'GEAR TJ':>9}  "
          f"{'MORTAR Any%':>12}  {'GEAR Any%':>10}  {'Δ TJ':>8}")
    print(f"  {'-'*22} {'-'*10}  {'-'*9}  {'-'*12}  {'-'*10}  {'-'*8}")
    for rel in sorted(set(m_results) | set(g_results)):
        m = m_results.get(rel, {})
        g = g_results.get(rel, {})
        mtj = m.get("tok_jaccard", float("nan"))
        gtj = g.get("tok_jaccard", float("nan"))
        delta = gtj - mtj if not math.isnan(mtj) and not math.isnan(gtj) else float("nan")
        print(f"  {rel:<22} {mtj:>10.4f}  {gtj:>9.4f}  "
              f"{m.get('any_shared_pct', float('nan')):>11.1f}%  "
              f"{g.get('any_shared_pct', float('nan')):>9.1f}%  "
              f"{delta:>+8.4f}")


def main():
    print("Loading MORTAR triples...")
    mortar = load_triples(MORTAR_FILES)
    print(f"  {len(mortar):,} triples")

    print("Loading GEAR triples...")
    gear_all    = load_triples(GEAR_FILES)
    gear_target = load_triples(GEAR_TARGET_FILES)
    print(f"  {len(gear_all):,} triples total, {len(gear_target):,} target triples")

    # Only target-relation triples for the main analysis
    mortar_target = [(s,p,o) for s,p,o in mortar if p in TARGET_RELS]
    gear_target_t  = [(s,p,o) for s,p,o in gear_target if p in TARGET_RELS]

    m_res = analyse(mortar_target, "MORTAR — target-relation pairs")
    g_res = analyse(gear_target_t, "GEAR — target-relation pairs")

    # Null baselines
    null_baseline(mortar_target, "MORTAR")
    null_baseline(gear_target_t,  "GEAR")

    # Side-by-side comparison
    compare(m_res, g_res)

    # Representative examples for each relation in GEAR
    print(f"\n{'='*72}")
    print("  GEAR pair examples (sorted by token similarity)")
    print(f"{'='*72}")
    for rel in sorted(TARGET_RELS):
        token_overlap_examples(gear_target_t, rel, n=6)

    print(f"\n{'='*72}")
    print("  MORTAR pair examples")
    print(f"{'='*72}")
    for rel in sorted(TARGET_RELS):
        token_overlap_examples(mortar_target, rel, n=6)


if __name__ == "__main__":
    main()
