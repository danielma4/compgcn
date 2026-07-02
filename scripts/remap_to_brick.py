"""Remap local-namespace spatial types and predicates to canonical Brick equivalents.

Targets the converted TSV triple files (subject\tpredicate\tobject). Handles
buildings that mix Brick schema with a local IFC/BIM spatial hierarchy
(e.g. building:Room, building:Level, building:locatedIn) where those local
types have no brick: counterpart, making transfer learning fail.

Two remappings:
  - rdf:type objects: any *:LocalName whose local name matches a known spatial
    class is replaced with its brick: equivalent (Room->brick:Room, etc.)
  - predicates: local-namespace predicates whose local name matches a known
    relation are replaced with a Brick equivalent (locatedIn->brick:isLocationOf)

Run on an already-converted dataset directory to produce a remapped copy:
    python scripts/remap_to_brick.py --in-dir data/ttl_converted --out-dir data/ttl_converted_brick
"""

import argparse
from pathlib import Path

# Local class name -> brick: equivalent for rdf:type triples
TYPE_REMAP = {
    "Room":     "brick:Room",
    "Level":    "brick:Floor",
    "Floor":    "brick:Floor",
    "Story":    "brick:Floor",
    "Building": "brick:Building",
    "Site":     "brick:Site",
    "Space":    "brick:Space",
    "Zone":     "brick:Zone",
}

# Explicit REC/local-namespace predicate -> brick: equivalent.
# Mirrors REC_TO_BRICK_REL in ttl_to_compgcn.py; applied here as a post-process
# fallback on already-converted TSV files where the converter ran without this mapping.
PRED_REMAP = {
    "locatedIn":    "brick:hasLocation",   # special: direction fix
    "isLocationOf": "brick:isLocationOf",
    "hasLocation":  "brick:hasLocation",
    "feeds":        "brick:feeds",
    "isFedBy":      "brick:isFedBy",
    "hasPoint":     "brick:hasPoint",
    "isPointOf":    "brick:isPointOf",
    "hasPart":      "brick:hasPart",
    "isPartOf":     "brick:isPartOf",
    "includes":     "brick:hasPart",       # special: collection membership
}

SCHEMA_PREFIXES = {"brick", "rdf", "rdfs", "owl", "xsd", "ref", "tag", "unit"}


def is_schema(token):
    return ":" in token and token.split(":", 1)[0] in SCHEMA_PREFIXES


def remap(s, p, o):
    if not is_schema(p):
        loc = p.split(":", 1)[1] if ":" in p else p
        if loc in PRED_REMAP:
            p = PRED_REMAP[loc]

    if p == "rdf:type" and not is_schema(o):
        loc = o.split(":", 1)[1] if ":" in o else o
        if loc in TYPE_REMAP:
            o = TYPE_REMAP[loc]

    return s, p, o


def process_file(src, dst):
    seen = set()
    out = []
    type_remaps = pred_remaps = 0

    with open(src) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            orig = tuple(parts)
            s, p, o = remap(*parts)
            if p != orig[1]:
                pred_remaps += 1
            if o != orig[2]:
                type_remaps += 1
            triple = (s, p, o)
            if triple not in seen:
                seen.add(triple)
                out.append(triple)

    with open(dst, "w") as f:
        for s, p, o in out:
            f.write(f"{s}\t{p}\t{o}\n")

    return type_remaps, pred_remaps, len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir",  required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src, dst = Path(args.in_dir), Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    total_type = total_pred = 0
    for f in sorted(src.glob("*.txt")):
        tr, pr, n = process_file(f, dst / f.name)
        total_type += tr; total_pred += pr
        print(f"  {f.name}: {n} triples  ({tr} type remaps, {pr} pred remaps)")

    print(f"\nDone -> {dst}   type remaps: {total_type}   pred remaps: {total_pred}")


if __name__ == "__main__":
    main()
