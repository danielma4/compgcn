"""Convert Brick-schema TTL files into CompGCN's TSV triple format.

One file per building. URIs are emitted in compact ``prefix:local`` form: shared
schema terms (Brick classes/relations, RDF/OWL) keep a canonical prefix so they
align across buildings and Brick versions; every other (instance) URI is prefixed
with the building stem so entities never collide across buildings.

Only the three inverse forms of the predicted relations are folded onto their
canonical direction (``isPointOf``->``hasPoint`` etc., swapping s/o). All other
relations are left as-is — they become context the GNN uses to predict the targets.

Per split, two files are written:
    <split>_graph.txt : context (message-passing graph)
    <split>.txt       : held-out target triples to predict
With --hold-rate 1.0 every target edge is held out, so a building's graph contains
only its non-target edges (the inductive deployment setting).
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import rdflib
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Namespace -> canonical prefix. Legacy Brick/BrickFrame versions map to the same
# prefix as current Brick, which merges them. Most-specific namespaces first.
SCHEMA_NS = [
    ("https://brickschema.org/schema/Brick/ref#", "ref"),
    ("http://qudt.org/vocab/unit/", "unit"),
    ("https://brickschema.org/schema/BrickTag#", "tag"),
    ("https://brickschema.org/schema/1.0.1/BrickTag#", "tag"),
    ("https://brickschema.org/schema/1.0.2/BrickTag#", "tag"),
    ("https://brickschema.org/schema/Brick#", "brick"),
    ("https://brickschema.org/schema/1.0.1/Brick#", "brick"),
    ("https://brickschema.org/schema/1.0.2/Brick#", "brick"),
    ("https://brickschema.org/schema/1.0.1/BrickFrame#", "brick"),
    ("https://brickschema.org/schema/1.0.2/BrickFrame#", "brick"),
    (str(RDF), "rdf"), (str(RDFS), "rdfs"), (str(OWL), "owl"), (str(XSD), "xsd"),
]

# Inverse forms of the three predicted relations -> canonical direction (s/o swap).
FOLD = {"isPointOf": "hasPoint", "isPartOf": "hasPart", "isFedBy": "feeds"}

DEFAULT_TARGETS = ["brick:hasPoint", "brick:hasPart", "brick:feeds"]
DEFAULT_TEST = ["bldg6", "bldg10", "bldg16", "bldg17", "bldg22", "bldg23", "bldg25", "bldg26", "bldg44"]
DEFAULT_VALID = ["bldg1", "bldg2", "bldg5"]


def qname(term, stem):
    """URIRef -> 'prefix:local'. Schema terms keep their shared prefix; everything
    else is prefixed with the building stem."""
    s = str(term)
    for ns, prefix in SCHEMA_NS:
        if s.startswith(ns):
            return f"{prefix}:{s[len(ns):]}"
    local = s.split("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]
    return f"{stem}:{local}"


def extract_triples(ttl_path):
    """Deduplicated (s, p, o) prefixed triples from one TTL file, folding the three
    inverse relations onto their canonical direction."""
    g = rdflib.Graph()
    g.parse(str(ttl_path), format="turtle")
    stem = ttl_path.stem
    out, seen = [], set()
    for s, p, o in g:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue  # skip literals / blank nodes
        s, p, o = qname(s, stem), qname(p, stem), qname(o, stem)
        if ":" in p:
            local = p.split(":", 1)[1]
            if local in FOLD:
                s, p, o = o, f"brick:{FOLD[local]}", s  # inverse -> canonical, swap s/o
        if (s, p, o) not in seen:
            seen.add((s, p, o))
            out.append((s, p, o))
    return out


def split_context_target(triples, targets, hold_rate, rng):
    """Partition a building's triples into (context, held-out targets). Holdout is
    random but stratified per relation: each target relation's edges are shuffled and
    a ``hold_rate`` fraction held out, so every relation keeps ~(1-rate) of its edges
    as context even in tiny buildings. Non-target edges are always context."""
    by_rel = defaultdict(list)
    context = []
    for t in triples:
        if t[1] in targets:
            by_rel[t[1]].append(t)
        else:
            context.append(t)
    held = []
    for edges in by_rel.values():
        rng.shuffle(edges)
        k = round(hold_rate * len(edges))
        held.extend(edges[:k])
        context.extend(edges[k:])
    return context, held


def write(triples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ttl-dir", default="data/mortar_ttl")
    ap.add_argument("--out-dir", default="data/brick_mortar")
    ap.add_argument("--test-buildings", default=",".join(DEFAULT_TEST))
    ap.add_argument("--valid-buildings", default=",".join(DEFAULT_VALID))
    ap.add_argument("--target-relations", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--hold-rate", type=float, default=0.4,
                    help="Fraction of each relation's target edges held out per building to "
                         "predict; the rest (~60%% by default) stay as observed context.")
    ap.add_argument("--seed", type=int, default=41504)
    args = ap.parse_args()

    ttl_dir, out_dir = Path(args.ttl_dir), Path(args.out_dir)
    targets = {r.strip() for r in args.target_relations.split(",") if r.strip()}
    test_b  = [b.strip() for b in args.test_buildings.split(",") if b.strip()]
    valid_b = [b.strip() for b in args.valid_buildings.split(",") if b.strip()]
    buildings = sorted(p.stem for p in ttl_dir.glob("*.ttl"))
    if not buildings:
        raise FileNotFoundError(f"No .ttl files in {ttl_dir}")
    held = set(valid_b) | set(test_b)
    splits = {"train": [b for b in buildings if b not in held],
              "valid": valid_b, "test": test_b}
    rng = random.Random(args.seed)

    print(f"## {ttl_dir} -> {out_dir}  | targets={sorted(targets)} hold_rate={args.hold_rate}")
    print(f"## buildings: train {len(splits['train'])}, valid {len(splits['valid'])}, test {len(splits['test'])}")
    for split in ("train", "valid", "test"):
        context, held_targets = [], []
        for stem in splits[split]:
            path = ttl_dir / f"{stem}.ttl"
            if not path.exists():
                raise FileNotFoundError(f"Missing TTL: {path}")
            ctx, h = split_context_target(extract_triples(path), targets, args.hold_rate, rng)
            context.extend(ctx)
            held_targets.extend(h)
        write(context, out_dir / f"{split}_graph.txt")
        write(held_targets, out_dir / f"{split}.txt")
        print(f"  {split}: {len(context):,} context, {len(held_targets):,} held-out targets")
    print("## Done.")


if __name__ == "__main__":
    main()
