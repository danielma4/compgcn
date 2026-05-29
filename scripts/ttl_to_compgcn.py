"""Convert Brick-schema TTL files into CompGCN's TSV triple format.

Reads multiple .ttl files, extracts URI->URI triples (and optionally
rdf:type triples), and writes them into ``train.txt``/``valid.txt``/
``test.txt`` split by building.

Output format matches what CompGCN's ``run.py`` expects:
    subject<TAB>relation<TAB>object   (one per line, no header)
"""

import argparse
import os
from pathlib import Path

import rdflib
from rdflib import URIRef
from rdflib.namespace import RDF


# Default split: 4 buildings train, 1 valid, 1 test.
DEFAULT_SPLITS = {
    "train": ["ebu3b_brick", "ghc_brick", "gtc_brick", "ibm_b3"],
    "valid": ["soda_brick"],
    "test":  ["rice_brick"],
}


def extract_triples(ttl_path, include_types):
    """Extracts URI->URI triples from a single TTL file.

    Parameters
    ----------
    ttl_path : path to the .ttl file to parse.
    include_types : if True, also emit rdf:type triples (entity -> class).

    Returns
    -------
    A list of (subject_uri, predicate_uri, object_uri) string tuples.
    """
    g = rdflib.Graph()
    g.parse(str(ttl_path), format="turtle")

    triples = []
    for s, p, o in g:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue  # skip literals and blank nodes
        if p == RDF.type and not include_types:
            continue
        triples.append((str(s), str(p), str(o)))
    return triples


def write_split(triples, out_path):
    """Writes triples to a tab-separated CompGCN file.

    Parameters
    ----------
    triples : iterable of (subject, predicate, object) string tuples.
    out_path : destination path for the .txt file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def main():
    """Parses CLI args and converts all Brick TTLs into CompGCN splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ttl-dir",
        default="data/kaggle_brick",
        help="Directory containing the Brick .ttl files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/brick",
        help="Output directory for train/valid/test .txt files.",
    )
    parser.add_argument(
        "--no-types",
        action="store_true",
        help="Exclude rdf:type triples from the output.",
    )
    args = parser.parse_args()

    ttl_dir = Path(args.ttl_dir)
    out_dir = Path(args.out_dir)
    include_types = not args.no_types

    print(f"## Reading TTLs from: {ttl_dir}")
    print(f"## Writing CompGCN data to: {out_dir}")
    print(f"## Include rdf:type triples: {include_types}")

    for split_name, building_stems in DEFAULT_SPLITS.items():
        all_triples = []
        for stem in building_stems:
            ttl_path = ttl_dir / f"{stem}.ttl"
            if not ttl_path.exists():
                raise FileNotFoundError(f"Missing TTL file: {ttl_path}")
            t = extract_triples(ttl_path, include_types)
            print(f"  [{split_name}] {stem}: {len(t):,} triples")
            all_triples.extend(t)

        out_path = out_dir / f"{split_name}.txt"
        write_split(all_triples, out_path)
        print(f"  -> wrote {len(all_triples):,} triples to {out_path}")

    print("## Done.")


if __name__ == "__main__":
    main()
