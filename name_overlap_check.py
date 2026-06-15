"""Model-free test: does s-o name overlap separate true edges from type-compatible
non-edges, and does it transfer across buildings? Features are parameter-free, so
a high AUROC on an unseen building means the naming convention itself transfers.

  brick mode (default): reads the converted data/brick split files, groups by building.
  ttl mode:             reads raw .ttl files (one building per file) via the
                        converter's extract_triples, so folding/namespaces match.

Usage:
    python name_overlap_check.py
    python name_overlap_check.py --mode ttl --ttl-dir data/mortar
"""

import re
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import roc_auc_score

TARGET = {'brick:haspoint', 'brick:haspart', 'brick:feeds'}
BRICK_FILES = ['data/brick/train_graph.txt', 'data/brick/valid_graph.txt', 'data/brick/test_graph.txt',
               'data/brick/train.txt', 'data/brick/valid.txt', 'data/brick/test.txt']
SEED, NEG_PER_POS = 41504, 10


def building_of(uri):
    u = uri.lower()
    for b in ('ebu3b', 'soda_hall', 'rice', 'ibm_b3', 'gtc', 'ghc'):
        if b in u:
            return b
    return u.split(':')[0].split('/')[0]


def local_name(uri):
    return uri.split('#')[-1] if '#' in uri else uri.split(':')[-1]


def tokens(uri):
    name = local_name(uri).lower()
    name = re.sub(r'(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])', '_', name)   # split digit/letter runs
    return {p for p in re.split(r'[^a-z0-9]+', name) if p}


def ngrams(uri, n=3):
    s = re.sub(r'[^a-z0-9]', '', local_name(uri).lower())
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def feats(s, o):
    ts, to = tokens(s), tokens(o)
    nums = {t for t in ts if t.isdigit()} & {t for t in to if t.isdigit()}
    return {'tok_jaccard': jaccard(ts, to),
            'id_shared':   1.0 if nums else 0.0,
            'char3_jac':   jaccard(ngrams(s), ngrams(o))}


def load_brick():
    """data/brick split files -> {building: [(s, r, o)]} (grouped by entity prefix)."""
    by_b = defaultdict(list)
    for path in BRICK_FILES:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            s, r, o = (x.lower() for x in line.split('\t'))
            by_b[building_of(s)].append((s, r, o))
    return by_b


def load_ttls(ttl_dir):
    """raw .ttl files -> {filename-stem: [(s, r, o)]}, folded like the real converter."""
    sys.path.insert(0, 'scripts')
    import ttl_to_compgcn as conv
    by_b = {}
    for p in sorted(Path(ttl_dir).glob('*.ttl')):
        trips = conv.extract_triples(p, include_types=True, fold_inverses=True, merge_schema=True)
        by_b[p.stem] = [(s.lower(), r.lower(), o.lower()) for s, r, o in trips]
    return by_b


def report(by_building):
    rng = random.Random(SEED)
    pos, pool, true_pairs = (defaultdict(lambda: defaultdict(list)),
                             defaultdict(lambda: defaultdict(list)), defaultdict(set))
    for b, triples in by_building.items():
        for s, r, o in triples:
            if r in TARGET:
                pos[b][r].append((s, o)); pool[b][r].append(o); true_pairs[b].add((s, o))

    fnames = ['tok_jaccard', 'id_shared', 'char3_jac']
    print(f"{'building':<12}{'#pos':>7}{'#rel':>5}   " + "".join(f"{f:>13}" for f in fnames))
    print('-' * 74)
    for b in sorted(pos):
        labels, scores, npos = [], {f: [] for f in fnames}, 0
        for r, pairs in pos[b].items():
            objs = pool[b][r]
            for s, o in pairs:
                npos += 1
                fp = feats(s, o); labels.append(1)
                for f in fnames:
                    scores[f].append(fp[f])
                for _ in range(NEG_PER_POS):                      # type-plausible neg: same rel, corrupted tail
                    on = objs[rng.randrange(len(objs))]
                    for _retry in range(5):
                        if on != o and (s, on) not in true_pairs[b]:
                            break
                        on = objs[rng.randrange(len(objs))]
                    fn = feats(s, on); labels.append(0)
                    for f in fnames:
                        scores[f].append(fn[f])
        if npos == 0 or len(set(labels)) < 2:
            print(f"{b:<12}{npos:>7}{len(pos[b]):>5}   (insufficient data)")
            continue
        aucs = {f: roc_auc_score(labels, scores[f]) for f in fnames}
        print(f"{b:<12}{npos:>7}{len(pos[b]):>5}   " + "".join(f"{aucs[f]:>13.3f}" for f in fnames))

    print('-' * 74)
    print("AUROC = P(true edge scores above a type-plausible non-edge). 0.5 = no signal.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['brick', 'ttl'], default='brick')
    ap.add_argument('--ttl-dir', default='data/mortar')
    args = ap.parse_args()
    report(load_ttls(args.ttl_dir) if args.mode == 'ttl' else load_brick())


if __name__ == '__main__':
    main()
