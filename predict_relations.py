"""Predict missing relations in a Brick knowledge graph with a trained CompGCN.

The trained entity decoder scores, for a given ``(subject, relation)``, *all*
objects in a single matmul. We reuse that to do relation prediction: for every
subject and every candidate relation we score all objects and emit the
``(s, r, o)`` triples whose sigmoid confidence clears a threshold (and which
aren't already in the graph). Because the model is frozen at inference, the
graph encoder is run **once** and the resulting node/relation embeddings are
reused across every scoring batch (see Runner/model.encode).

Optionally restricts predictions to type-compatible pairs: the allowed
``(subject_class, object_class)`` signatures per relation are read off the
training edges (a hard filter, not learned).

Usage:
    python predict_relations.py -yaml config/brick.yaml \\
        -name brick_conve_corr_<timestamp> -threshold 0.9 \\
        -type_filter -out predictions.tsv
"""

import os
from collections import defaultdict

import torch

from helper import set_gpu
from run import build_parser, parse_with_yaml, Runner


def build_class_membership(runner, device):
    """Returns (M, class2id) where M[e, c] == 1 iff entity e has Brick class c.

    Types are read from rdf:type triples in *all* splits (at inference we
    legitimately know each node's type; only relational edges are targets).
    Index 0 is reserved for '<unk>' (entities with no rdf:type).
    """
    type_rel = runner.rel2id.get('rdf:type')
    if type_rel is None:
        raise ValueError("-type_filter needs rdf:type triples, but none are in the data.")

    class2id = {'<unk>': 0}
    ent2classes = defaultdict(list)
    for split in ['train', 'valid', 'test']:
        for s, r, o in runner.data[split]:
            if r == type_rel:
                cid = class2id.setdefault(runner.id2ent[o], len(class2id))
                ent2classes[s].append(cid)

    M = torch.zeros(runner.p.num_ent, len(class2id), device=device)
    for e in range(runner.p.num_ent):
        for c in (ent2classes.get(e) or [0]):
            M[e, c] = 1.0
    return M, type_rel


def build_allowed(runner, M, rel_ids, num_class, device):
    """Per-relation (num_class, num_class) boolean matrix of allowed (subj, obj)
    class signatures, learned by counting training edges."""
    R = {r: torch.zeros(num_class, num_class, device=device) for r in rel_ids}
    for s, rel, o in runner.data['train']:
        if rel in R:
            cs = M[s].nonzero(as_tuple=True)[0]
            co = M[o].nonzero(as_tuple=True)[0]
            for a in cs.tolist():
                R[rel][a, co] = 1.0
    return R


def main():
    parser = build_parser()
    parser.add_argument('--load', '-load',             dest='load_path',  default=None,   help='Checkpoint path (default: ./checkpoints/<name>)')
    parser.add_argument('--threshold', '-threshold',   type=float,        default=0.9,    help='Minimum sigmoid confidence to emit a triple')
    parser.add_argument('--topk', '-topk',             type=int,          default=0,      help='If >0, keep at most this many objects per (s, r) above threshold')
    parser.add_argument('--relations', '-relations',   default=None,      help='Comma-separated relation names to predict (default: all base relations except rdf:type)')
    parser.add_argument('--type_filter', '-type_filter', action='store_true', help='Only emit type-compatible (s, r, o) (domain/range read from train edges)')
    parser.add_argument('--pred_batch', '-pred_batch', type=int,          default=512,    help='Number of subjects scored per batch')
    parser.add_argument('--out', '-out',               dest='out_path',   default='predictions.tsv', help='Output TSV path (subject<TAB>relation<TAB>object<TAB>confidence)')
    args = parse_with_yaml(parser)

    set_gpu(args.gpu)
    runner = Runner(args)                       # loads data + builds model
    load_path = args.load_path or os.path.join('./checkpoints', args.name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path} (pass -name <run> or -load <path>)")
    runner.load_model(load_path)

    model  = runner.model
    device = runner.device
    num_ent, num_rel = runner.p.num_ent, runner.p.num_rel
    model.eval()

    # Relations to predict.
    if args.relations:
        rel_ids = [runner.rel2id[n.strip().lower()] for n in args.relations.split(',')]
    else:
        type_rel = runner.rel2id.get('rdf:type')
        rel_ids = [r for r in range(num_rel) if r != type_rel]

    print(f"## Predicting {len(rel_ids)} relations over {num_ent:,} entities "
          f"(threshold={args.threshold}, type_filter={args.type_filter})")

    with torch.no_grad():
        all_ent, all_rel = model.encode()       # ENCODE ONCE, reuse for every batch

        if args.type_filter:
            M, _ = build_class_membership(runner, device)
            R = build_allowed(runner, M, rel_ids, M.size(1), device)

        subjects  = torch.arange(num_ent, device=device)
        n_emitted = 0
        with open(args.out_path, 'w') as out:
            for rel in rel_ids:
                rel_name = runner.id2rel[rel]
                for start in range(0, num_ent, args.pred_batch):
                    sub_b  = subjects[start:start + args.pred_batch]
                    rel_b  = torch.full_like(sub_b, rel)
                    scores = model.decode(sub_b, rel_b, all_ent, all_rel)   # (B, num_ent)

                    if args.type_filter:                                    # mask type-incompatible objects
                        allowed_cls = (M[sub_b] @ R[rel]) > 0               # (B, num_class)
                        allowed_obj = (allowed_cls.float() @ M.t()) > 0     # (B, num_ent)
                        scores = scores.masked_fill(~allowed_obj, -1.0)

                    for i, s in enumerate(sub_b.tolist()):                  # drop edges already in the graph
                        known = runner.sr2o_all.get((s, rel))
                        if known:
                            scores[i, torch.tensor(known, device=device)] = -1.0

                    if args.topk > 0:
                        vals, inds = scores.topk(min(args.topk, num_ent), dim=1)
                        rows, cols, scs = [], [], []
                        for i in range(sub_b.size(0)):
                            for j in range(vals.size(1)):
                                if vals[i, j].item() >= args.threshold:
                                    rows.append(sub_b[i].item()); cols.append(inds[i, j].item()); scs.append(vals[i, j].item())
                    else:
                        keep = scores >= args.threshold
                        r_idx, c_idx = keep.nonzero(as_tuple=True)
                        rows = sub_b[r_idx].tolist()
                        cols = c_idx.tolist()
                        scs  = scores[r_idx, c_idx].tolist()

                    for s, o, sc in zip(rows, cols, scs):
                        out.write(f"{runner.id2ent[s]}\t{rel_name}\t{runner.id2ent[o]}\t{sc:.4f}\n")
                    n_emitted += len(rows)

    print(f"## Wrote {n_emitted:,} predicted triples to {args.out_path}")


if __name__ == '__main__':
    main()
