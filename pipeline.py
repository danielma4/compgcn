"""Two-stage edge pipeline evaluator: existence (Stage 1) -> relation (Stage 2).

Loads the Stage-1 and Stage-2 checkpoints (single files, or ``.folds`` lists from
cross-validation), scores ONE fixed test candidate set (held-out true edges + 10:1
within-building type-constrained tail-corruption negatives), and reports the metric
views from metrics.py. With ``.folds`` lists, fold i of Stage 1 is paired with fold i
of Stage 2 (same building allocation + seed) and every metric is reported as mean±std
over the folds.

Usage:
    python pipeline.py --exist_load checkpoints/link_exist_mortar.folds \
                       --rel_load   checkpoints/rel_pred_mortar.folds --gpu -1
"""

import argparse
import random

import numpy as np
import torch

from helper import set_gpu
from run import Runner
from link_existence import LinkRunner
import presence as P
import metrics as M


def load_runner(cls, ckpt_path, gpu):
    """Rebuild the exact model from the checkpoint's saved args, then load weights."""
    state = torch.load(ckpt_path, map_location='cpu')
    saved = dict(state['args']); saved['gpu'] = gpu; saved['restore'] = False
    for k in ('ent_class_idx', 'ent_class_mask'):
        saved.pop(k, None)
    runner = cls(argparse.Namespace(**saved))
    runner.load_model(ckpt_path)
    return runner


def load_folds(cls, path, gpu):
    """One runner per fold if path is a .folds list, else a single-element list."""
    ckpts = open(path).read().strip().splitlines() if path.endswith('.folds') else [path]
    return [load_runner(cls, c, gpu) for c in ckpts]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exist_load', required=True, help='Stage-1 checkpoint or .folds list')
    ap.add_argument('--rel_load',   required=True, help='Stage-2 checkpoint or .folds list')
    ap.add_argument('--gpu', default='-1')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Existence decision threshold (default: the value baked into the Stage-1 model)')
    ap.add_argument('--neg_ratio', type=float, default=10.0)
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    set_gpu(args.gpu)
    exist_models = load_folds(LinkRunner, args.exist_load, args.gpu)
    rel_models   = load_folds(Runner,     args.rel_load,   args.gpu)
    if len(exist_models) != len(rel_models):
        raise ValueError(f'Stage-1 has {len(exist_models)} folds but Stage-2 has {len(rel_models)}; '
                         'they must match to pair fold-i with fold-i.')
    ref   = exist_models[0]
    thr   = args.threshold if args.threshold is not None else getattr(ref.p, 'threshold', 0.6)
    split = args.split
    target_rels = ref.target_rels

    # One fixed candidate set (positives + within-building tail-corruption negatives),
    # built once with a fixed seed so every fold is scored on the identical test set.
    rng = random.Random(ref.p.seed)
    e2c = P.build_ent2classes(ref)
    allowed, sig_rel = P.build_signatures(ref, e2c)
    known = P.known_pairs(ref)
    eset  = P.build_eval_set(ref, split, e2c, allowed, sig_rel, known, args.neg_ratio, rng)
    is_pos   = eset['is_pos']
    true_rel = eset['true_r']
    pairs    = ([(s, o) for (s, o, _) in eset['cov']] +
                [(s, o) for (s, o, _) in eset['unc']] +
                [(s, o) for (s, o, _) in eset['neg']])

    st = eset['stats']
    print(f"\n## Split '{split}': pos {st['n_pos']}, neg {st['n_neg']}, "
          f"base-rate {st['n_pos'] / (st['n_pos'] + st['n_neg']):.3f}, "
          f"{len(exist_models)} fold(s), existence thr {thr:.2f}")

    fold_results = []
    for i, (em, rm) in enumerate(zip(exist_models, rel_models)):
        score       = em.score_existence(eset)                       # existence prob per candidate
        pred_rel    = rm.score_relations_for_pairs(split, pairs)     # argmax relation per candidate
        ranks, rrel = rm.relation_ranks(split)                       # Stage-2 alone, over true edges
        n_params    = getattr(em, 'n_params', 0) + getattr(rm, 'n_params', 0)
        train_time  = getattr(em, 'train_time', 0.0) + getattr(rm, 'train_time', 0.0)
        fold_results.append(M.evaluate_fold(
            score, is_pos, true_rel, pred_rel, ranks, rrel, target_rels, thr,
            n_params=n_params, train_time=train_time))
        print(f"  scored fold {i + 1}/{len(exist_models)}")

    agg = M.aggregate(fold_results)
    print(M.format_report(agg, target_rels, ref.id2rel, thr,
                          title='CompGCN two-stage (existence: type+names | relation: type)',
                          n_folds=len(fold_results)))


if __name__ == '__main__':
    main()
