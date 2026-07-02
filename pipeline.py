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


def load_runner(cls, ckpt_path, gpu, dataset_override=None):
    """Rebuild the exact model from the checkpoint's saved args, then load weights.

    For transfer learning (dataset_override), loads the source dataset first to
    obtain its vocabulary, then rebuilds the model on the target dataset and aligns
    embeddings slot-by-slot by matching relation/class names. GCN conv weights are
    shape-invariant and transfer directly."""
    state = torch.load(ckpt_path, map_location='cpu')
    saved = dict(state['args']); saved['gpu'] = gpu; saved['restore'] = False
    for k in ('ent_class_idx', 'ent_class_mask'):
        saved.pop(k, None)

    if not dataset_override:
        runner = cls(argparse.Namespace(**saved))
        runner.load_model(ckpt_path)
        return runner

    # --- Transfer learning: vocab-aligned weight copy ---
    # Build source runner for vocab only (weights may not load cleanly if data
    # was re-converted; we only need rel2id and class2id, not model weights).
    src_runner = cls(argparse.Namespace(**saved))
    src_rel2id   = src_runner.rel2id
    src_class2id = getattr(src_runner, 'class2id', {})
    src_num_rel  = src_runner.p.num_rel
    # Invert: checkpoint class-slot index -> class name, using current src vocab.
    # Works as long as the first N class names in the checkpoint match src_class2id.
    src_id2class = {v: k for k, v in src_class2id.items()}

    # Build target runner (new dataset)
    tgt_saved = dict(saved); tgt_saved['dataset'] = dataset_override
    tgt_runner = cls(argparse.Namespace(**tgt_saved))
    tgt_rel2id   = tgt_runner.rel2id
    tgt_class2id = getattr(tgt_runner, 'class2id', {})
    tgt_num_rel  = tgt_runner.p.num_rel

    ckpt_state = state['state_dict']
    model_dict  = tgt_runner.model.state_dict()

    # Step 1: GCN conv weights — shape-invariant, copy directly
    transfer_dict = {}
    for k, v in ckpt_state.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            transfer_dict[k] = v

    # Step 2: class embeddings — align by name using src_id2class as the key
    ckpt_ce = ckpt_state['class_embed']          # [src_num_class, dim]
    tgt_ce  = model_dict['class_embed'].clone()  # [tgt_num_class, dim]
    aligned_classes = 0
    for src_slot in range(ckpt_ce.shape[0]):
        name = src_id2class.get(src_slot)
        if name and name in tgt_class2id:
            tgt_ce[tgt_class2id[name]] = ckpt_ce[src_slot]
            aligned_classes += 1
    transfer_dict['class_embed'] = tgt_ce

    # Step 3: relation embeddings — align by name
    ckpt_re = ckpt_state['init_rel']             # [src_num_rel*2, dim]
    tgt_re  = model_dict['init_rel'].clone()     # [tgt_num_rel*2, dim]
    src_id2rel = {v: k for k, v in src_rel2id.items()}
    aligned_rels = 0
    for src_slot in range(ckpt_re.shape[0]):
        name = src_id2rel.get(src_slot)
        if name and name in tgt_rel2id:
            tgt_re[tgt_rel2id[name]] = ckpt_re[src_slot]
            aligned_rels += 1
    transfer_dict['init_rel'] = tgt_re

    tgt_runner.model.load_state_dict(transfer_dict, strict=False)
    print(f'  Vocab-aligned transfer: {aligned_classes}/{len(tgt_class2id)} classes, '
          f'{aligned_rels}/{len(tgt_rel2id)} relations, '
          f'{len(transfer_dict)} tensors total')
    return tgt_runner


def load_folds(cls, path, gpu, dataset_override=None):
    """One runner per fold if path is a .folds list, else a single-element list."""
    ckpts = open(path).read().strip().splitlines() if path.endswith('.folds') else [path]
    return [load_runner(cls, c, gpu, dataset_override) for c in ckpts]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exist_load', required=True, help='Stage-1 checkpoint or .folds list')
    ap.add_argument('--rel_load',   required=True, help='Stage-2 checkpoint or .folds list')
    ap.add_argument('--gpu', default='-1')
    ap.add_argument('--dataset', default=None,
                    help='Override checkpoint dataset (for transfer learning evaluation)')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Existence decision threshold (default: the value baked into the Stage-1 model)')
    ap.add_argument('--neg_ratio', type=float, default=10.0)
    ap.add_argument('--split', default='test')
    ap.add_argument('--target_rels', nargs='+', default=None,
                    help='Filter evaluation to only these relations (e.g., brick:haspoint brick:haspart brick:feeds)')
    args = ap.parse_args()

    set_gpu(args.gpu)
    exist_models = load_folds(LinkRunner, args.exist_load, args.gpu, args.dataset)
    rel_models   = load_folds(Runner,     args.rel_load,   args.gpu, args.dataset)
    if len(exist_models) != len(rel_models):
        raise ValueError(f'Stage-1 has {len(exist_models)} folds but Stage-2 has {len(rel_models)}; '
                         'they must match to pair fold-i with fold-i.')
    ref   = exist_models[0]
    thr   = args.threshold if args.threshold is not None else getattr(ref.p, 'threshold', 0.6)
    split = args.split
    target_rels = ref.target_rels

    # Filter to user-specified relations if provided (for transfer learning: ignore unknown relations)
    if args.target_rels:
        # Map relation names to indices
        rel_filter = set()
        for rel_name in args.target_rels:
            for rid, rname in ref.id2rel.items():
                if rname.lower() == rel_name.lower():
                    rel_filter.add(rid)
                    break
        target_rels = [r for r in target_rels if r in rel_filter]
        print(f"Filtering to relations: {[ref.id2rel[r] for r in target_rels]}")

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

    # Build mask: keep all negatives + positives whose relation is a target relation.
    # (true_rel == -1 for negatives, so ~is_pos catches them; the second clause
    # drops positives whose relation is not in the target set.)
    target_mask = None
    if args.target_rels:
        target_set = set(target_rels)
        target_mask = (~is_pos) | np.array([r in target_set for r in true_rel])

    st = eset['stats']
    print(f"\n## Split '{split}': pos {st['n_pos']}, neg {st['n_neg']}, "
          f"base-rate {st['n_pos'] / (st['n_pos'] + st['n_neg']):.3f}, "
          f"{len(exist_models)} fold(s), existence thr {thr:.2f}")

    fold_results = []
    for i, (em, rm) in enumerate(zip(exist_models, rel_models)):
        score       = em.score_existence(eset)                       # existence prob per candidate
        pred_rel    = rm.score_relations_for_pairs(split, pairs)     # argmax relation per candidate
        ranks, rrel = rm.relation_ranks(split)                       # Stage-2 alone, over true edges

        # Filter to target relations if mask provided
        eval_is_pos, eval_true_rel, eval_score, eval_pred_rel = is_pos, true_rel, score, pred_rel
        eval_target_rels = target_rels
        if target_mask is not None:
            eval_is_pos = is_pos[target_mask]
            eval_true_rel = true_rel[target_mask]
            eval_score = score[target_mask]
            eval_pred_rel = pred_rel[target_mask]

        n_params    = getattr(em, 'n_params', 0) + getattr(rm, 'n_params', 0)
        train_time  = getattr(em, 'train_time', 0.0) + getattr(rm, 'train_time', 0.0)
        fold_results.append(M.evaluate_fold(
            eval_score, eval_is_pos, eval_true_rel, eval_pred_rel, ranks, rrel, eval_target_rels, thr,
            n_params=n_params, train_time=train_time))
        print(f"  scored fold {i + 1}/{len(exist_models)}")

    agg = M.aggregate(fold_results)
    print(M.format_report(agg, target_rels, ref.id2rel, thr,
                          title='CompGCN two-stage (existence: type+names | relation: type)',
                          n_folds=len(fold_results)))


if __name__ == '__main__':
    main()
