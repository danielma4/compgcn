"""Presence + relation evaluation — the real deployment task. EVAL ONLY.

Loads a trained checkpoint (no training) and, for each candidate node pair
(s, o), asks BOTH (a) is there a relation? and (b) which one? The model's
per-triple probability sigmoid(decode(s,r,o)) is the presence score; presence =
max over target relations, predicted relation = argmax. Thresholding it exposes
false positives / negatives.

Candidates = held-out true edges + sampled type-compatible non-edges (hard,
data-derived gate). Threshold tuned on validation to maximise F-beta (beta<1
favours precision, since a spurious edge is worse than a missed one). Baseline
``accept-all`` accepts every type-compatible pair (recall-maximal, precision-poor);
the model must beat its precision.

Usage:
    python eval_presence.py --config config/brick.yaml \
        --load 'checkpoints/<run>' --gpu -1 --neg_ratio 10 --beta 0.5
"""

import os
import random

from helper import set_gpu
from run import build_parser, parse_with_yaml, Runner
from presence import (build_ent2classes, build_signatures, known_pairs,
                      build_eval_set, score_eval_set, metrics_at, best_threshold, auprc)


def main():
    parser = build_parser()
    parser.add_argument('--load', '-load', dest='load_path', default=None, help='Checkpoint path')
    parser.add_argument('--neg_ratio', '-neg_ratio', type=float, default=10.0,
                        help='Type-compatible negatives sampled per positive')
    args = parse_with_yaml(parser)   # --beta comes from the shared parser (default 0.5, precision-favouring)

    set_gpu(args.gpu)
    runner = Runner(args)
    load_path = args.load_path or os.path.join('./checkpoints', args.name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path} (pass --load <path>)")
    runner.load_model(load_path)

    rng = random.Random(runner.p.seed)
    ent2classes      = build_ent2classes(runner)
    allowed, sig_rel = build_signatures(runner, ent2classes)
    known            = known_pairs(runner)

    esets, scores = {}, {}
    for sp in ['valid', 'test']:
        esets[sp]  = build_eval_set(runner, sp, ent2classes, allowed, sig_rel, known, args.neg_ratio, rng)
        scores[sp] = score_eval_set(runner, esets[sp])

    # Tune threshold on validation (maximise F-beta).
    v_pres, _ = scores['valid']
    thr = best_threshold(v_pres, esets['valid']['is_pos'], args.beta)

    print(f"\n## Presence + relation eval  (neg_ratio={args.neg_ratio}, beta={args.beta}, "
          f"threshold tuned on valid = {thr:.4f})")
    print(f"## Checkpoint: {load_path}\n")

    for sp in ['valid', 'test']:
        eset = esets[sp]; pres, pred = scores[sp]
        is_pos, true_r, base_r, st = eset['is_pos'], eset['true_r'], eset['base_r'], eset['stats']
        covered = st['n_cov'] / st['n_pos'] if st['n_pos'] else 0.0
        print(f"=== {sp}  (pos={st['n_pos']}, type-covered={st['n_cov']} "
              f"[{covered:.0%}, scored too], uncovered={st['n_unc']}, neg={st['n_neg']}) ===")

        ap      = auprc(pres, is_pos)
        acc_all = metrics_at(pres, is_pos, true_r, base_r, thr=0.0, beta=args.beta)   # accept every scored candidate
        mdl     = metrics_at(pres, is_pos, true_r, pred,   thr,    beta=args.beta)
        print(f"  accept-all : P {acc_all['precision']:.3f}  R {acc_all['recall']:.3f}  "
              f"F1 {acc_all['f1']:.3f}  F{args.beta:g} {acc_all['fbeta']:.3f}  rel-acc {acc_all['rel_acc']:.3f}")
        print(f"  model      : P {mdl['precision']:.3f}  R {mdl['recall']:.3f}  "
              f"F1 {mdl['f1']:.3f}  F{args.beta:g} {mdl['fbeta']:.3f}  rel-acc {mdl['rel_acc']:.3f}  "
              f"(AUPRC {ap:.3f}; TP {mdl['tp']} FP {mdl['fp']} FN {mdl['fn']})")
        print()


if __name__ == '__main__':
    main()
