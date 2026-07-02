"""Sweep composition functions on the 40% holdout dataset (brick_mortar).

Stage 1 (link existence): sweeps opn in [sub, mult, corr] — score_func doesn't
apply (Stage 1 always uses a binary BCE head).

Stage 2 (relation prediction): sweeps opn × score_func (9 combos). Near-perfect
results expected; included for completeness.

Each combo saves to checkpoints/ablation_h40/<stage>/<opn>[_<score>]/.

Usage:
    # Stage 1 only (the real bottleneck)
    python scripts/ablation_h40.py --config config/link_exist.yaml --stage1 --gpu 0

    # Stage 2 only
    python scripts/ablation_h40.py --config config/brick.yaml --stage2 --gpu 0

    # Both (runs Stage 1 then Stage 2)
    python scripts/ablation_h40.py --config config/brick.yaml --gpu 0
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from helper import set_gpu
from run import build_parser, parse_with_yaml, Runner
from link_existence import LinkRunner

OPN        = ['sub', 'mult', 'corr']
SCORE_FUNC = ['transe', 'distmult', 'conve']
GCN_LAYERS = [2, 3, 4]


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def run_stage1(base_args, opn, gcn_layer):
    p = copy.deepcopy(base_args)
    p.opn       = opn
    p.gcn_layer = gcn_layer
    p.cv_alloc  = 1
    p.cv_seeds  = 1
    p.name      = f'ablation_h40/stage1/{opn}_l{gcn_layer}/link_exist_mortar'

    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    runner     = LinkRunner(p)
    runner.bce = torch.nn.BCEWithLogitsLoss()
    runner.fit_existence(p.neg_ratio, p.beta)

    val  = runner.eval_existence('valid', p.neg_ratio, p.beta, p.threshold)
    test = runner.eval_existence('test',  p.neg_ratio, p.beta, p.threshold)
    return {
        'opn':        opn,
        'gcn_layer':  gcn_layer,
        'val_auprc':  val['auprc'],
        'val_f1':     val['f1'],
        'val_prec':   val['precision'],
        'val_rec':    val['recall'],
        'test_auprc': test['auprc'],
        'test_f1':    test['f1'],
        'test_prec':  test['precision'],
        'test_rec':   test['recall'],
    }


def sweep_stage1(args):
    combos  = [(o, l) for o in OPN for l in GCN_LAYERS]
    results = []
    for i, (opn, gcn_layer) in enumerate(combos, 1):
        print(f'\n{"="*60}\n[Stage 1  {i}/{len(combos)}] opn={opn}  gcn_layer={gcn_layer}\n{"="*60}')
        try:
            r = run_stage1(args, opn, gcn_layer)
            results.append(r)
            print(f'  val  AUPRC={r["val_auprc"]:.4f}  F1={r["val_f1"]:.4f}  '
                  f'P={r["val_prec"]:.4f}  R={r["val_rec"]:.4f}')
            print(f'  test AUPRC={r["test_auprc"]:.4f}  F1={r["test_f1"]:.4f}  '
                  f'P={r["test_prec"]:.4f}  R={r["test_rec"]:.4f}')
        except Exception as exc:
            print(f'  FAILED: {exc}')
            nan = float('nan')
            results.append({'opn': opn, 'gcn_layer': gcn_layer,
                            'val_auprc': nan, 'val_f1': nan,
                            'val_prec': nan, 'val_rec': nan,
                            'test_auprc': nan, 'test_f1': nan,
                            'test_prec': nan, 'test_rec': nan})

    print('\n\n' + '='*76)
    print('STAGE 1 ABLATION  opn × gcn_layer  (brick_mortar, 40% holdout, sorted by test AUPRC)')
    print('='*76)
    print(f'{"opn":<6} {"layers":>7} {"val AUPRC":>10} {"val F1":>8} {"val P":>7} {"val R":>7} '
          f'{"tst AUPRC":>10} {"tst F1":>8} {"tst P":>7} {"tst R":>7}')
    print('-'*76)
    for r in sorted(results, key=lambda x: -x['test_auprc']):
        print(f'{r["opn"]:<6} {r["gcn_layer"]:>7} '
              f'{r["val_auprc"]:>10.4f} {r["val_f1"]:>8.4f} '
              f'{r["val_prec"]:>7.4f} {r["val_rec"]:>7.4f} '
              f'{r["test_auprc"]:>10.4f} {r["test_f1"]:>8.4f} '
              f'{r["test_prec"]:>7.4f} {r["test_rec"]:>7.4f}')
    best = max(results, key=lambda x: x['test_auprc'])
    base = next((r for r in results if r['opn'] == 'corr' and r['gcn_layer'] == 2), None)
    print(f'\nBaseline (corr, 2 layers): test AUPRC={base["test_auprc"]:.4f}' if base else '')
    print(f'Best     ({best["opn"]}, {best["gcn_layer"]} layers): test AUPRC={best["test_auprc"]:.4f}')
    return results


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def run_stage2(base_args, opn, score_func):
    p = copy.deepcopy(base_args)
    p.opn        = opn
    p.score_func = score_func
    p.cv_alloc   = 1
    p.cv_seeds   = 1
    p.name       = f'ablation_h40/stage2/{opn}_{score_func}/rel_pred_mortar'

    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    runner = Runner(p)
    runner.fit()
    test = runner.predict_relation('test')
    val  = runner.predict_relation('valid')
    return {
        'opn':        opn,
        'score_func': score_func,
        'val_mrr':    val['mrr'],
        'val_h1':     val['hits@1'],
        'test_mrr':   test['mrr'],
        'test_h1':    test['hits@1'],
    }


def sweep_stage2(args):
    combos  = [(o, s) for o in OPN for s in SCORE_FUNC]
    results = []
    for i, (opn, score_func) in enumerate(combos, 1):
        print(f'\n{"="*60}\n[Stage 2  {i}/{len(combos)}] opn={opn}  score_func={score_func}\n{"="*60}')
        try:
            r = run_stage2(args, opn, score_func)
            results.append(r)
            print(f'  val  MRR={r["val_mrr"]:.4f}  H@1={r["val_h1"]:.4f}')
            print(f'  test MRR={r["test_mrr"]:.4f}  H@1={r["test_h1"]:.4f}')
        except Exception as exc:
            print(f'  FAILED: {exc}')
            nan = float('nan')
            results.append({'opn': opn, 'score_func': score_func,
                            'val_mrr': nan, 'val_h1': nan,
                            'test_mrr': nan, 'test_h1': nan})

    print('\n\n' + '='*62)
    print('STAGE 2 ABLATION  (brick_mortar, 40% holdout, sorted by test MRR)')
    print('='*62)
    print(f'{"opn":<8} {"score_func":<12} {"val MRR":>8} {"val H@1":>8} {"test MRR":>9} {"test H@1":>9}')
    print('-'*62)
    for r in sorted(results, key=lambda x: -x['test_mrr']):
        print(f'{r["opn"]:<8} {r["score_func"]:<12} '
              f'{r["val_mrr"]:>8.4f} {r["val_h1"]:>8.4f} '
              f'{r["test_mrr"]:>9.4f} {r["test_h1"]:>9.4f}')
    best = max(results, key=lambda x: x['test_mrr'])
    base = next((r for r in results if r['opn'] == 'corr' and r['score_func'] == 'distmult'), None)
    print(f'\nBaseline (corr + distmult): test MRR={base["test_mrr"]:.4f}' if base else '')
    print(f'Best     ({best["opn"]} + {best["score_func"]}): test MRR={best["test_mrr"]:.4f}')
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    parser.add_argument('--stage1', action='store_true', help='Only sweep Stage 1 (link existence).')
    parser.add_argument('--stage2', action='store_true', help='Only sweep Stage 2 (relation prediction).')
    # link_existence-specific args (ignored for Stage 2)
    parser.add_argument('--neg_ratio',  type=float, default=10.0)
    parser.add_argument('--threshold',  type=float, default=0.6)
    parser.add_argument('--name_feats', action='store_true')
    args = parse_with_yaml(parser)
    set_gpu(args.gpu)

    run1 = args.stage1 or (not args.stage1 and not args.stage2)
    run2 = args.stage2 or (not args.stage1 and not args.stage2)

    if run1:
        sweep_stage1(args)
    if run2:
        sweep_stage2(args)


if __name__ == '__main__':
    main()
