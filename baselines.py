"""Baselines for both pipeline stages.

Two modes controlled by --mode:
  relation  (default) — type/majority scorers for Stage 2; directly comparable
                        to run.py's MRR/Hits@1.
  existence           — lightweight classifiers for Stage 1; directly comparable
                        to link_existence.py's AUPRC/P/R. Tests whether the GNN
                        is earning its keep or name features alone suffice.

Usage:
    python baselines.py --config config/brick.yaml
    python baselines.py --config config/link_exist.yaml --mode existence
"""

import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from link_existence import LinkRunner
from name_overlap_check import tokens, ngrams, jaccard, feats as name_feats_uri
from presence import build_ent2classes, build_signatures, known_pairs, build_eval_set
from run import build_parser, parse_with_yaml, Runner

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Shared feature extraction
# ---------------------------------------------------------------------------

def _name_cache(id2ent, n_ent):
    tok = [tokens(id2ent[i]) for i in range(n_ent)]
    ng  = [ngrams(id2ent[i]) for i in range(n_ent)]
    return tok, ng


def name_feats(subs, objs, tok, ng):
    """(N, 3) array: tok_jac, id_shared, char3_jac."""
    rows = []
    for s, o in zip(subs, objs):
        ts, to_ = tok[s], tok[o]
        nums = {t for t in ts if t.isdigit()} & {t for t in to_ if t.isdigit()}
        rows.append((jaccard(ts, to_), 1.0 if nums else 0.0, jaccard(ng[s], ng[o])))
    return np.array(rows, dtype=np.float32)


def type_feats(subs, objs, ent2classes, sig_rel, target_rels):
    """(N, len(target_rels)) — count of train edges with same (cs,co) sig per relation."""
    C = len(target_rels)
    rel2c = {r: c for c, r in enumerate(target_rels)}
    out = np.zeros((len(subs), C), dtype=np.float32)
    for i, (s, o) in enumerate(zip(subs, objs)):
        for cs in (ent2classes.get(s) or {-1}):
            for co in (ent2classes.get(o) or {-1}):
                for r, cnt in sig_rel.get((cs, co), {}).items():
                    if r in rel2c:
                        out[i, rel2c[r]] += cnt
    return out


def build_features(pairs, tok, ng, ent2classes, sig_rel, target_rels):
    subs = [p[0] for p in pairs]
    objs = [p[1] for p in pairs]
    nf = name_feats(subs, objs, tok, ng)
    tf = type_feats(subs, objs, ent2classes, sig_rel, target_rels)
    return np.concatenate([nf, tf], axis=1)


def build_train_set(runner, ent2classes, allowed, known, neg_ratio, rng):
    """True train edges (target rels) + within-building type-compatible negatives."""
    target = set(runner.target_rels)
    positives = [(s, o) for s, r, o in runner.graph_data['train'] if r in target]

    bof = lambda e: runner.id2ent[e].split(':', 1)[0]
    by_bc = defaultdict(list)
    for s, r, o in runner.graph_data['train']:
        for e in (s, o):
            b = bof(e)
            for c in (ent2classes.get(e) or {-1}):
                by_bc[(b, c)].append(e)
    sigs_by_b = {}
    for b in {b for (b, _) in by_bc}:
        sigs_by_b[b] = [sig for sig in allowed if by_bc.get((b, sig[0])) and by_bc.get((b, sig[1]))]

    n_per = max(1, int(round(neg_ratio)))
    negatives, seen = [], set(known)
    for (s, o) in positives:
        b = bof(s)
        sigs = sigs_by_b.get(b)
        if not sigs:
            continue
        made, attempts = 0, 0
        while made < n_per and attempts < n_per * 50:
            attempts += 1
            cs, co = sigs[rng.randrange(len(sigs))]
            ns = by_bc[(b, cs)][rng.randrange(len(by_bc[(b, cs)]))]
            no = by_bc[(b, co)][rng.randrange(len(by_bc[(b, co)]))]
            if ns == no or (ns, no) in seen:
                continue
            seen.add((ns, no)); negatives.append((ns, no)); made += 1

    pairs  = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=int)
    return pairs, labels


# ---------------------------------------------------------------------------
# Existence baselines
# ---------------------------------------------------------------------------

def _prf(tp, fp, fn):
    p  = tp / (tp + fp) if tp + fp else 0.0
    r  = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def existence_metrics(scores, labels, thr=0.5):
    labels = np.asarray(labels, int)
    acc = scores >= thr
    tp = int(( acc & (labels == 1)).sum())
    fp = int(( acc & (labels == 0)).sum())
    fn = int((~acc & (labels == 1)).sum())
    p, r, f1 = _prf(tp, fp, fn)
    auprc = float(average_precision_score(labels, scores))
    auroc = float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else float('nan')
    return dict(auprc=auprc, auroc=auroc, precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)


def rule_scorer(pairs, tok, ng):
    """Accept if any name token overlaps — no training, zero parameters."""
    scores = []
    for s, o in pairs:
        ts, to_ = tok[s], tok[o]
        nums_s = {t for t in ts if t.isdigit()}
        nums_o = {t for t in to_ if t.isdigit()}
        # score = tok_jac + 2*id_shared + char3_jac, normalised to [0,1]
        score = (jaccard(ts, to_) + 2.0 * (1.0 if nums_s & nums_o else 0.0) + jaccard(ng[s], ng[o])) / 4.0
        scores.append(score)
    return np.array(scores, dtype=np.float32)


def run_existence_baselines(args):
    runner = LinkRunner(args)
    rng = random.Random(args.seed)

    e2c     = build_ent2classes(runner)
    allowed, sig_rel = build_signatures(runner, e2c)
    known   = known_pairs(runner)
    tok, ng = _name_cache(runner.id2ent, runner.p.num_ent)

    # Build train set and eval set (same candidate construction as the GNN)
    train_pairs, train_y = build_train_set(runner, e2c, allowed, known, neg_ratio=5, rng=rng)
    eset = build_eval_set(runner, 'test', e2c, allowed, sig_rel, known, neg_ratio=10, rng=rng)
    test_pairs = ([(s, o) for s, o, _ in eset['cov']] +
                  [(s, o) for s, o, _ in eset['unc']] +
                  [(s, o) for s, o, _ in eset['neg']])
    test_y = eset['is_pos'].astype(int)

    X_train = build_features(train_pairs, tok, ng, e2c, sig_rel, runner.target_rels)
    X_test  = build_features(test_pairs,  tok, ng, e2c, sig_rel, runner.target_rels)

    st = eset['stats']
    base_rate = st['n_pos'] / (st['n_pos'] + st['n_neg'])
    print(f"\n## Existence baselines — test split"
          f"  (pos={st['n_pos']}, neg={st['n_neg']}, base-rate={base_rate:.3f})\n")

    models = {}

    # Rule-based: name overlap only, no training
    rule_scores = rule_scorer(test_pairs, tok, ng)
    models['rule(names)'] = rule_scores

    # Logistic regression: name features only
    lr_name = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    lr_name.fit(X_train[:, :3], train_y)
    models['logreg(names)'] = lr_name.predict_proba(X_test[:, :3])[:, 1]

    # Logistic regression: name + type features
    lr_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    lr_full.fit(X_train, train_y)
    models['logreg(names+type)'] = lr_full.predict_proba(X_test)[:, 1]

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            use_label_encoder=False, eval_metric='logloss',
                            n_jobs=4, verbosity=0, random_state=args.seed)
        xgb.fit(X_train, train_y)
        models['xgb(names+type)'] = xgb.predict_proba(X_test)[:, 1]
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                        random_state=args.seed)
        gb.fit(X_train, train_y)
        models['gbm(names+type)'] = gb.predict_proba(X_test)[:, 1]

    hdr = f"{'model':<22} {'AUPRC':>7} {'AUROC':>7} {'P@.75':>7} {'R@.75':>7} {'F1@.75':>7}"
    print(hdr); print('-' * len(hdr))
    for name, scores in models.items():
        m = existence_metrics(scores, test_y, thr=0.5)
        m75 = existence_metrics(scores, test_y, thr=0.75)
        print(f"{name:<22} {m['auprc']:>7.3f} {m['auroc']:>7.3f} "
              f"{m75['precision']:>7.3f} {m75['recall']:>7.3f} {m75['f1']:>7.3f}")
    print(f"\n(GNN existence baseline: AUPRC ~0.938  P@.75 ~0.899  R@.75 ~0.781)")


# ---------------------------------------------------------------------------
# Relation baselines (unchanged)
# ---------------------------------------------------------------------------

def object_type_scorer(runner):
    num_rel = runner.p.num_rel
    ent2classes = build_ent2classes(runner)
    counts = defaultdict(lambda: np.zeros(num_rel))
    for s, r, o in runner.graph_data['train']:
        if r >= num_rel:
            continue
        for co in (ent2classes.get(o) or {-1}):
            counts[co][r] += 1

    def score_matrix(subs, objs):
        out = np.zeros((len(subs), num_rel))
        for i, o in enumerate(objs):
            for co in (ent2classes.get(o) or {-1}):
                sig = counts.get(co)
                if sig is not None:
                    out[i] += sig
        return torch.tensor(out, dtype=torch.float)
    return score_matrix


def majority_scorer(runner):
    num_rel = runner.p.num_rel
    freq = np.zeros(num_rel)
    for _, r, _ in runner.graph_data['train']:
        if r < num_rel:
            freq[r] += 1
    freq_t = torch.tensor(freq, dtype=torch.float)

    def score_matrix(subs, objs):
        return freq_t.unsqueeze(0).expand(len(subs), -1).clone()
    return score_matrix


def type_signature_scorer(runner):
    num_rel = runner.p.num_rel
    ent2classes = build_ent2classes(runner)
    counts = defaultdict(lambda: np.zeros(num_rel))
    for s, r, o in runner.graph_data['train']:
        if r >= num_rel:
            continue
        for cs in (ent2classes.get(s) or {-1}):
            for co in (ent2classes.get(o) or {-1}):
                counts[(cs, co)][r] += 1

    def score_matrix(subs, objs):
        out = np.zeros((len(subs), num_rel))
        for i, (s, o) in enumerate(zip(subs, objs)):
            for cs in (ent2classes.get(s) or {-1}):
                for co in (ent2classes.get(o) or {-1}):
                    sig = counts.get((cs, co))
                    if sig is not None:
                        out[i] += sig
        return torch.tensor(out, dtype=torch.float)
    return score_matrix


def evaluate_relation(runner, split, score_matrix, sample=None):
    triples = runner.data[split]
    if sample is not None and len(triples) > sample:
        rng = random.Random(runner.p.seed)
        triples = [triples[i] for i in rng.sample(range(len(triples)), sample)]

    cand    = runner.target_rels
    rel2col = {r: c for c, r in enumerate(cand)}
    C       = len(cand)

    so2r = defaultdict(set)
    for sp in ['train', 'valid', 'test']:
        for s, r, o in runner.graph_data[sp] + runner.data[sp]:
            if r in rel2col:
                so2r[(s, o)].add(r)

    subs = [t[0] for t in triples]
    objs = [t[2] for t in triples]
    tgt  = torch.tensor([rel2col[t[1]] for t in triples], dtype=torch.long)
    b    = torch.arange(len(triples))

    full   = score_matrix(subs, objs)
    scores = full[:, cand].clone()
    target = scores[b, tgt].clone()
    for i, t in enumerate(triples):
        for rr in so2r[(t[0], t[2])]:
            scores[i, rel2col[rr]] = -1e9
    scores[b, tgt] = target

    ranks = 1 + torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)[b, tgt]
    ranks = ranks.float()
    out = {'mrr': round((1.0 / ranks).mean().item(), 5),
           'hits@1': round((ranks <= 1).float().mean().item(), 5),
           'count': len(triples)}
    for k in (3, 5):
        if k < C:
            out['hits@{}'.format(k)] = round((ranks <= k).float().mean().item(), 5)
    return out


def run_relation_baselines(args):
    runner = Runner(args)
    split  = args.baseline_split

    print(f"\n## Relation-prediction baselines on '{split}' "
          f"(ranking among {len(runner.target_rels)} target relations: "
          f"{[runner.id2rel[t] for t in runner.target_rels]})\n")
    scorers = {
        'majority': majority_scorer(runner),
        'obj_type': object_type_scorer(runner),
        'type':     type_signature_scorer(runner),
    }
    hdr = f"{'baseline':<10} {'MRR':>8} {'H@1':>8}"
    print(hdr); print('-' * len(hdr))
    for name, sc in scorers.items():
        m = evaluate_relation(runner, split, sc)
        print(f"{name:<10} {m['mrr']:>8.4f} {m['hits@1']:>8.4f}")
    print(f"\n(n={m['count']} triples; same filtered metric as the model's rel-MRR)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    parser.add_argument('--baseline_split', default='valid', choices=['valid', 'test'])
    parser.add_argument('--mode', default='relation', choices=['relation', 'existence'],
                        help='relation: Stage-2 MRR baselines; existence: Stage-1 AUPRC baselines')
    # existence-specific keys so link_exist.yaml parses cleanly
    parser.add_argument('--name_feats', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--neg_ratio', type=float, default=10.0)
    args = parse_with_yaml(parser)
    args.gpu = '-1'

    if args.mode == 'existence':
        run_existence_baselines(args)
    else:
        run_relation_baselines(args)


if __name__ == '__main__':
    main()
