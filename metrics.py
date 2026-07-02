"""Metrics for the two-stage edge pipeline (existence -> relation), pooled over all
test-building instances (micro is primary; macro reported as reference).

One fold = one (existence model, relation model) pair evaluated on the fixed test
candidate set. ``evaluate_fold`` turns per-candidate arrays into a structured result;
``aggregate`` reduces a list of folds to mean/std; ``format_report`` renders the table.

Views (each with a per-relation breakdown for hasPoint/hasPart/feeds):
  Stage 1 alone   : existence — confusion matrix, P/R/F1, AUPRC, AUROC
  Stage 2 alone   : relation (oracle existence) — MRR, Hits@1
  Combined        : end-to-end — P/R/F1
  Overall         : macro & micro P/R/F1, accuracy, AUPRC, AUROC (combined pipeline)
"""

import math

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

NO_EDGE = -1


def _binary(y_true, y_pred):
    """tp/fp/fn/tn + P/R/F1 for a binary problem (positive label = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average='binary', zero_division=0)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(p), recall=float(r), f1=float(f1))


def _auprc(y_true, score):
    y_true = np.asarray(y_true, int)
    return float(average_precision_score(y_true, score)) if 0 < y_true.sum() < len(y_true) else float('nan')


def _auroc(y_true, score):
    y_true = np.asarray(y_true, int)
    return float(roc_auc_score(y_true, score)) if 0 < y_true.sum() < len(y_true) else float('nan')


def _existence(score, is_pos, thr):
    """Stage-1 binary metrics over a candidate set."""
    score = np.asarray(score, float)
    y_true = np.asarray(is_pos, int)
    out = _binary(y_true, (score >= thr).astype(int))
    out['auprc'] = _auprc(y_true, score)
    out['auroc'] = _auroc(y_true, score)
    return out


def _relation(ranks):
    """Stage-2 ranking metrics. ranks = 1-indexed rank of the true relation."""
    ranks = np.asarray(ranks, float)
    if not len(ranks):
        return dict(mrr=float('nan'), hits1=float('nan'), n=0)
    return dict(mrr=float((1.0 / ranks).mean()), hits1=float((ranks <= 1).mean()), n=int(len(ranks)))


def evaluate_fold(score, is_pos, true_rel, pred_rel, ranks, rank_rel,
                  target_rels, thr, n_params=None, train_time=None):
    """Structured metrics for one fold.

    score     : (N,) existence probability per candidate
    is_pos    : (N,) bool, candidate is a true edge
    true_rel  : (N,) true relation id (NO_EDGE for negatives)
    pred_rel  : (N,) Stage-2 argmax relation id for every candidate
    ranks     : (M,) 1-indexed rank of the true relation, one per true edge (oracle existence)
    rank_rel  : (M,) the true relation id for each entry of `ranks`
    """
    score = np.asarray(score, float); is_pos = np.asarray(is_pos, bool)
    true_rel = np.asarray(true_rel); pred_rel = np.asarray(pred_rel)
    ranks = np.asarray(ranks, float); rank_rel = np.asarray(rank_rel)
    accepted = score >= thr

    # ---- Stage 1 (existence) ----
    s1 = {'overall': _existence(score, is_pos, thr), 'per_relation': {}}
    for r in target_rels:                                   # positives-of-r vs all negatives
        keep = (~is_pos) | (is_pos & (true_rel == r))
        s1['per_relation'][r] = _existence(score[keep], is_pos[keep], thr)

    # ---- Stage 2 (relation, oracle existence) ----
    s2 = {'overall': _relation(ranks), 'per_relation': {}}
    for r in target_rels:
        s2['per_relation'][r] = _relation(ranks[rank_rel == r])

    # ---- multiclass view (NO_EDGE = rejected / non-edge) ----
    pred_cls = np.where(accepted, pred_rel, NO_EDGE)
    true_cls = np.where(is_pos, true_rel, NO_EDGE)
    p, r, f1, _ = precision_recall_fscore_support(
        true_cls, pred_cls, labels=target_rels, average=None, zero_division=0)
    cmb_per = {}
    for i, rel in enumerate(target_rels):
        tp = int(((pred_cls == rel) & (true_cls == rel)).sum())
        fp = int(((pred_cls == rel) & (true_cls != rel)).sum())
        fn = int(((pred_cls != rel) & (true_cls == rel)).sum())
        tn = int(((pred_cls != rel) & (true_cls != rel)).sum())
        cmb_per[rel] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                            precision=float(p[i]), recall=float(r[i]), f1=float(f1[i]))

    # ---- Combined (end-to-end existence+relation correct) ----
    correct = is_pos & (pred_rel == true_rel)
    ctp = int((accepted & correct).sum())
    cfp = int((accepted & ~correct).sum())
    cfn = int((is_pos & ~(accepted & correct)).sum())
    cp = ctp / (ctp + cfp) if ctp + cfp else 0.0
    cr = ctp / (ctp + cfn) if ctp + cfn else 0.0
    combined = {'overall': dict(tp=ctp, fp=cfp, fn=cfn, precision=cp, recall=cr,
                                f1=(2 * cp * cr / (cp + cr) if cp + cr else 0.0)),
                'per_relation': cmb_per}

    # ---- Overall (macro/micro over relations, accuracy over all candidates) ----
    maP, maR, maF, _ = precision_recall_fscore_support(
        true_cls, pred_cls, labels=target_rels, average='macro', zero_division=0)
    miP, miR, miF, _ = precision_recall_fscore_support(
        true_cls, pred_cls, labels=target_rels, average='micro', zero_division=0)
    overall = dict(macro_precision=float(maP), macro_recall=float(maR), macro_f1=float(maF),
                   micro_precision=float(miP), micro_recall=float(miR), micro_f1=float(miF),
                   accuracy=float(accuracy_score(true_cls, pred_cls)),
                   auprc=_auprc(is_pos.astype(int), score), auroc=_auroc(is_pos.astype(int), score))

    return dict(stage1=s1, stage2=s2, combined=combined, overall=overall,
                price=dict(n_params=float(n_params or 0), train_time=float(train_time or 0)))


def aggregate(folds):
    """Reduce a list of identically-structured fold dicts to (mean, std) at every
    numeric leaf, ignoring NaNs."""
    def rec(vals):
        if isinstance(vals[0], dict):
            return {k: rec([v[k] for v in vals]) for k in vals[0]}
        arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
        return (float(arr.mean()), float(arr.std())) if len(arr) else (float('nan'), 0.0)
    return rec(folds)


def _ms(t, nd=3):
    """Format a (mean, std) tuple as 'mean±std'."""
    m, s = t
    if isinstance(m, float) and math.isnan(m):
        return 'n/a'
    return f"{m:.{nd}f}±{s:.{nd}f}"


def format_report(agg, target_rels, id2rel, thr, title='CompGCN two-stage', n_folds=None):
    """Render the aggregated mean±std table."""
    rn = lambda r: id2rel.get(r, str(r)) if id2rel else str(r)
    L = [f"\n===== {title}  (existence thr={thr:.2f}" + (f", {n_folds} folds" if n_folds else "") + ") ====="]

    L.append("\nPer-relation:")
    for r in target_rels:
        s2 = agg['stage2']['per_relation'][r]
        cm = agg['combined']['per_relation'][r]
        L.append(f"  {rn(r)}:")
        L.append(f"    Stage2 (relation)   MRR {_ms(s2['mrr'])}  Hits@1 {_ms(s2['hits1'])}")
        L.append(f"    Combined            P {_ms(cm['precision'])}  R {_ms(cm['recall'])}  F1 {_ms(cm['f1'])}")

    s1o, s2o, co, ov = (agg['stage1']['overall'], agg['stage2']['overall'],
                        agg['combined']['overall'], agg['overall'])
    L.append("\nStage 1 alone (existence, pooled):")
    L.append(f"  P {_ms(s1o['precision'])}  R {_ms(s1o['recall'])}  F1 {_ms(s1o['f1'])}")
    L.append("Stage 2 alone (relation, oracle existence):")
    L.append(f"  MRR {_ms(s2o['mrr'])}  Hits@1 {_ms(s2o['hits1'])}")
    L.append("Combined pipeline (end-to-end):")
    L.append(f"  P {_ms(co['precision'])}  R {_ms(co['recall'])}  F1 {_ms(co['f1'])}")

    L.append("\nOverall (combined Stage 1 & 2):")
    L.append(f"  Micro   P {_ms(ov['micro_precision'])}  R {_ms(ov['micro_recall'])}  F1 {_ms(ov['micro_f1'])}   <- primary")
    L.append(f"  Macro   P {_ms(ov['macro_precision'])}  R {_ms(ov['macro_recall'])}  F1 {_ms(ov['macro_f1'])}")
    L.append(f"  Accuracy {_ms(ov['accuracy'])}   AUPRC {_ms(ov['auprc'])}   AUROC {_ms(ov['auroc'])}")
    price = agg['price']
    L.append(f"  Params {price['n_params'][0]:,.0f}   Total train time {_ms(price['train_time'],1)}s")
    return "\n".join(L)
