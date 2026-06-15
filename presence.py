"""Shared presence + relation evaluation logic (used by run.py for per-epoch
tracking and by eval_presence.py for the full report).

Presence score for a pair (s, o) = max over target relations of the per-triple
probability sigmoid(decode(s, r, o)); predicted relation = argmax. Candidates =
held-out true edges (positives) + sampled type-compatible non-edges (negatives).
Type-compatibility is a hard data-derived gate; positives with a signature
unseen in training are unreachable (automatic false negatives).
"""

import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve


def build_ent2classes(runner):
    """entity id -> set of Brick class ids (rdf:type objects), from the observed
    context graphs (rdf:type is always-known context)."""
    type_rel = runner.rel2id.get('rdf:type')
    if type_rel is None:
        raise ValueError("type-based eval needs rdf:type triples, but none are in the data.")
    ent2classes = defaultdict(set)
    for split in ['train', 'valid', 'test']:
        for s, r, o in runner.graph_data[split]:
            if r == type_rel:
                ent2classes[s].add(o)
    return ent2classes


def build_signatures(runner, ent2classes):
    """Allowed (cs, co) type signatures for target relations + per-signature
    relation counts, from the TRAIN context graph."""
    target = set(runner.target_rels)
    allowed = set()
    sig_rel = defaultdict(lambda: defaultdict(int))
    for s, r, o in runner.graph_data['train']:
        if r in target:
            for cs in (ent2classes.get(s) or {-1}):
                for co in (ent2classes.get(o) or {-1}):
                    allowed.add((cs, co))
                    sig_rel[(cs, co)][r] += 1
    return allowed, sig_rel


def known_pairs(runner):
    """Directed (s, o) pairs that are true target edges in any split."""
    target = set(runner.target_rels)
    known = set()
    for sp in ['train', 'valid', 'test']:
        for s, r, o in runner.graph_data[sp] + runner.data[sp]:
            if r in target:
                known.add((s, o))
    return known


def sig_compatible(s, o, ent2classes, allowed):
    for cs in (ent2classes.get(s) or {-1}):
        for co in (ent2classes.get(o) or {-1}):
            if (cs, co) in allowed:
                return True
    return False


def sig_majority_fn(ent2classes, sig_rel):
    def f(s, o):
        best_r, best_c = -1, -1
        for cs in (ent2classes.get(s) or {-1}):
            for co in (ent2classes.get(o) or {-1}):
                for r, c in sig_rel.get((cs, co), {}).items():
                    if c > best_c:
                        best_c, best_r = c, r
        return best_r
    return f


def _building_of(e):
    """Building id = the entity-URI prefix (stem before ':'), e.g. 'bldg6:AC03' -> 'bldg6'."""
    return e.split(':', 1)[0]


def build_candidates(runner, split, ent2classes, allowed, known, neg_ratio, rng):
    """(covered_pos, uncovered_pos, negatives); each item (s, o, true_rel|None).
    Negatives are sampled WITHIN the positive's building (type-compatible, not a
    known edge), so pooled multi-building splits don't get trivially-easy
    cross-building pairs."""
    target = set(runner.target_rels)
    positives = [(s, o, r) for s, r, o in runner.data[split] if r in target]
    covered   = [(s, o, r) for (s, o, r) in positives if sig_compatible(s, o, ent2classes, allowed)]
    uncovered = [(s, o, r) for (s, o, r) in positives if not sig_compatible(s, o, ent2classes, allowed)]

    # Entities grouped by (building, class) so a tail can be corrupted within the
    # positive's building. Entities are integer ids; resolve to URI for the building key.
    bof = lambda e: _building_of(runner.id2ent[e])
    by_bc = defaultdict(list)
    for s, r, o in runner.graph_data[split] + runner.data[split]:
        for e in (s, o):
            for c in (ent2classes.get(e) or {-1}):
                by_bc[(bof(e), c)].append(e)
    by_bc = {k: list(set(v)) for k, v in by_bc.items()}    # dedup the tail pools

    # Negatives = type-constrained tail corruption (Krompass 2015): keep the true
    # subject, replace the tail with another entity of the SAME Brick class as the
    # true object, in the same building, that is not itself a true edge.
    negatives, seen = [], set()
    n_per = max(1, int(round(neg_ratio)))
    for (s, o, _r) in positives:
        b = bof(s)
        pool = [e for c in (ent2classes.get(o) or {-1}) for e in by_bc.get((b, c), [])]
        if not pool:
            continue
        made, attempts = 0, 0
        while made < n_per and attempts < n_per * 50:
            attempts += 1
            no = pool[rng.randrange(len(pool))]
            if no == s or (s, no) in known or (s, no) in seen:
                continue
            seen.add((s, no)); negatives.append((s, no, None)); made += 1
    return covered, uncovered, negatives


def model_presence(runner, split, pairs):
    """presence prob (max over target relations) and predicted relation id, per pair."""
    if not pairs:
        return np.array([]), np.array([], dtype=int)
    runner.model.eval()
    graph = {'valid': (runner.edge_index_valid, runner.edge_type_valid),
             'test':  (runner.edge_index_test,  runner.edge_type_test)}.get(
                 split, (runner.edge_index, runner.edge_type))
    cand = runner.target_rels
    pres = np.zeros(len(pairs)); pred = np.zeros(len(pairs), dtype=int)
    with torch.no_grad():
        all_ent, all_rel = runner.model.encode(*graph)
        bs = runner.p.batch_size
        for st in range(0, len(pairs), bs):
            b       = pairs[st:st + bs]
            subs    = torch.tensor([p[0] for p in b], dtype=torch.long, device=runner.device)
            objs    = torch.tensor([p[1] for p in b], dtype=torch.long, device=runner.device)
            b_range = torch.arange(len(b), device=runner.device)
            scores  = torch.empty(len(b), len(cand), device=runner.device)
            for c, r in enumerate(cand):
                rel_b = torch.full((len(b),), r, dtype=torch.long, device=runner.device)
                scores[:, c] = runner.model.decode(subs, rel_b, all_ent, all_rel)[b_range, objs]
            mx, am = scores.max(dim=1)
            pres[st:st + len(b)] = mx.cpu().numpy()
            pred[st:st + len(b)] = [cand[i] for i in am.cpu().tolist()]
    return pres, pred


def build_eval_set(runner, split, ent2classes, allowed, sig_rel, known, neg_ratio, rng):
    """Static candidate set (pairs + labels), independent of model weights, so it
    can be built once and rescored each epoch."""
    cov, unc, neg = build_candidates(runner, split, ent2classes, allowed, known, neg_ratio, rng)
    sig_maj = sig_majority_fn(ent2classes, sig_rel)
    is_pos = np.concatenate([np.ones(len(cov), bool), np.ones(len(unc), bool), np.zeros(len(neg), bool)])
    true_r = np.array([r for _, _, r in cov] + [r for _, _, r in unc] + [-1] * len(neg))
    base_r = np.array([sig_maj(s, o) for s, o, _ in cov] + [-1] * len(unc) +
                      [sig_maj(s, o) for s, o, _ in neg])
    return dict(split=split, cov=cov, unc=unc, n_unc=len(unc), neg=neg, is_pos=is_pos,
                true_r=true_r, base_r=base_r,
                stats=dict(n_pos=len(cov) + len(unc), n_cov=len(cov), n_unc=len(unc), n_neg=len(neg)))


def score_eval_set(runner, eset):
    """Score a static eval set with the current model. Uncovered positives (unseen
    type signature) are scored like any other pair, not force-missed."""
    cov_pres, cov_pred = model_presence(runner, eset['split'], eset['cov'])
    unc_pres, unc_pred = model_presence(runner, eset['split'], eset['unc'])
    neg_pres, neg_pred = model_presence(runner, eset['split'], eset['neg'])
    pres = np.concatenate([cov_pres, unc_pres, neg_pres])
    pred = np.concatenate([cov_pred, unc_pred, neg_pred])
    return pres, pred


def fbeta(p, r, beta):
    b2 = beta * beta
    return 0.0 if (p == 0 and r == 0) else (1 + b2) * p * r / (b2 * p + r + 1e-12)


def metrics_at(pres, is_pos, true_r, pred_r, thr, beta):
    acc = pres >= thr
    tp = int((acc & is_pos).sum()); fp = int((acc & ~is_pos).sum()); fn = int((~acc & is_pos).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    m = acc & is_pos
    rel_acc = float((pred_r[m] == true_r[m]).mean()) if m.sum() else 0.0
    return dict(precision=prec, recall=rec, f1=fbeta(prec, rec, 1),
                fbeta=fbeta(prec, rec, beta), tp=tp, fp=fp, fn=fn, rel_acc=rel_acc)


def best_threshold(pres, is_pos, beta):
    """Threshold maximising F-beta over the PR curve."""
    prec, rec, thr = precision_recall_curve(is_pos.astype(int), pres)
    best_f, best_t = -1.0, 0.5
    for i, t in enumerate(thr):                       # prec/rec have one extra trailing point
        f = fbeta(prec[i], rec[i], beta)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def auprc(pres, is_pos):
    return float(average_precision_score(is_pos.astype(int), pres))
