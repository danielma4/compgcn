"""Standalone **edge-existence** model — binary classification only.

A separate model from run.py's multi-objective CompGCN. It answers ONE question
for each ordered node pair (s, o): should *some* target relation exist between
them? (relation-agnostic). This is triple classification (Socher 2013; Wang 2014)
reduced to existence, in the inductive / unseen-building setting (GraIL,
Teru 2020).

Why a separate model: in run.py the existence signal was a max over the three
DistMult relation-decoders, sharing a representation with the relation-prediction
objective. Here existence gets its own binary head and its own loss, with no
relation objective competing for capacity. The CompGCN encoder is reused (typed
relations still drive message passing); only the decoder changes.

Training: positives = held-out train target edges (s, o) regardless of relation;
negatives = type-compatible hard negatives (run.py's sample_hard_negatives).
BCEWithLogits. Eval reuses presence.py's candidate sets and metrics, scoring each
pair with the binary edge head instead of max-over-relations. Checkpoint selected
on validation AUPRC (threshold-free); decision rule is a fixed confidence
threshold (--threshold, default 0.6), applied identically to valid and test.

NOTE (see prior diagnosis): edge existence did not transfer to unseen buildings
with the shared model — test AUPRC ~= base rate. This script measures whether a
dedicated binary head changes that. Expect to confirm the ceiling, not erase it.

Usage:
    python link_existence.py --config config/brick.yaml --name link_exist \
        --neg_ratio 10 --beta 2
    python link_existence.py --config config/brick.yaml --load checkpoints/link_exist \
        --eval_only
"""

import os
import time
import random

import numpy as np
import torch

from helper import set_gpu
from model.models import CompGCN_LinkPredict
from run import build_parser, parse_with_yaml, Runner
from name_overlap_check import tokens, ngrams, jaccard
import presence as P

PAIR_FEAT_DIM = 3   # tok_jaccard, id_shared, char3_jac


class LinkRunner(Runner):
	"""Reuses Runner's data pipeline (splits, type features, adjacencies,
	hard-negative machinery, presence candidate sets); swaps in the binary
	edge-existence model and a single-objective train/eval loop."""

	def add_model(self, model, score_func):
		self.p.n_pair_feats = PAIR_FEAT_DIM if getattr(self.p, 'name_feats', False) else 0
		m = CompGCN_LinkPredict(self.edge_index, self.edge_type, params=self.p)
		m.to(self.device)
		return m

	def _build_name_cache(self):
		"""Precompute per-entity token / numeric / char-ngram sets from the entity
		URI strings, so per-pair name-overlap features are cheap each batch."""
		self._tok = [tokens(self.id2ent[i]) for i in range(self.p.num_ent)]
		self._num = [{t for t in s if t.isdigit()} for s in self._tok]
		self._ng  = [ngrams(self.id2ent[i]) for i in range(self.p.num_ent)]

	def name_feats(self, subs, objs):
		"""(B, 3) name-overlap features for ordered pairs, or None if disabled."""
		if not getattr(self.p, 'name_feats', False):
			return None
		if getattr(self, '_tok', None) is None:
			self._build_name_cache()
		rows = []
		for s, o in zip(subs.tolist(), objs.tolist()):
			rows.append((jaccard(self._tok[s], self._tok[o]),
				     1.0 if (self._num[s] & self._num[o]) else 0.0,
				     jaccard(self._ng[s], self._ng[o])))
		return torch.tensor(rows, dtype=torch.float, device=self.device)

	# ---- training ---------------------------------------------------------
	def train_existence(self, epoch):
		self.model.train()
		pos = self.rel_train[torch.randperm(self.rel_train.size(0))]
		bs   = self.p.batch_size
		hn_n = max(1, int(getattr(self.p, 'hard_neg', 5)))
		losses = []
		for st in range(0, pos.size(0), bs):
			self.optimizer.zero_grad()
			pb = pos[st:st + bs].to(self.device)
			ps, pr, po = pb[:, 0], pb[:, 1], pb[:, 2]
			ns, nr, no = self.sample_hard_negatives(ps, pr, hn_n)

			all_ent, _ = self.model.encode_train(self.model.drop, self.model.drop)
			pos_logit  = self.model.edge_score(ps, po, all_ent, self.name_feats(ps, po))
			neg_logit  = self.model.edge_score(ns, no, all_ent, self.name_feats(ns, no))
			logit = torch.cat([pos_logit, neg_logit])
			label = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])

			loss = self.bce(logit, label)
			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())
		return float(np.mean(losses)) if losses else 0.0

	# ---- evaluation -------------------------------------------------------
	def _ensure_eval_sets(self, neg_ratio):
		if getattr(self, '_pres_cache', None) is None:
			rng = random.Random(self.p.seed)
			e2c = P.build_ent2classes(self)
			allowed, sig_rel = P.build_signatures(self, e2c)
			known = P.known_pairs(self)
			self._pres_cache = {sp: P.build_eval_set(self, sp, e2c, allowed, sig_rel, known, neg_ratio, rng)
					    for sp in ['valid', 'test']}

	def score_existence(self, eset):
		"""Presence probability per candidate from the binary edge head. All
		positives are scored, including uncovered ones (unseen type signature) —
		the name-overlap head doesn't depend on the type gate."""
		self.model.eval()
		graph = {'valid': (self.edge_index_valid, self.edge_type_valid),
			 'test':  (self.edge_index_test,  self.edge_type_test)}.get(
				 eset['split'], (self.edge_index, self.edge_type))
		bs = self.p.batch_size
		with torch.no_grad():
			all_ent, _ = self.model.encode(*graph)

			def score(pairs):
				out = np.zeros(len(pairs))
				for st in range(0, len(pairs), bs):
					b = pairs[st:st + bs]
					subs = torch.tensor([p[0] for p in b], dtype=torch.long, device=self.device)
					objs = torch.tensor([p[1] for p in b], dtype=torch.long, device=self.device)
					out[st:st + len(b)] = torch.sigmoid(
						self.model.edge_score(subs, objs, all_ent, self.name_feats(subs, objs))).cpu().numpy()
				return out

			cov_pres = score(eset['cov'])
			unc_pres = score(eset['unc'])
			neg_pres = score(eset['neg'])
		return np.concatenate([cov_pres, unc_pres, neg_pres])

	def eval_existence(self, split, neg_ratio, beta, threshold):
		"""Metrics at a fixed decision threshold. AUPRC is threshold-free."""
		self._ensure_eval_sets(neg_ratio)
		eset   = self._pres_cache[split]
		pres   = self.score_existence(eset)
		is_pos = eset['is_pos']
		thr    = threshold

		acc = pres >= thr
		tp = int((acc & is_pos).sum()); fp = int((acc & ~is_pos).sum()); fn = int((~acc & is_pos).sum())
		prec = tp / (tp + fp) if tp + fp else 0.0
		rec  = tp / (tp + fn) if tp + fn else 0.0
		st   = eset['stats']
		return dict(precision=prec, recall=rec, f1=P.fbeta(prec, rec, 1.0),
			    fbeta=P.fbeta(prec, rec, beta), auprc=P.auprc(pres, is_pos), thr=thr,
			    tp=tp, fp=fp, fn=fn, n_pos=st['n_pos'], n_cov=st['n_cov'],
			    n_unc=st['n_unc'], n_neg=st['n_neg'])

	# ---- driver -----------------------------------------------------------
	def fit_existence(self, neg_ratio, beta):
		self._fit_start = time.time()
		save_base = os.path.join('./checkpoints', self.p.name)
		os.makedirs('./checkpoints', exist_ok=True)
		ckpt = self._fit_existence_fold(save_base, neg_ratio, beta)
		self.load_model(ckpt)
		self.report(neg_ratio, beta)

	def _fit_existence_fold(self, save_base, neg_ratio, beta, fold_label=''):
		"""Single existence-model training run (one fold). Selection on validation
		AUPRC (threshold-free); the saved checkpoint records TOTAL training time."""
		fold_start = time.time()
		self._train_time = None
		self.best_val_auprc, self.best_val, self.best_epoch = 0.0, {}, 0
		self._best_save_path = save_base
		thr = self.p.threshold
		prefix = f'[{fold_label}] ' if fold_label else ''
		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss = self.train_existence(epoch)
			val = self.eval_existence('valid', neg_ratio, beta, thr)
			self.logger.info(
				'{}[Epoch {}] loss {:.4} | AUPRC {:.3} | @{:.2}: '
				'P {:.3} R {:.3} F1 {:.3} F{:g} {:.3} (TP {} FP {} FN {})'.format(
					prefix, epoch, train_loss, val['auprc'], thr, val['precision'],
					val['recall'], val['f1'], beta, val['fbeta'], val['tp'], val['fp'], val['fn']))
			if val['auprc'] > self.best_val_auprc:
				self.best_val, self.best_val_auprc, self.best_epoch = val, val['auprc'], epoch
				self._best_save_path = '{}_auprc{:.3f}_e{:03d}'.format(save_base, val['auprc'], epoch)
				self.save_model(self._best_save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt > 25:
					self.logger.info('{}Early Stopping!!'.format(prefix))
					break
		# Re-stamp the best checkpoint with the fold's total training time.
		self._train_time = time.time() - fold_start
		if os.path.exists(self._best_save_path):
			self.load_model(self._best_save_path)
			self.save_model(self._best_save_path)
		self._train_time = None
		return self._best_save_path

	def _train_one_fold(self, save_base, label):
		"""CV hook (called by Runner.cross_validate): train one existence fold."""
		self.bce = torch.nn.BCEWithLogitsLoss()
		return self._fit_existence_fold(save_base, self.p.neg_ratio, self.p.beta, fold_label=label)

	def report(self, neg_ratio, beta):
		thr = self.p.threshold                                             # fixed cut, same on valid and test
		for sp in ['valid', 'test']:
			m   = self.eval_existence(sp, neg_ratio, beta, thr)
			covered = m['n_cov'] / m['n_pos'] if m['n_pos'] else 0.0
			base    = m['n_pos'] / (m['n_pos'] + m['n_neg']) if (m['n_pos'] + m['n_neg']) else 0.0
			self.logger.info(
				'=== {} (pos {}, type-covered {} [{:.0%}, now scored too], neg {}, '
				'base-rate {:.3}) ==='.format(sp, m['n_pos'], m['n_cov'], covered, m['n_neg'], base))
			self.logger.info(
				'  model: P {:.3} R {:.3} F1 {:.3} F{:g} {:.3} AUPRC {:.3} '
				'(thr {:.3}; TP {} FP {} FN {})'.format(
					m['precision'], m['recall'], m['f1'], beta, m['fbeta'], m['auprc'],
					m['thr'], m['tp'], m['fp'], m['fn']))


def main():
	parser = build_parser()
	parser.add_argument('--load', '-load', dest='load_path', default=None, help='Checkpoint to restore')
	parser.add_argument('--eval_only', '-eval_only', dest='eval_only', action='store_true',
			    help='Skip training; load --load and report.')
	parser.add_argument('--neg_ratio', '-neg_ratio', type=float, default=10.0,
			    help='Type-compatible negatives per positive in the eval candidate set.')
	parser.add_argument('--threshold', '-threshold', dest='threshold', type=float, default=0.6,
			    help='Fixed confidence threshold: accept an edge iff sigmoid(score) >= this.')
	parser.add_argument('--name_feats', '-name_feats', dest='name_feats', action='store_true',
			    help='Append s-o name-overlap features (tok/id/char3) to the edge head.')
	args = parse_with_yaml(parser)   # --beta comes from the shared parser (default 0.5, precision-favouring)


	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	runner = LinkRunner(args)
	runner.bce = torch.nn.BCEWithLogitsLoss()

	if args.eval_only:
		load_path = args.load_path or os.path.join('./checkpoints', args.name)
		if not os.path.exists(load_path):
			raise FileNotFoundError('Checkpoint not found: {} (pass --load)'.format(load_path))
		runner.load_model(load_path)
		runner.report(args.neg_ratio, args.beta)
	elif args.cv_alloc > 1 or args.cv_seeds > 1:
		runner.cross_validate(args.cv_alloc, args.cv_seeds)
	else:
		if args.load_path:
			runner.load_model(args.load_path)
		runner.fit_existence(args.neg_ratio, args.beta)


if __name__ == '__main__':
	main()
