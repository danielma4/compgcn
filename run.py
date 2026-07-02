from helper import *
from data_loader import *

# sys.path.append('./')
from model.models import *

import wandb

class Runner(object):

	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

		# Data layout (see scripts/ttl_to_compgcn.py): each split has an
		# observed *context* graph ``<split>_graph.txt`` (message-passing graph)
		# and a held-out *target* set ``<split>.txt`` (edges to predict). Whole
		# buildings are held out per split (inductive transfer), and within each
		# building a fraction of the target relations is held out.
		splits = ['train', 'valid', 'test']
		def read_file(path):
			with open(path) as f:
				for line in f:
					line = line.strip()
					if line:
						yield tuple(map(str.lower, line.split('\t')))

		def graph_path(sp):  return './data/{}/{}_graph.txt'.format(self.p.dataset, sp)
		def target_path(sp): return './data/{}/{}.txt'.format(self.p.dataset, sp)

		# Vocab from ALL files (context + targets, every split).
		ent_set, rel_set = OrderedSet(), OrderedSet()
		for sp in splits:
			for path in (graph_path(sp), target_path(sp)):
				for s, r, o in read_file(path):
					ent_set.add(s); rel_set.add(r); ent_set.add(o)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		# Context graphs (message-passing) and target triples (to predict), per split.
		self.graph_data = {sp: [] for sp in splits}
		self.data       = {sp: [] for sp in splits}
		for sp in splits:
			for s, r, o in read_file(graph_path(sp)):
				self.graph_data[sp].append((self.ent2id[s], self.rel2id[r], self.ent2id[o]))
			for s, r, o in read_file(target_path(sp)):
				self.data[sp].append((self.ent2id[s], self.rel2id[r], self.ent2id[o]))

		# Relations we predict = those appearing in the target files. Relation
		# prediction ranks among exactly these (not the full relation vocab).
		self.target_rels = sorted({r for sp in splits for (_, r, _) in self.data[sp]})

		if getattr(self.p, 'node_feat', 'identity') == 'type':
			self.build_type_features()                      # reads rdf:type from the context graphs (split-invariant)

		self._build_train_derived()

	def _build_train_derived(self):
		"""(Re)build everything downstream of the train/valid/test split: relation-
		prediction tensors, negative-sampling pools, KvsAll labels, the train
		dataloader, and the per-split message-passing graphs. Called once at load,
		and again per fold when cross-validation re-partitions buildings between
		train and valid (the vocab and type features are split-invariant, so they
		are NOT rebuilt)."""
		# Relation-prediction objective: held-out TRAIN targets only (absent from
		# the train graph, so predicting them is non-trivial).
		self.rel_train = torch.tensor(self.data['train'], dtype=torch.long)

		# Hard-negative machinery: per target relation, the pool of entities seen
		# as its object (type-appropriate tails), and all true train target pairs
		# (to avoid sampling a real edge as a negative).
		target_set = set(self.target_rels)
		self.tail_pool = ddict(list)
		self.known_train_pairs = set()
		for sub, rel, obj in self.graph_data['train'] + self.data['train']:
			if rel in target_set:
				self.tail_pool[rel].append(obj)
				self.known_train_pairs.add((sub, obj))
		self.tail_pool = dict(self.tail_pool)

		# Auxiliary entity-prediction objective (KvsAll): predict objects for
		# (s, r) over all train-building edges (context + held-out targets), so
		# no true edge is treated as a negative.
		sr2o = ddict(set)
		for sub, rel, obj in self.graph_data['train'] + self.data['train']:
			sr2o[(sub, rel)].add(obj)
			sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.sr2o = {k: list(v) for k, v in sr2o.items()}

		# sr2o_all: every known edge across splits (used to exclude knowns at
		# prediction time in predict_relations.py).
		for sp in ['valid', 'test']:
			for sub, rel, obj in self.graph_data[sp] + self.data[sp]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		self.triples = {'train': [
			{'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1}
			for (sub, rel) in self.sr2o
		]}

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		if self.triples['train']:
			self.data_iter = {'train': get_data_loader(TrainDataset, 'train', self.p.batch_size)}
		else:
			self.data_iter = {}

		# Phase-specific message-passing graphs: train on the train buildings'
		# context; encode each held-out building over its OWN context at eval.
		self.edge_index,       self.edge_type       = self.construct_adj(self.graph_data['train'])
		self.edge_index_valid, self.edge_type_valid = self.construct_adj(self.graph_data['valid'])
		self.edge_index_test,  self.edge_type_test  = self.construct_adj(self.graph_data['test'])

		self.so2r = None             # filtered-ranking cache (split-invariant union, rebuilt lazily)
		self._pres_cache = None      # presence candidate sets (depend on the valid/test split)

	def build_type_features(self):
		"""Builds per-entity Brick-class features for inductive transfer.

		Each entity is represented by the set of classes it is an instance of
		(its rdf:type objects). Because the Brick class vocabulary is shared
		across buildings, entities in an unseen building map onto known class
		embeddings, so the trained model can embed and score them. Sets
		``self.p.num_class`` and ``self.p.ent_class_idx`` / ``ent_class_mask``,
		which the model averages into each node's initial embedding.

		rdf:type assertions are treated as known inputs (we always know what a
		node *is*); only the relational edges are prediction targets, so using
		types from every split is not leakage.
		"""
		type_rel = self.rel2id.get('rdf:type')
		if type_rel is None:
			raise ValueError(
				"node_feat='type' requires rdf:type triples, but none were "
				"found. Regenerate the data without --no-types.")

		class2id = {'<unk>': 0}                      # index 0 = no-type fallback
		ent2classes = ddict(list)
		for split in ['train', 'valid', 'test']:
			for sub, rel, obj in self.graph_data[split]:   # rdf:type lives in the context graphs
				if rel == type_rel:
					cid = class2id.setdefault(self.id2ent[obj], len(class2id))
					ent2classes[sub].append(cid)

		max_k = max((len(v) for v in ent2classes.values()), default=1)
		idx  = torch.zeros((self.p.num_ent, max_k), dtype=torch.long)
		mask = torch.zeros((self.p.num_ent, max_k), dtype=torch.float)
		for e in range(self.p.num_ent):
			classes = ent2classes.get(e) or [0]      # no rdf:type -> '<unk>'
			for j, c in enumerate(classes):
				idx[e, j], mask[e, j] = c, 1.0

		self.p.num_class      = len(class2id)
		self.p.ent_class_idx  = idx
		self.p.ent_class_mask = mask
		self.class2id         = class2id
		self.logger.info('Type features: {} classes, up to {} types/entity'.format(
			self.p.num_class, max_k))

	def construct_adj(self, triples):
		"""Builds the (edge_index, edge_type) message-passing graph from a list
		of (sub, rel, obj) triples, adding a reverse edge (rel + num_rel) for
		each, as CompGCN expects."""
		edge_index, edge_type = [], []

		for sub, rel, obj in triples:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in triples:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = self.add_optimizer(self.model.parameters())
		self.rel_criterion     = torch.nn.CrossEntropyLoss()        # relation-prediction objective
		self.hardneg_criterion = torch.nn.BCEWithLogitsLoss()       # hard-negative presence objective

		self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		self.logger.info('Model parameters: {:,}'.format(self.n_params))


	def add_model(self, model, score_func):
		"""
		Creates the computational graph

		Parameters
		----------
		model_name:     Contains the model name to be created
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

		model.to(self.device)
		return model

	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer for learning the parameters of the model
		
		"""
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split

		
		Returns
		-------
		Head, Relation, Tails, labels
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

		Parameters
		----------
		save_path: path where the model is saved
		
		Returns
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p),
			'n_params'	: getattr(self, 'n_params', 0),
			# Total fold training time (all epochs), stamped after the loop ends; falls
			# back to elapsed-since-fit-start for the rare mid-training save.
			'train_time'	: self._train_time if getattr(self, '_train_time', None) is not None
					  else ((time.time() - self._fit_start) if hasattr(self, '_fit_start') else 0.0),
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		state			= torch.load(load_path, map_location=self.device)
		state_dict		= state['state_dict']
		self.best_val		= state.get('best_val', {})
		self.best_val_mrr	= self.best_val.get('mrr', 0.0)
		self.train_time		= state.get('train_time', 0.0)

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		return results

	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]'.format(split.title(), mode.title(), step))

		return results

	def relation_ranks(self, split='valid', sample=None):
		"""Per-edge filtered rank of the true relation, for Stage-2 evaluation.

		For each held-out ``(s, r, o)`` target triple, score every target relation
		against the known object and rank the true one among ``self.target_rels``
		(not the full vocab). The split's own context graph is the message-passing
		input (inductive). Filtered: other true target-relations on the same (s, o)
		are masked so they don't penalise the target's rank.

		Returns (ranks, rels): 1-indexed rank and the true relation id per edge.
		"""
		self.model.eval()
		if getattr(self, 'so2r', None) is None:                   # (sub, obj) -> true target relations
			so2r = ddict(set)
			for sp in ['train', 'valid', 'test']:
				for s, r, o in self.graph_data[sp] + self.data[sp]:
					if r in self.target_rels:
						so2r[(s, o)].add(r)
			self.so2r = {k: list(v) for k, v in so2r.items()}

		graph = {
			'valid': (self.edge_index_valid, self.edge_type_valid),
			'test':  (self.edge_index_test,  self.edge_type_test),
		}.get(split, (self.edge_index, self.edge_type))

		triples = self.data[split]
		if sample is not None and len(triples) > sample:
			rng = random.Random(self.p.seed)
			triples = [triples[i] for i in rng.sample(range(len(triples)), sample)]

		cand    = self.target_rels
		rel2col = {r: c for c, r in enumerate(cand)}
		C       = len(cand)
		all_ranks, all_rels = [], []
		with torch.no_grad():
			all_ent, all_rel = self.model.encode(*graph)          # encode once, reuse
			for start in range(0, len(triples), self.p.batch_size):
				batch    = triples[start:start + self.p.batch_size]
				subs     = torch.tensor([t[0] for t in batch], dtype=torch.long, device=self.device)
				objs     = torch.tensor([t[2] for t in batch], dtype=torch.long, device=self.device)
				tgt_col  = torch.tensor([rel2col[t[1]] for t in batch], dtype=torch.long, device=self.device)
				B        = subs.size(0)
				b_range  = torch.arange(B, device=self.device)

				scores = torch.empty(B, C, device=self.device)
				for c, r in enumerate(cand):                       # score each target relation
					rel_b = torch.full((B,), r, dtype=torch.long, device=self.device)
					scores[:, c] = self.model.decode(subs, rel_b, all_ent, all_rel)[b_range, objs]

				target = scores[b_range, tgt_col].clone()         # filtered ranking
				for i, t in enumerate(batch):
					others = [rel2col[r] for r in self.so2r.get((t[0], t[2]), []) if r in rel2col]
					if others:
						scores[i, torch.tensor(others, device=self.device)] = -1e7
				scores[b_range, tgt_col] = target

				ranks = 1 + torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)[b_range, tgt_col]
				all_ranks.extend(ranks.float().cpu().tolist())
				all_rels.extend(t[1] for t in batch)
		return np.array(all_ranks), np.array(all_rels)

	def predict_relation(self, split='valid', sample=None):
		"""MRR / Hits@k over a split's held-out target edges (wraps relation_ranks)."""
		ranks, _ = self.relation_ranks(split, sample=sample)
		if not len(ranks):
			return {'mrr': 0.0, 'count': 0, 'hits@1': 0.0}
		C   = len(self.target_rels)
		ks  = [k for k in (1, 3) if k < C] or [1]                 # Hits@k meaningful only for k < #candidates
		out = {'mrr': round(float((1.0 / ranks).mean()), 5), 'count': int(len(ranks))}
		out.update({'hits@{}'.format(k): round(float((ranks <= k).mean()), 5) for k in ks})
		return out


	def score_relations_for_pairs(self, split, pairs):
		"""Argmax target relation for each (s, o) pair (Stage-2 labelling of arbitrary
		pairs), encoding over the split's graph. Returns an np.array of relation ids."""
		self.model.eval()
		preds = np.full(len(pairs), -1, dtype=int)
		if not pairs:
			return preds
		graph = {'valid': (self.edge_index_valid, self.edge_type_valid),
			 'test':  (self.edge_index_test,  self.edge_type_test)}.get(
				 split, (self.edge_index, self.edge_type))
		cand = self.target_rels
		with torch.no_grad():
			all_ent, all_rel = self.model.encode(*graph)
			for start in range(0, len(pairs), self.p.batch_size):
				batch   = pairs[start:start + self.p.batch_size]
				subs    = torch.tensor([p[0] for p in batch], dtype=torch.long, device=self.device)
				objs    = torch.tensor([p[1] for p in batch], dtype=torch.long, device=self.device)
				B       = subs.size(0); b_range = torch.arange(B, device=self.device)
				scores  = torch.empty(B, len(cand), device=self.device)
				for c, r in enumerate(cand):
					rel_b = torch.full((B,), r, dtype=torch.long, device=self.device)
					scores[:, c] = self.model.decode(subs, rel_b, all_ent, all_rel)[b_range, objs]
				am = scores.argmax(dim=1).cpu().tolist()
				preds[start:start + B] = [cand[i] for i in am]
		return preds

	def eval_presence(self, split, neg_ratio=10.0, beta=0.5, threshold=None):
		"""Presence (edge-existence) metrics on a split: AUPRC + precision/recall/
		F1/F-beta. If ``threshold`` is None it is tuned on this split to maximise
		F-beta (use for valid / model selection); pass a valid-tuned threshold for
		an honest test number. beta<1 favours precision. The candidate set
		(positives + type-compatible negatives) is built once and cached, then
		rescored each epoch."""
		import presence as P
		if getattr(self, '_pres_cache', None) is None:           # build candidate sets once
			rng    = random.Random(self.p.seed)
			e2c    = P.build_ent2classes(self)
			allowed, sig_rel = P.build_signatures(self, e2c)
			known  = P.known_pairs(self)
			self._pres_cache = {sp: P.build_eval_set(self, sp, e2c, allowed, sig_rel, known, neg_ratio, rng)
					    for sp in ['valid', 'test']}
		eset       = self._pres_cache[split]
		pres, pred = P.score_eval_set(self, eset)
		thr        = P.best_threshold(pres, eset['is_pos'], beta) if threshold is None else threshold
		m          = P.metrics_at(pres, eset['is_pos'], eset['true_r'], pred, thr, beta)
		m['auprc'] = P.auprc(pres, eset['is_pos'])
		m['thr']   = thr
		return m

	def sample_hard_negatives(self, subs, rels, n_neg):
		"""Type-constrained tail corruption (Krompaß et al. 2015): for each
		positive (s, r), draw ``n_neg`` objects from the pool of entities that
		actually appear as objects of r, rejecting any (s, o) that is a real
		train edge. These are *hard* negatives — right type, wrong specific pair."""
		ns, nr, no = [], [], []
		for s, r in zip(subs.tolist(), rels.tolist()):
			pool = self.tail_pool.get(r)
			if not pool:
				continue
			for _ in range(n_neg):
				o = pool[random.randrange(len(pool))]
				for _retry in range(4):                     # avoid sampling a real edge
					if o != s and (s, o) not in self.known_train_pairs:
						break
					o = pool[random.randrange(len(pool))]
				ns.append(s); nr.append(r); no.append(o)
		dev = self.device
		return (torch.tensor(ns, dtype=torch.long, device=dev),
			torch.tensor(nr, dtype=torch.long, device=dev),
			torch.tensor(no, dtype=torch.long, device=dev))

	def run_epoch(self, epoch, val_mrr = 0):
		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		self.model.train()
		losses, rel_losses, hn_losses = [], [], []
		train_iter = iter(self.data_iter['train'])

		# Objectives (all share one GCN encode pass per step):
		#  - ent_weight: main KvsAll *entity* prediction (predict o given (s,r)). We
		#    never rank entities at eval, so this is off by default for Stage 2; it
		#    only shapes the per-entity bias (irrelevant to relation ranking).
		#  - rel_weight: relation prediction (predict r from (s,o); arXiv:2110.02834)
		#    — the aligned Stage-2 objective; trains the same trilinear term eval ranks.
		#  - hard_neg: edge-existence presence (Stage-1's job; off for Stage 2).
		ent_w   = getattr(self.p, 'ent_weight', 1.0)
		rel_w   = getattr(self.p, 'rel_weight', 0.0)
		hn_n    = int(getattr(self.p, 'hard_neg', 0))
		hn_w    = getattr(self.p, 'hard_neg_weight', 0.0)
		use_rel = rel_w > 0 and hasattr(self.model, 'rel_score_all')
		use_hn  = use_rel and hn_n > 0 and hn_w > 0
		if use_rel:
			rel_data = self.rel_train[torch.randperm(self.rel_train.size(0))]
			rel_ptr  = 0

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')

			loss = None
			if ent_w > 0:
				pred = self.model.forward(sub, rel)
				loss = ent_w * self.model.loss(pred, label)

			if use_rel:
				if rel_ptr + self.p.batch_size > rel_data.size(0):       # reshuffle when exhausted
					rel_data = self.rel_train[torch.randperm(self.rel_train.size(0))]
					rel_ptr  = 0
				rb       = rel_data[rel_ptr:rel_ptr + self.p.batch_size].to(self.device)
				rel_ptr += self.p.batch_size

				all_ent, all_rel = self.model.objective_encode()        # one GCN pass, shared below
				logits   = self.model.rel_score_all(rb[:, 0], rb[:, 2], all_ent, all_rel)
				rel_loss = self.rel_criterion(logits, rb[:, 1])
				rel_losses.append(rel_loss.item())
				rel_term = rel_w * rel_loss
				loss     = rel_term if loss is None else loss + rel_term

				if use_hn:
					ns, nr, no = self.sample_hard_negatives(rb[:, 0], rb[:, 1], hn_n)
					pos = self.model.triple_score(rb[:, 0], rb[:, 1], rb[:, 2], all_ent, all_rel)
					neg = self.model.triple_score(ns, nr, no, all_ent, all_rel)
					pres_logit = torch.cat([pos, neg])
					pres_label = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
					hn_loss    = self.hardneg_criterion(pres_logit, pres_label)
					hn_losses.append(hn_loss.item())
					loss	   = loss + hn_w * hn_loss

			if loss is None:                                # no active objective (ent_weight=0 and rel_weight=0)
				raise ValueError('No training objective active: set ent_weight>0 and/or rel_weight>0.')
			loss.backward()
			if getattr(self.p, 'grad_clip', 1.0) > 0:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.p.grad_clip)
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				avg_loss = np.mean(losses)
				rel_msg  = ',  Rel Loss:{:.5}'.format(np.mean(rel_losses)) if use_rel else ''
				hn_msg   = ',  HN Loss:{:.5}'.format(np.mean(hn_losses))   if use_hn  else ''
				self.logger.info('[E:{}| {}]: Train Loss:{:.5}{}{},  Best Val MRR:{:.4}'.format(epoch, step, avg_loss, rel_msg, hn_msg, getattr(self, 'best_val_mrr', 0.0)))
				wandb.log({'train/step_loss': avg_loss})

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss


	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		----------

		Returns
		-------
		"""
		wandb.init(
			project='compgcn',
			name=self.p.name,
			config=vars(self.p),
			mode='disabled' if getattr(self.p, 'wandb_disabled', False) else 'online',
		)

		self._fit_start = time.time()
		save_base = os.path.join('./checkpoints', self.p.name)
		self._save_path_base = save_base
		os.makedirs(os.path.dirname(save_base), exist_ok=True)

		if self.p.restore:
			self.load_model(save_base)
			self.logger.info('Successfully Loaded previous model')

		ckpt = self._train_fold(save_base)
		self.load_model(ckpt)
		test_rel = self.predict_relation('test')
		self.logger.info('[Test] rel: MRR {:.5} H@1 {:.5}'.format(test_rel['mrr'], test_rel['hits@1']))
		wandb.log({'test/rel_mrr': test_rel['mrr'], 'test/rel_hits@1': test_rel['hits@1']})

	def _train_fold(self, save_base, fold_label=''):
		"""Single training run (one fold or the full run). Returns best checkpoint
		path. Selection is on validation MRR; the saved checkpoint records the TOTAL
		training time (all epochs), not time-to-best."""
		fold_start = time.time()
		self._train_time = None
		self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0
		self._save_path_base = save_base
		self._best_save_path = save_base
		prefix = f'[{fold_label}] ' if fold_label else ''
		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss = self.run_epoch(epoch)
			val_rel = self.predict_relation('valid')
			self.logger.info('{}[Epoch {}] rel(valid): MRR {:.4} H@1 {:.4}'.format(
				prefix, epoch, val_rel['mrr'], val_rel['hits@1']))
			wandb.log({'train/loss': train_loss, 'valid/rel_mrr': val_rel['mrr'],
			           'valid/rel_hits@1': val_rel['hits@1']})
			if val_rel['mrr'] > self.best_val_mrr:
				self.best_val = val_rel
				self.best_val_mrr = val_rel['mrr']
				self.best_epoch = epoch
				ckpt = '{}_mrr{:.3f}_e{:03d}'.format(save_base, val_rel['mrr'], epoch)
				self.save_model(ckpt)
				if self._best_save_path != ckpt and os.path.exists(self._best_save_path):
					os.remove(self._best_save_path)        # keep only the current best per fold
				self._best_save_path = ckpt
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt > 15:
					self.logger.info('{}Early Stopping!!'.format(prefix))
					break
			self.logger.info('{}[Epoch {}]: Loss {:.5}, Best MRR {:.4}\n'.format(
				prefix, epoch, train_loss, self.best_val_mrr))
		# Re-stamp the best checkpoint with the fold's total wall-clock training time.
		self._train_time = time.time() - fold_start
		if os.path.exists(self._best_save_path):
			self.load_model(self._best_save_path)
			self.save_model(self._best_save_path)
		self._train_time = None
		return self._best_save_path

	def _train_one_fold(self, save_base, label):
		"""CV hook: train one fold for the current train/valid split. Overridden by
		LinkRunner for the existence objective."""
		return self._train_fold(save_base, fold_label=label)

	def cross_validate(self, n_alloc=5, n_seeds=3):
		"""Building-allocation x seed cross-validation. The test buildings stay fixed;
		the non-test buildings are split into train/valid n_alloc ways (folds over
		buildings), and each allocation is trained with n_seeds random seeds. Writes a
		``.folds`` list of all n_alloc*n_seeds checkpoints (deterministic order, so a
		Stage-1 fold pairs with the same-index Stage-2 fold in the pipeline)."""
		wandb.init(project='compgcn', name=self.p.name, config=vars(self.p),
		           mode='disabled' if getattr(self.p, 'wandb_disabled', False) else 'online')
		self._fit_start = time.time()
		save_base = os.path.join('./checkpoints', self.p.name)
		os.makedirs(os.path.dirname(save_base), exist_ok=True)

		# Group the pooled (train+valid) buildings' data and context by building.
		bof = lambda e: self.id2ent[e].split(':', 1)[0]
		data_by_b, graph_by_b = ddict(list), ddict(list)
		for t in self.data['train'] + self.data['valid']:
			data_by_b[bof(t[0])].append(t)
		for t in self.graph_data['train'] + self.graph_data['valid']:
			graph_by_b[bof(t[0])].append(t)
		buildings = sorted(set(data_by_b) | set(graph_by_b))
		random.Random(self.p.seed).shuffle(buildings)
		fold_sz = max(1, len(buildings) // n_alloc)
		self.logger.info(f'CV over {len(buildings)} non-test buildings: '
		                 f'{n_alloc} allocations x {n_seeds} seeds = {n_alloc * n_seeds} runs')

		fold_ckpts = []
		for a in range(n_alloc):
			lo = a * fold_sz
			hi = (a + 1) * fold_sz if a < n_alloc - 1 else len(buildings)
			valid_b = set(buildings[lo:hi])
			train_b = [b for b in buildings if b not in valid_b]
			self.data['valid']       = [t for b in valid_b for t in data_by_b[b]]
			self.graph_data['valid'] = [t for b in valid_b for t in graph_by_b[b]]
			self.data['train']       = [t for b in train_b for t in data_by_b[b]]
			self.graph_data['train'] = [t for b in train_b for t in graph_by_b[b]]
			self.logger.info('===== Allocation {}/{}: valid={} ====='.format(
				a + 1, n_alloc, sorted(valid_b)))
			for s in range(n_seeds):
				seed = self.p.seed + 1000 * a + s
				random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
				self._build_train_derived()
				self.model     = self.add_model(self.p.model, self.p.score_func)
				self.optimizer = self.add_optimizer(self.model.parameters())
				label = f'a{a}s{s}'
				ckpt  = self._train_one_fold(f'{save_base}_{label}', label)
				fold_ckpts.append(ckpt)

		fold_list = save_base + '.folds'
		with open(fold_list, 'w') as f:
			f.write('\n'.join(fold_ckpts))
		self.logger.info(f'Saved {len(fold_ckpts)} fold checkpoints → {fold_list}')
		self.logger.info('Run the pipeline for the full mean±std report.')

	def score_relation_logits_for_pairs(self, split, pairs):
		"""Raw logit matrix (N, n_target_rels) — use for ensemble averaging before argmax."""
		if not pairs:
			return np.zeros((0, len(self.target_rels)))
		self.model.eval()
		graph = {'valid': (self.edge_index_valid, self.edge_type_valid),
		         'test':  (self.edge_index_test,  self.edge_type_test)}.get(
		             split, (self.edge_index, self.edge_type))
		cand = self.target_rels
		out = np.zeros((len(pairs), len(cand)), dtype=np.float32)
		with torch.no_grad():
			all_ent, all_rel = self.model.encode(*graph)
			for st in range(0, len(pairs), self.p.batch_size):
				b = pairs[st:st + self.p.batch_size]
				subs = torch.tensor([p[0] for p in b], dtype=torch.long, device=self.device)
				objs = torch.tensor([p[1] for p in b], dtype=torch.long, device=self.device)
				b_range = torch.arange(len(b), device=self.device)
				for c, r in enumerate(cand):
					rel_b = torch.full((len(b),), r, dtype=torch.long, device=self.device)
					out[st:st + len(b), c] = self.model.decode(
					    subs, rel_b, all_ent, all_rel)[b_range, objs].cpu().numpy()
		return out

def build_parser():
	"""Builds the shared argument parser (used by run.py and predict_relations.py)."""
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--name', '-name',		default='testrun',				help='Set run name for saving/restoring models')
	parser.add_argument('--data', '-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('--model', '-model',	dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('--score_func', '-score_func',	dest='score_func',	default='conve',	help='Score Function for Link prediction')
	parser.add_argument('--opn', '-opn',            dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
	parser.add_argument('--node_feat', '-node_feat',	dest='node_feat',	default='identity', choices=['identity', 'type'], help="Initial node features: 'identity' (per-entity embedding, transductive) or 'type' (mean of Brick-class embeddings, enables transfer to unseen buildings)")

	parser.add_argument('--batch', '-batch',        dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('--gamma', '-gamma',	type=float,             default=40.0,			help='Margin')
	parser.add_argument('--gpu', '-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('--epoch', '-epoch',	dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('--l2', '-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('--lr', '-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('--lbl_smooth', '-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('--ent_weight', '-ent_weight',	dest='ent_weight',	type=float,     default=1.0,	help='Weight of the main KvsAll entity-prediction loss (predict o given (s,r)). 0 = off; we never rank entities, so Stage 2 can drop or down-weight this.')
	parser.add_argument('--rel_weight', '-rel_weight',	dest='rel_weight',	type=float,     default=0.0,	help='Weight of the auxiliary relation-prediction loss (0 = off). Predicts r from (s,o); aligns training with the relation-prediction task.')
	parser.add_argument('--hard_neg', '-hard_neg',		dest='hard_neg',	type=int,       default=0,	help='Type-compatible hard negatives sampled per positive for the presence objective (0 = off).')
	parser.add_argument('--hard_neg_weight', '-hard_neg_weight', dest='hard_neg_weight', type=float, default=1.0,	help='Weight of the hard-negative presence (edge-existence) loss.')
	parser.add_argument('--beta', '-beta',		dest='beta',		type=float,     default=0.5,	help='F-beta beta for presence threshold tuning / model selection. <1 favours precision (a spurious edge is worse than a missed one), >1 favours recall.')
	parser.add_argument('--num_workers', '-num_workers',	type=int,               default=10,             help='Number of processes to construct batches')
	parser.add_argument('--seed', '-seed',          dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('--restore', '-restore',    dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('--bias', '-bias',          dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('--num_bases', '-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('--init_dim', '-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('--gcn_dim', '-gcn_dim',	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('--embed_dim', '-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('--gcn_layer', '-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('--gcn_drop', '-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('--hid_drop', '-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('--hid_drop2', '-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('--feat_drop', '-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('--k_w', '-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('--k_h', '-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('--num_filt', '-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('--ker_sz', '-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('--logdir', '-logdir',      dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('--config_dir', '-config',  dest='config_dir',      default='./config/',            help='Config directory (for log_config.json)')
	parser.add_argument('--config', '-yaml',        dest='yaml_config',     default=None,                   help='Path to a YAML config file. Its values override the built-in defaults; any explicit CLI flag still overrides the YAML.')
	parser.add_argument('--grad_clip', type=float, default=1.0,
	                    help='Max gradient norm (clip_grad_norm_) for training stability. 0 disables.')
	parser.add_argument('--cv_alloc', type=int, default=1,
	                    help='Building-allocation folds: split the non-test buildings into K '
	                         'train/valid partitions (test stays fixed). 1 = no cross-validation.')
	parser.add_argument('--cv_seeds', type=int, default=1,
	                    help='Random seeds trained per building allocation. Total CV runs = cv_alloc*cv_seeds.')
	parser.add_argument('--wandb_disabled', action='store_true',
	                    help='Disable wandb logging (equivalent to WANDB_MODE=disabled).')
	return parser


def parse_with_yaml(parser):
	"""Parses args with precedence: explicit CLI flag > YAML file > built-in default."""
	args, _ = parser.parse_known_args()
	if args.yaml_config:
		import yaml
		with open(args.yaml_config) as f:
			yaml_cfg = yaml.safe_load(f) or {}
		valid_keys = set(vars(args).keys())
		unknown = set(yaml_cfg) - valid_keys
		if unknown:
			raise ValueError(
				f"Unknown keys in {args.yaml_config}: {sorted(unknown)}.\n"
				f"Valid keys (argparse dest names): {sorted(valid_keys)}")
		parser.set_defaults(**yaml_cfg)   # YAML overrides defaults; re-parse lets CLI override YAML
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_with_yaml(build_parser())


	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Runner(args)
	if args.cv_alloc > 1 or args.cv_seeds > 1:
		model.cross_validate(args.cv_alloc, args.cv_seeds)
	else:
		model.fit()
