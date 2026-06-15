from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.device		= self.edge_index.device

		# Node features: 'identity' = a learned embedding per entity (default,
		# transductive); 'type' = mean of the entity's Brick-class embeddings,
		# which lets the model embed entities in unseen buildings (inductive).
		self.node_feat = getattr(self.p, 'node_feat', 'identity')
		if self.node_feat == 'type':
			self.class_embed = get_param((self.p.num_class, self.p.init_dim))
			self.register_buffer('ent_class_idx',  self.p.ent_class_idx)
			self.register_buffer('ent_class_mask', self.p.ent_class_mask.unsqueeze(-1))
		else:
			self.init_embed	= get_param((self.p.num_ent,   self.p.init_dim))

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def get_node_embed(self):
		"""Returns the (num_ent, init_dim) initial node embedding matrix.

		For 'type' features, each entity embedding is the mean of its Brick
		class embeddings (entities with no rdf:type fall back to '<unk>').
		"""
		if self.node_feat == 'type':
			emb	= self.class_embed[self.ent_class_idx] * self.ent_class_mask
			denom	= self.ent_class_mask.sum(dim=1).clamp(min=1.0)
			return emb.sum(dim=1) / denom
		return self.init_embed

	def encode(self, edge_index=None, edge_type=None):
		"""Runs the CompGCN graph encoder once and returns (all_ent, all_rel).

		``edge_index``/``edge_type`` select the message-passing graph; they
		default to the training graph but at inference are set to a held-out
		building's own context graph (inductive transfer).

		For inference only (call under model.eval()/no_grad): with dropout and
		BatchNorm frozen the output is deterministic, so it can be computed
		once and reused across many decode() calls instead of being recomputed
		per batch. Mathematically identical to what forward_base computes.
		"""
		edge_index	= self.edge_index if edge_index is None else edge_index
		edge_type	= self.edge_type  if edge_type  is None else edge_type
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.get_node_embed(), edge_index, edge_type, rel_embed=r)
		x, r	= self.conv2(x, edge_index, edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x, r)
		return x, r

	def encode_train(self, drop1, drop2):
		"""Runs the GCN encoder with dropout (training mode), returning
		(all_ent, all_rel). Shared by the relation-prediction objective so it
		reuses the exact same node/relation representations as forward_base."""
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.get_node_embed(), self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x
		return x, r

	def objective_encode(self):
		"""One training-mode GCN encode pass whose (all_ent, all_rel) are shared
		by the relation-prediction and hard-negative objectives."""
		drop = getattr(self, 'drop', None) or getattr(self, 'hidden_drop')
		return self.encode_train(drop, drop)

	def forward_base(self, sub, rel, drop1, drop2):

		# initialize relations
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		# apply convolutions and dropout
		x, r	= self.conv1(self.get_node_embed(), self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		# get embeddings for sub and rel
		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		# add relation to subject
		obj_emb				= sub_emb + rel_emb

		# compute scores
		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		score	= torch.sigmoid(x)

		return score

	def decode(self, sub, rel, all_ent, all_rel):
		"""Scores (sub, rel) against all entities using cached encoder output."""
		obj_emb	= all_ent[sub] + all_rel[rel]
		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		return torch.sigmoid(x)

	def rel_logits(self, sub, obj):
		"""Relation-prediction logits: scores (sub, ?, obj) over all base relations."""
		x, r	= self.encode_train(self.drop, self.drop)
		delta	= x[obj] - x[sub]						# (B, d)
		R	= r[:self.p.num_rel]						# (num_rel, d)
		return self.p.gamma - torch.norm(R.unsqueeze(0) - delta.unsqueeze(1), p=1, dim=2)	# (B, num_rel)

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

	def decode(self, sub, rel, all_ent, all_rel):
		"""Scores (sub, rel) against all entities using cached encoder output."""
		obj_emb	= all_ent[sub] * all_rel[rel]
		x	= torch.mm(obj_emb, all_ent.transpose(1, 0)) + self.bias   # broadcast over batch
		return torch.sigmoid(x)

	def rel_score_all(self, sub, obj, all_ent, all_rel):
		"""Relation-prediction logits <sub * rel, obj> over all base relations,
		from already-encoded embeddings. (B, num_rel)."""
		so = all_ent[sub] * all_ent[obj]
		return torch.mm(so, all_rel[:self.p.num_rel].transpose(1, 0))

	def triple_score(self, sub, rel, obj, all_ent, all_rel):
		"""Per-triple presence logit <sub * rel, obj> + obj bias (pre-sigmoid),
		matching decode(). Used by the hard-negative presence objective."""
		return (all_ent[sub] * all_rel[rel] * all_ent[obj]).sum(dim=-1) + self.bias[obj]

	def rel_logits(self, sub, obj):
		"""Convenience: encode then score all relations (B, num_rel)."""
		all_ent, all_rel = self.objective_encode()
		return self.rel_score_all(sub, obj, all_ent, all_rel)

class CompGCN_LinkPredict(CompGCNBase):
	"""Relation-agnostic **edge-existence** model: a binary classifier over
	ordered (s, o) pairs answering only "should *some* target relation connect
	these two nodes?" — not which one.

	This is triple classification (Socher et al. 2013; Wang et al. 2014) reduced
	to the existence question, in the inductive setting (GraIL, Teru et al. 2020):
	the CompGCN encoder still uses the typed relations for message passing, but
	the decoder is a single edge-scoring head over [h_s ; h_o ; h_s ⊙ h_o]
	(the standard concat + Hadamard edge feature for link prediction). Trained
	with BCE against type-compatible hard negatives. Decoupled from the
	relation-prediction objective so the two tasks don't share a decoder.
	"""
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		d = self.p.embed_dim
		self.n_pair = int(getattr(self.p, 'n_pair_feats', 0))   # extra scalar pair features (name overlap)
		self.edge_mlp = torch.nn.Sequential(
			torch.nn.Linear(3 * d + self.n_pair, d),
			torch.nn.ReLU(),
			torch.nn.Dropout(self.p.hid_drop),
			torch.nn.Linear(d, 1),
		)

	def edge_score(self, sub, obj, all_ent, pair_feats=None):
		"""Edge-existence logit per ordered pair (s, o) (pre-sigmoid). Concatenation
		keeps direction; the Hadamard term adds an interaction; pair_feats appends
		precomputed scalars (e.g. name overlap)."""
		hs, ho = all_ent[sub], all_ent[obj]
		parts  = [hs, ho, hs * ho]
		if pair_feats is not None:
			parts.append(pair_feats)
		return self.edge_mlp(torch.cat(parts, dim=-1)).squeeze(-1)

	def forward(self, sub, obj, pair_feats=None):
		"""Training forward: encode the train graph (with dropout), score pairs."""
		all_ent, _ = self.encode_train(self.drop, self.drop)
		return self.edge_score(sub, obj, all_ent, pair_feats)


class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

	def decode(self, sub, rel, all_ent, all_rel):
		"""Scores (sub, rel) against all entities using cached encoder output."""
		stk_inp	= self.concat(all_ent[sub], all_rel[rel])
		x	= self.bn0(stk_inp)
		x	= self.m_conv1(x)
		x	= self.bn1(x)
		x	= F.relu(x)
		x	= self.feature_drop(x)
		x	= x.view(-1, self.flat_sz)
		x	= self.fc(x)
		x	= self.hidden_drop2(x)
		x	= self.bn2(x)
		x	= F.relu(x)
		x	= torch.mm(x, all_ent.transpose(1, 0)) + self.bias   # broadcast over batch
		return torch.sigmoid(x)
