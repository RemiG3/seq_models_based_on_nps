import torch
import torch.nn as nn
import numpy as np
import math


class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

# Custom Argmax with a one-hot vector encoding
class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        op = torch.zeros(input.size()).to(input.device)
        op.scatter_(1, idx[:, None], 1)
        ctx.save_for_backward(op)
        return op

    @staticmethod
    def backward(ctx, grad_output):
        op, = ctx.saved_tensors
        grad_input = grad_output * op
        return grad_input

# Weight with 2 or 3 dimensions
class Weight(nn.Module):
    def __init__(self, din, dout, num_blocks=None, bias=True, init=None, use_bias_only=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_bias_only = use_bias_only
        if init is None:
            init = 1. / math.sqrt(dout)
        if num_blocks is None:
            self.weight = nn.Linear(din, dout, bias=bias)
            bias = False
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_blocks, din, dout).uniform_(-init, init))
        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(num_blocks, dout).uniform_(-init, init))
        else:
            self.bias = None
        assert (use_bias_only and bias) or (not use_bias_only)
    
    def forward(self, x):
        if self.use_bias_only:
            return x + self.bias
        if self.num_blocks is None:
            return self.weight(x)
        else:
            x = x.permute(1, 0, 2)
            x = torch.bmm(x, self.weight)
            x = x.permute(1, 0, 2)
            if self.bias is not None:
                x = x + self.bias
            return x


# Custom MLP composed of Weights (useful to define multiple MLPs in one block)
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, hidden_dim=32, bias=True, use_bias_only=False):
        super().__init__()
        if use_bias_only:
            self.w1 = Weight(in_dim, out_dim, num_blocks, bias=bias, use_bias_only=True)
            self.w2 = Weight(in_dim, out_dim, num_blocks, bias=bias, use_bias_only=True)
        else:
            self.w1 = Weight(in_dim, hidden_dim, num_blocks, bias=bias, use_bias_only=False)
            self.w2 = Weight(hidden_dim, out_dim, num_blocks, bias=bias, use_bias_only=False)
        # self.layer_norm = LayerNorm()

    def forward(self, x):
        return self.w2( torch.relu( self.w1(x) ) )

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# Concatenate informations for the keys or query for the attentions
class Concatenator:
    def __init__(self, dic_trans, prefix_name=''):
        self.list_trans = list(dic_trans.values())
        self.name = prefix_name + ' - ' + ' '.join(list(dic_trans.keys()))
    
    def __call__(self, s, s_p, s_c, r, pos, it, Svst, Rvst):
        bs = s.size(0)
        comp = [s, s_p, s_c, r, pos, it, Svst, Rvst]
        trans_values = []
        for transformation in self.list_trans:
            trans_values.append( transformation(comp, bs) )
        res = trans_values[0]
        for i in range(1, len(trans_values)):
            if (res.size(1) > trans_values[i].size(1)):
                res = torch.cat((res, trans_values[i].repeat(1, res.size(1), 1)), dim=2)
            elif (res.size(1) < trans_values[i].size(1)):
                res = torch.cat((res.repeat(1, trans_values[i].size(1), 1), trans_values[i]), dim=2)
            else:
                res = torch.cat((res, trans_values[i]), dim=2)
        return res

# Class that defined the model based on NPS
class Model(nn.Module):
    def __init__(self, encoder, decoder, n_slots, n_classes, n_iter, n_rules, rule_mlp_hidden_dim, transform_input_callback,
                 stddev_noise_r=0., stddev_noise_c=0., use_entropy=False, replace_mode = False, #dropout=0.1, 
                 dim_attn_r=32, dim_attn_c=16, dim_key_r=None, dim_query_r=None, dim_key_c=None, dim_query_c=None,
                 tau_train_r=1., tau_eval_r=1., tau_train_c=1., tau_eval_c=1., hard_gs_train_r=True, hard_gs_eval_r=True,
                 query_r=None, keys_c=None, query_c=None, input_mlp=None, hard_gs_train_c=True, hard_gs_eval_c=True,
                 s_dim=8, r_dim=12, pos_dim=12, sp_dim=None, use_null_rule=False, tpr_order=0, reversed_archtiecture=False, simplified=False,
                 scores_constraint_pos_visited=False, use_pos_onehot=False, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super().__init__()
        self.n_slots = n_slots
        self.n_rules = n_rules
        self.n_iter = n_iter
        self.s_dim = s_dim
        self.sp_dim = (s_dim*r_dim*pos_dim if (tpr_order == 3) else s_dim*pos_dim if (tpr_order == 2) else s_dim) if sp_dim is None else sp_dim
        self.pos_dim = pos_dim
        self.r_dim = r_dim
        self.use_entropy = use_entropy
        self.use_null_rule = use_null_rule
        self.query_r = None if(query_r == []) else query_r
        self.keys_c = None if(keys_c == []) else keys_c
        self.query_c = None if(query_c == []) else query_c
        self.input_mlp = None if(input_mlp == []) else input_mlp
        self.stddev_noise_r = stddev_noise_r
        self.stddev_noise_c = stddev_noise_c
        self.tau_train_r, self.tau_eval_r = tau_train_r, tau_eval_r
        self.tau_train_c, self.tau_eval_c = tau_train_c, tau_eval_c
        self.hard_gs_train_r, self.hard_gs_eval_r = hard_gs_train_r, hard_gs_eval_r
        self.hard_gs_train_c, self.hard_gs_eval_c = hard_gs_train_c, hard_gs_eval_c
        self.use_mlp_rules = (self.input_mlp is not None)
        self.tpr_order = tpr_order
        self.use_tpr = ((tpr_order != 0) and (tpr_order is not None))
        self.use_pos = (pos_dim > 0)
        self.scores_constraint_pos_visited = scores_constraint_pos_visited
        self.use_pos_onehot = use_pos_onehot
        self.replace_mode = replace_mode
        self.simplified = simplified
        self.reversed_architecture = reversed_archtiecture
        self.transform_input_callback = transform_input_callback
        self.n_classes = n_classes
        self.device = device

        self.encoder = encoder
        self.decoder = decoder
        if self.decoder is not None:
            self.autoencoder = nn.Sequential(
                self.encoder,
                self.decoder
            )
        
        self.reset_arrays_record()
        self._init_concatenators()
        self._init_classifier()
        
        # Rule embeddings (conditions to apply rules)
        self.r = nn.Parameter(torch.randn(1, n_rules, r_dim).to(device))

        # Model with attention to compute scores useful for primary slot and rule selection
        self.w_k = Weight(self.wk_r_in_dim, dim_attn_r, dim_key_r, bias=True)
        self.w_q = Weight(self.wq_r_in_dim, dim_attn_r, dim_query_r, bias=True)
        self.d_r = dim_attn_r
        
        # Model with attention to compute the contextual slot
        self.w_k_tild = Weight(self.wk_c_in_dim, dim_attn_c, dim_key_c, bias=True)
        self.w_q_tild = Weight(self.wq_c_in_dim, dim_attn_c, dim_query_c, bias=True)
        self.d_c = dim_attn_c
        
        # self.add_wq_tild = Weight(self.wq_c_in_dim, self.wq_c_in_dim, bias=True)

        # Contextual slots embeddings (position of contextual slots)
        if self.use_pos:
            if self.use_pos_onehot:
                self.pos = torch.eye(n_slots).to(device)
            else:
                self.pos = nn.Embedding(n_slots, pos_dim)
        else:
            self.pos = Struct(**{'weight': None})

        # Model MLP for rule application
        if self.use_mlp_rules:
            self.r_mlp = MLP(in_dim=self.rule_mlp_in_dim, out_dim=self.s_dim if(sp_dim is None) else self.sp_dim, num_blocks=n_rules, hidden_dim=rule_mlp_hidden_dim,
                            bias=True, use_bias_only=False)
        
        assert (use_pos_onehot and (self.pos_dim == n_slots)) or (not use_pos_onehot)
        assert (self.simplified and not self.reversed_architecture) or (self.reversed_architecture and not self.simplified) or (not self.reversed_architecture and not self.simplified)
        assert self.tpr_order in [None, 0, 2, 3]
        assert (sp_dim is not None) or ((sp_dim is None) and (self.use_tpr))
    

    def get_activations(self):
        # return array(s) of arrays [r, sc]
        return np.array(self.activations)
    
    def get_rule_probabilities(self):
        return self.rule_probabilities

    def reset_arrays_record(self):
        self.activations = []
        self.rule_probabilities = []
    
    def _rule_selection(self, slots, s_p, s_c=None, attn_c=None, it=None):
        entropy = 0
        bs = slots.size(0)

        # Get rule embeddings (keys)
        rules = self.r.repeat(bs, 1, 1)

        # Get the query (or queries) for the attention
        if self.query_r is not None:
            if self.use_pos_onehot:
                quer_r_pos = self.pos.unsqueeze(0).repeat(bs, 1, 1) if(attn_c is None) else attn_c.unsqueeze(1)
            else:
                quer_r_pos = self.pos.weight.unsqueeze(0).repeat(bs, 1, 1) if(attn_c is None) else self.pos(torch.argmax(attn_c, dim=1)).unsqueeze(1)
            query_r = self.concatenator_wq_r(slots, s_p, None if(s_c is None) else s_c.unsqueeze(1), None, quer_r_pos, it, self.slots_visited, self.rules_visited)
        else:
            query_r = slots
        n_queries = query_r.size(1)
        
        # Compute scores for rules
        scores_r = torch.bmm(self.w_k(rules), self.w_q(query_r).permute(0, 2, 1)) / math.sqrt(self.d_r)
        if self.training and (self.stddev_noise_r > 0.):
            scores_r += torch.autograd.Variable(torch.randn(scores_r.size()).to(self.device) * self.stddev_noise_r)
        
        # Compute attention
        if self.training:
            # Training: Gumbel Softmax
            attn_r = torch.nn.functional.gumbel_softmax(scores_r.reshape(bs, -1), dim=1, tau=self.tau_train_r, hard=self.hard_gs_train_r)
            if self.use_entropy:
                probs = torch.softmax(scores_r.reshape(bs, -1), dim=1)
                probs = torch.mean(probs * attn_r, dim=0)
                entropy = self.n_rules * n_queries * torch.sum(probs)
            attn_r = attn_r.reshape(bs, self.n_rules, n_queries)
            self.rule_probabilities.append(attn_r.detach().clone())
        else:
            # Evaluation: Argmax
            if self.tau_eval_r == 0:
                attn_r = ArgMax.apply(scores_r.reshape(bs, -1)).reshape(bs, self.n_rules, n_queries)
            else:
                attn_r = torch.nn.functional.gumbel_softmax(scores_r.reshape(bs, -1), dim=1, tau=self.tau_eval_r, hard=self.hard_gs_eval_r).reshape(bs, self.n_rules, n_queries)
            self.rule_probabilities.append(torch.softmax(scores_r.permute(0, 2, 1).float().reshape(bs, -1), dim=1).reshape(bs, n_queries, self.n_rules).detach().clone())
        
        # Select contextual slot if simplified architecture
        if self.simplified:
            mask_c = attn_r.sum(dim=1)
            new__s_c = (slots * mask_c.unsqueeze(-1)).sum(dim=1)
        else:
            new__s_c = None
        
        # Select rule
        mask_r = attn_r.sum(dim=2).unsqueeze(-1)
        rule = (rules * mask_r).sum(dim=1).unsqueeze(1)

        return rule, new__s_c, attn_r, entropy
    
    def _contextual_slot_selection(self, slots, s_p, rule, it):
        # Get the keys for the attention
        if self.keys_c is not None:
            #keys_c = self.concatenator_wk_c(slots, s_p, None, rule, self.pos.weight)
            keys_c = self.concatenator_wk_c(slots, s_p, None, rule, self.pos.unsqueeze(0).repeat(slots.size(0), 1, 1) if self.use_pos_onehot else self.pos.weight.unsqueeze(0).repeat(slots.size(0), 1, 1), it, self.slots_visited, self.rules_visited)
        else:
            keys_c = slots
        
        # Get the query for the attention
        if self.query_c is not None:
            query_c = self.concatenator_wq_c(slots, s_p, None, rule, None, it, self.slots_visited, self.rules_visited)
        else:
            query_c = s_p
        
        # query_c = self.layer_norm( torch.relu( self.add_wq_tild(query_c) ) )
        # query_c = torch.relu( self.add_wq_tild(query_c) )
        
        # Compute scores for slots
        scores_c = torch.bmm(self.w_q_tild(query_c), self.w_k_tild(keys_c).permute(0, 2, 1)) / math.sqrt(self.d_c)
        if self.training and (self.stddev_noise_c > 0.):
            scores_c += torch.autograd.Variable(torch.randn(scores_c.size()).to(self.device) * self.stddev_noise_c)
        
        if self.scores_constraint_pos_visited:
            #scores_c = (scores_c - self.slots_visited).squeeze(1)
            scores_c = ((scores_c / (scores_c.max() - scores_c.min())) * torch.abs(self.slots_visited - 1.)).squeeze(1)
        else:
            scores_c = scores_c.squeeze(1)
        
        # Compute attention
        if self.training: # Training: Gumbel Softmax
            attn_c = torch.nn.functional.gumbel_softmax(scores_c, dim=1, hard=self.hard_gs_train_c, tau=self.tau_train_c)
        else: # Evaluation: Argmax
            if self.tau_eval_c == 0:
                attn_c = ArgMax.apply(scores_c)
            else:
                attn_c = torch.nn.functional.gumbel_softmax(scores_c, dim=1, hard=self.hard_gs_eval_c, tau=self.tau_eval_c)
        s_c = (slots * attn_c.unsqueeze(-1)).sum(dim=1)
        
        return s_c, attn_c

    def _one_iteration_forward(self, slots, s_p=None, it=None):
        entropy = 0
        
        if self.reversed_architecture:
            ### STEP 2 ###
            s_c, attn_c = self._contextual_slot_selection(slots, s_p, rule=None, it=it)
            ##############

            ### STEP 3 ###
            rule, _, attn_r, entropy = self._rule_selection(slots, s_p, s_c=s_c, attn_c=attn_c, it=it)
            ##############
        else:
            ### STEP 2 ###
            rule, s_c, attn_r, entropy = self._rule_selection(slots, s_p, s_c=None, attn_c=None, it=it)
            ##############

            ### STEP 3 ###
            if not self.simplified:
                s_c, attn_c = self._contextual_slot_selection(slots, s_p, rule=rule, it=it)
            else:
                attn_c = attn_r.sum(dim=1)
            ##############
        
        self.slots_visited = self.slots_visited + attn_c.unsqueeze(1)
        self.rules_visited = self.rules_visited + attn_r.unsqueeze(1).squeeze(-1)

        #### For printing (using argmax)
        rule_print = torch.argmax(attn_r.sum(dim=2).detach(), dim=1).detach().cpu().numpy()
        sc_print = torch.argmax(attn_c.detach(), dim=1).detach().cpu().numpy()
        self.activations.append([rule_print, sc_print])

        ### STEP 4 ###
        # Get contextual slot's embeddings
        if self.use_pos:
            if self.use_pos_onehot:
                c_pos = attn_c
            else:
                c_pos = self.pos(torch.argmax(attn_c, dim=1))
        else:
            c_pos = None

        # Applying MLP rules if specified
        if self.use_mlp_rules:
            r_mlp_input = self.concatenator_in_mlp(slots, s_p, s_c.unsqueeze(1), rule, c_pos.unsqueeze(1), it, self.slots_visited, self.rules_visited).repeat(1, self.n_rules, 1)
            result = (self.r_mlp(r_mlp_input) * attn_r).sum(dim=1)
            
            # Set to zero the last dimension for the rule Null
            if self.use_null_rule:
                result *= torch.where(attn_r[:,-1] == 1., 0., 1.).unsqueeze(1)
        else:
            result = s_c
        
        # Applying TPR order if specified
        if self.use_tpr:
            if self.tpr_order == 2:
                result = torch.einsum("bi, bj -> bij", result, c_pos)
            else: # Should be TPR 3
                result = torch.einsum("bi, bj, bk -> bijk", c_pos, result, rule.squeeze(1))
        ##############
        
        if self.use_entropy:
            return result, entropy
        else:
            return result
    

    def forward(self, X):
        ### STEP 1 ###
        # Encoding in slots
        if self.transform_input_callback is not None:
            slots = self.transform_input_callback(X, self.encoder)
        else:
            slots = self.encoder(X)
        s_p = torch.zeros((X.size(0), 1, self.sp_dim)).to(self.device)
        self.entropy = 0
        self.slots_visited = torch.zeros((X.size(0), 1, self.n_slots)).to(self.device)
        self.rules_visited = torch.zeros((X.size(0), 1, self.n_rules)).to(self.device)
        ##############

        for it in range(self.n_iter):
            # One iteration forward
            if self.use_entropy:
                rule_output, entropy = self._one_iteration_forward(slots, s_p, it)
                self.entropy += entropy
            else:
                rule_output = self._one_iteration_forward(slots, s_p, it)

            # Update of the primary slots
            if self.replace_mode:
                s_p = rule_output.reshape((X.size(0), 1, self.sp_dim))
            else:
                s_p = s_p + rule_output.reshape((X.size(0), 1, self.sp_dim))
        
        ### STEP 5 ###
        # Decode the output
        return self.classifier(s_p.squeeze(1))
        ##############

    def _init_concatenators(self):
        dic_single_select = {
            'S':    self._s_selection,
            'Sc':   self._sc_selection,
            'Sp':   self._sp_selection,
            'R':    self._r_selection,
            'POS':  self._pos_selection,
            'Yhat': self._pred_selection,
            'Iter': self._iter_selection,
            'Svst':  self._svst_selection,
            'Rvst':  self._rvst_selection,
        }

        dic_dim = {
            'Sp': self.sp_dim,
            'S': self.s_dim,
            'Sc': self.s_dim,
            'R': self.r_dim,
            'POS': self.pos_dim,
            'Yhat': self.n_classes,
            'Iter': self.n_iter,
            'Svst':  self.n_slots,
            'Rvst':  self.n_rules,
        }

        def get_subset_dict(dic, list_keys):
            return {k: dic[k] for k in list_keys}

        if self.query_r is not None:
            # self.concatenator_wq_r = Concatenator(get_subset_dict(dic_keys_attn_trans, self.query_r), prefix_name='WK_PS')
            self.concatenator_wq_r = Concatenator(get_subset_dict(dic_single_select, self.query_r), prefix_name='WK_PS')
            self.wq_r_in_dim = sum([dic_dim[comp] for comp in self.query_r])
        else:
            self.wq_r_in_dim = self.s_dim
        
        self.wk_r_in_dim = dic_dim['R']
        
        if self.keys_c is not None:
            # self.concatenator_wk_c = Concatenator(get_subset_dict(dic_keys_attn_trans, self.keys_c), prefix_name='WK_C')
            self.concatenator_wk_c = Concatenator(get_subset_dict(dic_single_select, self.keys_c), prefix_name='WK_C')
            self.wk_c_in_dim = sum([dic_dim[comp] for comp in self.keys_c])
        else:
            self.wk_c_in_dim = self.s_dim
        
        if self.query_c is not None:
            self.concatenator_wq_c = Concatenator(get_subset_dict(dic_single_select, self.query_c), prefix_name='WQ_C')
            self.wq_c_in_dim = sum([dic_dim[comp] for comp in self.query_c])
        else:
            self.wq_c_in_dim = dic_dim['Sp']
        
        if self.input_mlp is not None:
            self.concatenator_in_mlp = Concatenator(get_subset_dict(dic_single_select, self.input_mlp), prefix_name='IN_MLP')
            self.rule_mlp_in_dim = sum([dic_dim[comp] for comp in self.input_mlp])
        else:
            self.rule_mlp_in_dim = 0
    
    
    def _s_selection(self, comp, bs):
        return comp[0]

    def _sp_selection(self, comp, bs):
        return comp[1]

    def _sc_selection(self, comp, bs):
        return comp[2]

    def _r_selection(self, comp, bs):
        return comp[3]

    def _pos_selection(self, comp, bs):
        return comp[4]
    
    def _pred_selection(self, comp, bs):
        return self.classifier(comp[1].squeeze(1)).detach().unsqueeze(1)
    
    def _iter_selection(self, comp, bs):
        return torch.nn.functional.one_hot(torch.LongTensor([comp[5]]).unsqueeze(0).repeat(bs, 1), num_classes=self.n_iter).float().to(self.device)
    
    def _svst_selection(self, comp, bs):
        return comp[6]
    
    def _rvst_selection(self, comp, bs):
        return comp[7]
    

    def _init_classifier(self):
        self.classifier = nn.Sequential(nn.Linear(self.sp_dim, self.n_classes, bias=True),
                                        nn.Softmax(dim=1)).to(self.device)