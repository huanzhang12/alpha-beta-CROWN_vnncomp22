#### preprocessor-hint: private-file
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA.patches import Patches
from auto_LiRPA.bound_ops import *
import itertools
import random
import copy

class Cutter:
    def __init__(self, solver, A=None, x=None, number_cuts=50, fix_intermediate_bounds=False, device='cuda'):
        self.solver = solver
        self.update_net(solver.net)
        # FIXME a['/input.1'] may not always be the input name
        self.A = [a['/input.1'] for a in A.values()] if A is not None else None  
        self.x = x
        self.number_cuts = number_cuts
        self.fix_intermediate_bounds = fix_intermediate_bounds
        self.device = device
        self.unstable_idx_list, self.lower, self.upper = [], [], []
        self.lAs, self.uAs, self.lbs, self.ubs = [], [], [], []
        self.log_interval = 10
        self.beta_init = 0
        # Cutter should keep track of cuts
        self.cuts = []
        self.cut_module = None
        self.cut_timestamp = -1
        self.num_relus = len(self.relus)

    def update_net(self, net):
        self.net = net
        self.relus = net.relus

    def construct_cut_module(self, start_nodes=None, use_patches_cut=False):
        num_cuts = len(self.cuts)

        self.net.cut_used = False
        for m in self.relus:
            m.masked_beta = None
            # m.split_beta_used: True if any of this layer node is used in the new split constraints
            m.split_beta_used = False
            # m.history_beta_used: True if any of this layer node is used in the history constraints
            m.history_beta_used = False
            # if any of the current/history split constraints are single node constraint,
            m.single_beta_used = False
            # There are nodes used in cuts in this layer if cut_used=True
            m.cut_used = False
            m.relu_coeffs = m.arelu_coeffs = m.pre_coeffs = None

        self.start_nodes = start_nodes = [
            relu_layer.inputs[0] for relu_layer in self.net.relus] if not self.fix_intermediate_bounds else []
        start_nodes.append(self.net[self.net.output_name[0]])

        # init cut_module
        self.cut_module = cut_module = CutModule(self.relus)
        # maximum layer of the neuron in each constr
        # -1 is the input layer, 0 means the layer corresponding to the first relu layer
        max_layer_idx_in_constr = []
        cut_used = False
        self.pre_layers, self.relu_layers, self.arelu_layers = [], [], []
        self.use_x_cuts = False
        for cut_idx, ci in enumerate(self.cuts):
            # c = ci["c"] # 1: >0; -1: <0
            all_decisions = (ci["x_decision"] + ci["relu_decision"] +
                ci["arelu_decision"] + ci["pre_decision"])
            if len(all_decisions) > 0:
                max_layer_idx_in_constr.append(max([item[0] for item in all_decisions]))
            else:
                max_layer_idx_in_constr.append(-1)
            if len(ci["x_decision"]) > 0:
                cut_used = self.use_x_cuts = True
            self.relu_layers.extend([ self.relus[item[0]] for item in ci["relu_decision"]])
            self.arelu_layers.extend([ self.relus[item[0]] for item in ci["arelu_decision"]])
            self.pre_layers.extend([ self.relus[item[0]] for item in ci["pre_decision"]])
        self.pre_layers = set(self.pre_layers)
        self.relu_layers = set(self.relu_layers)
        self.arelu_layers = set(self.arelu_layers)
        for node in itertools.chain(self.relu_layers, self.arelu_layers, self.pre_layers):
            cut_used = node.cut_used = True
        if cut_used:
            self.net.cut_used = True

        # config active cuts
        print("all start nodes to check full crown or not:", start_nodes)
        print("use patches cut: ", use_patches_cut)
        cut_module.active_cuts = {}
        for node_idx, start_node in enumerate(start_nodes):
            active_cuts = []
            # use_patches_cut: disable to optimize intermediate layer that is in patches mode
            if not use_patches_cut and hasattr(start_node, "mode") and start_node.mode=="patches":
                print("skip cut beta crown opt for patches layer:", start_node)
                cut_module.active_cuts[start_node.name] = torch.tensor(active_cuts, device=self.device).long()
                continue
            if node_idx + 1 == len(start_nodes):
                cut_module.active_cuts[start_node.name] = torch.arange(num_cuts, device=self.device)
            else:
                for cut_idx in range(num_cuts):
                    # the constraints are active only if they have all constraint neurons before start node
                    # i.e., the max layer node in the constraint should be smaller than the current start node
                    if (max_layer_idx_in_constr[cut_idx] == -1 or 
                            start_nodes.index(self.relus[max_layer_idx_in_constr[cut_idx]].inputs[0]) < node_idx):
                        active_cuts.append(cut_idx)
                cut_module.active_cuts[start_node.name] = torch.tensor(active_cuts, device=self.device).long()

        if self.use_x_cuts:
            assert self.x.size(0) == 1  # do not have batch dimension.
            self.cut_module.x_coeffs = torch.zeros((num_cuts, self.x.numel()), device=self.device)
        #FIXME maybe just use self.relus
        for node in self.relu_layers:
            cut_module.relu_coeffs[node.name] = torch.zeros((num_cuts, node.flattened_nodes), device=self.device)
        for node in self.arelu_layers:
            cut_module.arelu_coeffs[node.name] = torch.zeros((num_cuts, node.flattened_nodes), device=self.device)
        for node in self.pre_layers:
            cut_module.pre_coeffs[node.name] = torch.zeros((num_cuts, node.flattened_nodes), device=self.device)

        self.update_cut_module()

        return cut_module

    def construct_beta(self, shapes):
        general_betas = {}
        start_node_shape = {}
        for i, node in enumerate(self.net.relus):
            start_node_shape[node.inputs[0].name] = shapes[i]
        start_node_shape[self.net.final_name] = shapes[-1]
        # current all start nodes, only final_name if crown-ibp
        # all_start_nodes = self.net.backward_from[self.net.optimizable_activations[0].name]
        # manually get all start nodes, self.net.backward_from might not include first pre-relu layer
        # all_start_nodes = [relu_layer.inputs[0] for relu_layer in nodes] + [self.net[self.net.output_name[0]]]
        for start_node in self.start_nodes:
            # each general_beta: 2 (lA, uA), spec (out_c, out_h, out_w), batch, num_cuts
            general_betas[start_node.name] = self.beta_init * torch.ones(
                (2, *start_node_shape[start_node.name][1:], 1, len(self.cuts)), device=self.device)
            general_betas[start_node.name] = general_betas[start_node.name].detach()
            general_betas[start_node.name].requires_grad = True
        self.cut_module.general_beta = general_betas
        self.net.cut_beta_params = []
        for start_node in self.start_nodes:
            self.net.cut_beta_params.append(self.cut_module.general_beta[start_node.name])

    def init_cut(self, c=1):
        return {
            'x_decision': [], 'x_coeffs': [], 'arelu_decision': [], 'arelu_coeffs': [],
            'relu_decision': [], 'relu_coeffs': [], 'pre_decision': [], 'pre_coeffs': [],
            'c': c
        }

    def update_cut_module(self):
        # config the cut constraints coeffs in cut_module
        # add cut into each relu layer
        cut_bias = []
        for cut_idx, ci in enumerate(self.cuts):
            c = ci["c"] # 1: >0; -1: <0
            for node, coeff in zip(ci["x_decision"], ci["x_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.x_coeffs[cut_idx, neuron_idx] += c * coeff
            for node, coeff in zip(ci["relu_decision"], ci["relu_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.relu_coeffs[self.relus[layer].name][cut_idx, neuron_idx] += c * coeff
            for node, coeff in zip(ci["arelu_decision"], ci["arelu_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.arelu_coeffs[self.relus[layer].name][cut_idx, neuron_idx] += c * coeff
            for node, coeff in zip(ci["pre_decision"], ci["pre_coeffs"]):
                layer, neuron_idx = node[0], node[1]
                self.cut_module.pre_coeffs[self.relus[layer].name][cut_idx, neuron_idx] += c * coeff
            ##### have to be careful, c * bias (c=1) if constraint > bias;
            cut_bias.append(c * ci["bias"])
            #FIXME bias should be on torch
        self.cut_module.cut_bias = torch.tensor(cut_bias, device=self.device)

    def update_cuts(self):
        pass

    def refine_cuts(self):
        pass


class OptimizedCutter(Cutter):
    """ Upper bounding planes with constrained x1, x2 """
    def __init__(self, solver, A, x, number_cuts=50, fix_intermediate_bounds=False, 
            device='cuda', opt=False, lr=0.02):
        super().__init__(solver, A, x, number_cuts, fix_intermediate_bounds, device)

        self.diff = ((self.x.ptb.x_U - self.x.ptb.x_L) / 2.0).flatten()
        self.center = ((self.x.ptb.x_U + self.x.ptb.x_L) / 2.0).flatten()        

        for i in range(self.num_relus):
            lA, uA = self.A[i]['lA'], self.A[i]['uA']
            l = self.relus[i].inputs[0].lower.flatten()
            u = self.relus[i].inputs[0].upper.flatten()
            if isinstance(lA, Patches):
                lA = lA.to_matrix(self.x.shape)
                uA = uA.to_matrix(self.x.shape)
                # convert unstable_idx from n-d list to flatten idx
                output = self.relus[i].inputs[0].lower
                idx = torch.arange(output.numel()).reshape(output[0].shape)
                _unstable_idx = self.A[i]['unstable_idx']
                _unstable_idx = idx[_unstable_idx]
                unstable_idx = _unstable_idx.to(device)
            else:
                # dense A matrix, we generate unstable_idx here
                unstable_idx = torch.logical_and(l < 0, u > 0).float()
                unstable_idx = (unstable_idx == 1).nonzero().flatten()

            num_unstable = len(unstable_idx)
            # FIXME this shouldn't be an assertion
            assert num_unstable > 0, print('No unstable neuron in this layer')
            print(f'Unstable neurons in {i}th layer: {num_unstable}/{l.numel()}')

            self.unstable_idx_list.append(unstable_idx)
            self.lower.append(l[unstable_idx])
            self.upper.append(u[unstable_idx])

            assert lA.shape[0] == 1
            lA = lA.reshape(lA.shape[1], -1)
            uA = uA.reshape(uA.shape[1], -1)
            lb = self.A[i]['lbias'].flatten()
            ub = self.A[i]['ubias'].flatten()

            # unstable_idx was not used for lA and uA
            if lA.shape[0] != len(unstable_idx):
                lA = lA[unstable_idx]; uA = uA[unstable_idx]
                lb = lb[unstable_idx]; ub = ub[unstable_idx]

            self.lAs.append(lA); self.uAs.append(uA)
            self.lbs.append(lb); self.ubs.append(ub)

        self.num_starts = self.num_relus
        self.opt = opt
        self.log_interval = 10
        self.params = []
        self.beta_init = 0
        self.fix_beta = False
        self.sim_threshold = 1e-2 
        self.preact_threshold = 1e-2
        self.lr = lr 
        # optimize directions for cutting pre-activation domains
        self.optimize_cut_pre_coeffs = True 
        self.min_cut_pre_coeff = 1e-2

    def concretize(self, A, sign=-1):
        # TODO utilize concretize() function from model.x.ptb
        return A.matmul(self.center) + sign * A.abs().matmul(self.diff)

    def get_parameters(self):
        # TODO configurable
        return {'params': self.params, 'lr': self.lr }

    def update_cut_module(self):
        """
        Cutting lines:
        L1 <= x1 + x2 <= U1
        L2 <= x1 - x2 <= U2

        8 points of intersection on the borders:
        b1: x1+x2=l1,    x2=lower(x2),   b1=l1-lower(x2)
        b2: x1-x2=u2,    x2=lower(x2),   b2=u2+lower(x2)
        b3: x1-x2=u2,    x1=upper(x1),   b3=upper(x1)-u2
        b4: x1+x2=u1,    x1=upper(x1),   b4=u1-upper(x1)
        b5: x1+x2=u1,    x2=upper(x2),   b5=u1-upper(x2)
        b6: x1-x2=l2,    x2=upper(x2),   b6=upper(x2)+l2
        b7: x1-x2=l2,    x1=lower(x1),   b7=lower(x1)-l2
        b8: x1+x2=l1,    x1=lower(x1),   b8=l1-lower(x1)

        lower1              upper1
        upper2  b6   b5     upper2
        b7                  b4
        b8                  b3
        lower1  b1  b2      upper1 
        lower2              lower2
        """
        #   x1 + coeffs[0] * x2     lower bound
        #   x1 - coeffs[1] * x2     upper bound
        #   x1 + coeffs[2] * x2     upper bound
        #   x1 - coeffs[3] * x2     lower bound

        lower1, upper1, lower2, upper2 = self.lower1, self.upper1, self.lower2, self.upper2

        def compute_intersections():
            lower1, upper1, lower2, upper2 = self.lower1, self.upper1, self.lower2, self.upper2

            L1 = (self.concretize(self.lA1 + self.cut_pre_coeffs[:, 0:1] * self.lA2, sign=-1) 
                + self.lb1 + self.cut_pre_coeffs[:, 0] * self.lb2)
            U1 = (self.concretize(self.uA1 + self.cut_pre_coeffs[:, 2:3] * self.uA2, sign=1) 
                + self.ub1 + self.cut_pre_coeffs[:, 2] * self.ub2)
            L2 = (self.concretize(self.lA1 - self.cut_pre_coeffs[:, 3:4] * self.uA2, sign=-1) 
                + self.lb1 - self.cut_pre_coeffs[:, 3] * self.ub2)
            U2 = (self.concretize(self.uA1 - self.cut_pre_coeffs[:, 1:2] * self.lA2, sign=1) 
                + self.ub1 - self.cut_pre_coeffs[:, 1] * self.lb2)

            b1 = L1 - self.cut_pre_coeffs[:, 0] * lower2
            b2 = U2 + self.cut_pre_coeffs[:, 1] * lower2
            b3 = (upper1 - U2) / self.cut_pre_coeffs[:, 1] 
            b4 = (U1 - upper1) / self.cut_pre_coeffs[:, 2]
            b5 = U1 - self.cut_pre_coeffs[:, 2] * upper2
            b6 = L2 + self.cut_pre_coeffs[:, 3] * upper2 
            b7 = (lower1 - L2) / self.cut_pre_coeffs[:, 3]
            b8 = (L1 - lower1) / self.cut_pre_coeffs[:, 0]

            eps_ = 1e-4
            assert eps_ * 2 < self.preact_threshold
            b1 = torch.max(torch.min(b1, b2 - eps_), lower1)
            b2 = torch.min(torch.max(b2, b1 + eps_), upper1)
            b3 = torch.max(torch.min(b3, b4 - eps_), lower2)
            b4 = torch.min(torch.max(b4, b3 + eps_), upper2)
            b6 = torch.max(torch.min(b6, b5 - eps_), lower1)
            b5 = torch.min(torch.max(b5, b6 + eps_), upper1)
            b8 = torch.max(torch.min(b8, b7 - eps_), lower2)
            b7 = torch.min(torch.max(b7, b8 + eps_), upper2)

            return b1, b2, b3, b4, b5, b6, b7, b8

        if False and self.count_updates == 0:
            # optimize self.cut_pre_coeffs with cut_ratio here
            lr = 0.2
            opt = torch.optim.Adam([self.cut_pre_coeffs], lr=lr)
            last_obj = -1
            for t in range(500):
                b1, b2, b3, b4, b5, b6, b7, b8 = compute_intersections()

                cut_ratio = 0.5 * (
                    (b6 - lower1) * (upper2 - b7) 
                    + (upper1 - b5) * (upper2 - b4)
                    + (b1 - lower1) * (b8 - lower2)
                    + (upper1 - b2) * (b3 - lower2)) / ((upper2 - lower2) * (upper1 - lower1)) 
                obj = cut_ratio.mean()
                if obj < last_obj:
                    lr *= 0.95
                    opt.param_groups[0]['lr'] = lr
                if lr < 0.01:
                    break
                last_obj = obj
                print(f'iteration {t}: cut ratio {obj:.5f}, lr {lr:.5f}')
                opt.zero_grad()
                (-obj).backward()
                opt.step()
        
        b1, b2, b3, b4, b5, b6, b7, b8 = compute_intersections()

        if True:
            eps_b = 1e-6
            cnt_to_improve = ((b1 - upper1 > -eps_b).sum() + (lower1 - b2 > -eps_b).sum()
                + (b3 - upper2 > -eps_b).sum() + (lower2 - b4 > -eps_b).sum()
                + (lower1 - b5 > -eps_b).sum() + (b6 - upper1 > -eps_b).sum()
                + (lower2 - b7 > -eps_b).sum() + (b8 - upper2 > -eps_b).sum())
            cnt_b_cross = ( (b1 - b2 > -eps_b).sum() + (b3 - b4 > -eps_b).sum()
                + (b6 - b5 > -eps_b).sum() + (b8 - b7 > -eps_b).sum())
            try:
                assert cnt_to_improve == 0 and cnt_b_cross == 0
            except:        
                import pdb; pdb.set_trace()

        # a * relu(x1) + b * relu(x2)
        a = self.relu_coeffs[:, 0]
        b = self.relu_coeffs[:, 1]

        def func(x1, x2, a, b):
            return a * F.relu(x1) + b * F.relu(x2)            

        # Take three points to form a bounding plane
        
        # Point 1: (b1, lower2) or (lower1, b8)
        # The point with the minimum function value
        v1 = func(b1, lower2, a, b)
        v2 = func(lower1, b8, a, b)        
        take_v1 = v1 <= v2
        x1_1 = torch.where(take_v1, b1, lower1)
        x2_1 = torch.where(take_v1, lower2, b8)
        x3_1 = func(x1_1, x2_1, a, b)

        # Point 2: (upper1, b4) or (b5, upper2)
        # The point with the maximum function value
        v3 = func(upper1, b4, a, b)
        v4 = func(b5, upper2, a, b)
        take_v3 = v3 >= v4
        x1_2 = torch.where(take_v3, upper1, b5)
        x2_2 = torch.where(take_v3, b4, upper2)
        x3_2 = func(x1_2, x2_2, a, b)

        # One of the other four points as the third point on the bounding plane
        x_3 = [(lower1, b7), (b6, upper2), (b2, lower2), (upper1, b3)]
        func_others = [func(x1_3, x2_3, a, b) for x1_3, x2_3 in x_3]

        pre_coeffs, relu_coeffs, biases = [], [], []
        for k, (x1_3, x2_3) in enumerate(x_3):
            x3_3 = func_others[k]
            # normal
            a_ = (x2_2 - x2_1) * (x3_3 - x3_1) - (x2_3 - x2_1) * (x3_2 - x3_1)
            b_ = (x3_2 - x3_1) * (x1_3 - x1_1) - (x3_3 - x3_1) * (x1_2 - x1_1)
            c_ = (x1_2 - x1_1) * (x2_3 - x2_1) - (x1_3 - x1_1) * (x2_2 - x2_1)
            assert c_.abs().min() > 1e-6
            # plane: a_ (x1 - x1_1) + b_ (x2 - x2_1) + c_ (z - z1) = 0
            #       (a_/c_) (x1 - x1_1) + (b_/c_) (x2 - x2_1) + (z - z1) = 0
            # =>    z = x3_1 - (a_/c_) (x1 - x1_1) - (b_/c_) (x2 - x2_1)
            # =>    z = x3_1 + (a_/c_) x1_1 + (b_/c_) x2_1 - (a_/c_) x1 - (b_/c_) x2
            #       p = -a_/c_, q = -b_/c_, 
            #       r = x3_1 - p x1_1 - q x2_1
            p = -a_ / c_
            q = -b_ / c_
            r = - p * x1_1 - q * x2_1 + x3_1 

            extra_bias = torch.zeros_like(r)
            for kk, (x1_4, x2_4) in enumerate(x_3):
                if k != kk:
                    extra_bias = torch.max(extra_bias,
                        func_others[kk] - (p * x1_4 + q * x2_4 + r))
            r += extra_bias

            # TODO if extra_bias is not close to zero, the cutting plane is probably bad

            relu_coeffs.append(torch.cat([a, b]).view(2, self.num_base_cuts).t())
            pre_coeffs.append(torch.cat([p, q]).view(2, self.num_base_cuts).t())
            biases.append(r)

        pre_coeffs = torch.cat(pre_coeffs, dim=0).view(4, self.num_base_cuts, 2)
        relu_coeffs = torch.cat(relu_coeffs, dim=0).view(4, self.num_base_cuts, 2)
        biases = torch.cat(biases, dim=0).view(4, self.num_base_cuts)

        # apply masked coefficients to cut_module
        for i in range(self.num_starts):
            if self.cut_module.relu_coeffs[self.relus[i].name] is not None:
                relu_coeffs_ = torch.zeros_like(
                    self.cut_module.relu_coeffs[self.relus[i].name]).view(4, self.num_base_cuts, -1)
                relu_coeffs_.scatter_(
                    src=relu_coeffs * self.relu_mask[i].unsqueeze(0), index=self.relu_decisions[i], dim=-1)
                self.cut_module.relu_coeffs[self.relus[i].name] = -relu_coeffs_.view(4 * self.num_base_cuts, -1)

            if self.cut_module.pre_coeffs[self.relus[i].name] is not None:
                pre_coeffs_ = torch.zeros_like(
                    self.cut_module.pre_coeffs[self.relus[i].name]).view(4, self.num_base_cuts, -1)
                pre_coeffs_.scatter_(
                    src=pre_coeffs * self.pre_mask[i].unsqueeze(0), index=self.pre_decisions[i], dim=-1)
                self.cut_module.pre_coeffs[self.relus[i].name] = pre_coeffs_.view(4 * self.num_base_cuts, -1)

        self.cut_module.cut_bias = -biases.flatten()

    def update_cuts(self):
        if not self.opt:
            return

        self.count_updates += 1
        self.relu_coeffs.data = self.relu_coeffs.clamp(min=0)
        self.cut_pre_coeffs.data = self.cut_pre_coeffs.clamp(min=self.min_cut_pre_coeff)
        self.update_cut_module()
        
    @torch.no_grad()
    def select_cuts(self):
        # Add cut decisions first. Calculate coefficients later
        candidates = []
        for i in range(self.num_relus):
            num_unstable = len(self.unstable_idx_list[i])
            lA = self.lAs[i]
            sim_lA = torch.matmul(lA, lA.t())
            sim_score = torch.triu(sim_lA, diagonal=1).abs()
            sorted_sim, sorted_idx = torch.sort(sim_score.flatten().abs(), 0, descending=True)
            i1 = sorted_idx // num_unstable
            i2 = sorted_idx % num_unstable
            is_candidate = torch.logical_and(
                sorted_sim >= self.sim_threshold,
                torch.logical_and(self.upper[i][i1] - self.lower[i][i1] >= self.preact_threshold,
                                self.upper[i][i2] - self.lower[i][i2] >= self.preact_threshold),
            )
            sorted_sim = sorted_sim[is_candidate]
            i1, i2 = i1[is_candidate], i2[is_candidate]
            for j in range(sorted_sim.shape[0]):
                candidates.append({
                    'score': sorted_sim[j], 
                    'layer': i, 
                    'idx1': i1[j].item(), 
                    'idx2': i2[j].item(),
                })  

        print(f'{len(candidates)} candidates in total')

        # Select candidates:
        if True:
            # Randomly select
            # random.seed(0)
            # candidates.reverse() # add cuts in later layers?
            random.shuffle(candidates)
            candidates = candidates[:self.number_cuts//4]
        else:
            # Sort by scores
            candidates = sorted(candidates, key=lambda x:x['score'], reverse=True)[:self.number_cuts//4]
        self.candidates = candidates
        print(f'{len(candidates)} candidates selected')

        return candidates

    @torch.no_grad()
    def add_cuts(self, candidates=None):
        start_time = time.time()
        if candidates is None:
            candidates = self.select_cuts()
        print('Time for selecting cuts:', time.time() - start_time)

        self.num_base_cuts = len(candidates)

        self.lA1 = torch.empty(self.num_base_cuts, self.x.numel()).to(self.x)
        self.uA1 = torch.empty(self.num_base_cuts, self.x.numel()).to(self.x)
        self.lA2 = torch.empty(self.num_base_cuts, self.x.numel()).to(self.x)
        self.uA2 = torch.empty(self.num_base_cuts, self.x.numel()).to(self.x)
        self.lb1 = torch.empty(self.num_base_cuts).to(self.x)
        self.ub1 = torch.empty(self.num_base_cuts).to(self.x)
        self.lb2 = torch.empty(self.num_base_cuts).to(self.x)
        self.ub2 = torch.empty(self.num_base_cuts).to(self.x)
        self.lower1 = torch.empty(self.num_base_cuts).to(self.x)
        self.upper1 = torch.empty(self.num_base_cuts).to(self.x)
        self.lower2 = torch.empty(self.num_base_cuts).to(self.x)
        self.upper2 = torch.empty(self.num_base_cuts).to(self.x)

        for i in range(self.num_relus):
            idx_cuts = [j for j in range(self.num_base_cuts) if candidates[j]['layer'] == i]
            idx1 = [cand['idx1'] for cand in candidates if cand['layer'] == i]
            idx2 = [cand['idx2'] for cand in candidates if cand['layer'] == i]
            self.lA1[idx_cuts] = self.lAs[i][idx1]; self.lA2[idx_cuts] = self.lAs[i][idx2]
            self.uA1[idx_cuts] = self.uAs[i][idx1]; self.uA2[idx_cuts] = self.uAs[i][idx2]
            self.lb1[idx_cuts] = self.lbs[i][idx1]; self.lb2[idx_cuts] = self.lbs[i][idx2]
            self.ub1[idx_cuts] = self.ubs[i][idx1]; self.ub2[idx_cuts] = self.ubs[i][idx2]    
            self.lower1[idx_cuts] = self.lower[i][idx1]; self.lower2[idx_cuts] = self.lower[i][idx2]
            self.upper1[idx_cuts] = self.upper[i][idx1]; self.upper2[idx_cuts] = self.upper[i][idx2]

        self.cuts = []
        for j, cand in enumerate(candidates):
            layer, idx1, idx2 = cand['layer'], cand['idx1'], cand['idx2']
            idx1_full = self.unstable_idx_list[layer][idx1].item()
            idx2_full = self.unstable_idx_list[layer][idx2].item()
            decision = [[layer, idx1_full], [layer, idx2_full] ]   
            cut = self.init_cut()
            cut['relu_decision'] = decision
            cut['relu_decision_unstable'] = [idx1, idx2]
            cut['relu_coeffs'] = [1, 1] # initialized value
            cut['pre_decision'] = decision 
            cut['pre_decision_unstable'] = [idx1, idx2]
            cut['pre_coeffs'] = [0, 0] # dummy 
            self.cuts.append(cut)
        for k in range(3):
            for i in range(self.num_base_cuts):
                self.cuts.append(copy.deepcopy(self.cuts[i]))

        assert self.num_relus == self.num_starts
        for cut in self.cuts:
            assert len(cut['arelu_decision']) == len(cut['x_decision']) == 0
            assert cut['c'] == 1
        self.max_cut_neurons = max([len(cut["relu_decision"]) for cut in self.cuts])
        assert self.max_cut_neurons == 2

        # Coefficients for cutting pre-activation domains
        # For each cut: 
        #   x1 + coeffs[0] * x2     lower bound
        #   x1 - coeffs[1] * x2     upper bound
        #   x1 + coeffs[2] * x2     upper bound
        #   x1 - coeffs[3] * x2     lower bound
        # Initially, set all to 1
        self.cut_pre_coeffs = torch.ones(self.num_base_cuts, 4, device=self.device)
        self.active = torch.ones(self.num_base_cuts, device=self.device)

        self.relu_coeffs = torch.zeros(self.num_base_cuts, self.max_cut_neurons, device=self.device)
        self.relu_mask = torch.zeros(self.num_starts, self.num_base_cuts, 1, device=self.device)
        self.relu_decisions = torch.zeros(self.num_starts, 4, self.num_base_cuts, self.max_cut_neurons, 
                            dtype=torch.long, device=self.device)
        self.relu_decisions_unstable = torch.zeros_like(self.relu_decisions).to(torch.long)
        for i in range(self.num_base_cuts):
            cut = self.cuts[i]
            layer = cut['relu_decision'][0][0]
            self.relu_mask[layer, i] = 1
            for j, (layer, idx) in enumerate(cut['relu_decision']):
                self.relu_coeffs[i, j] = cut["relu_coeffs"][j]
                self.relu_decisions[layer, :, i, j] = idx
                self.relu_decisions_unstable[layer, :, i, j] = cut["relu_decision_unstable"][j]

        self.pre_coeffs = torch.zeros_like(self.relu_coeffs) # FIXME dummy, maybe not needed
        self.pre_mask = self.relu_mask
        self.pre_decisions = self.relu_decisions
        self.pre_decisions_unstable = self.relu_decisions_unstable

        if self.opt:
            self.relu_coeffs = nn.Parameter(self.relu_coeffs)
            self.params.append(self.relu_coeffs)
            if self.optimize_cut_pre_coeffs:
                self.cut_pre_coeffs = nn.Parameter(self.cut_pre_coeffs)
                self.params.append(self.cut_pre_coeffs)
            self.count_updates = 0

        return self.cuts

    def refine_cuts(self, split_history=None):
        print('Refining cuts...')
        num_cuts = len(self.cuts)
        assert len(self.cut_module.general_beta) == 1
        beta = self.cut_module.general_beta[self.net.final_name]
        assert beta.ndim == 4
        beta_sum = beta.sum(dim=[0,1,2])

        selected_idx = beta_sum >= 1e-2
        self.cuts = [self.cuts[i] for i in range(len(self.cuts)) if selected_idx[i]]
        for key, value in self.cut_module.relu_coeffs.items():
            if value is not None:
                self.cut_module.relu_coeffs[key] = value[selected_idx].detach()
        for key, value in self.cut_module.pre_coeffs.items():
            if value is not None:
                self.cut_module.pre_coeffs[key] = value[selected_idx].detach()
        self.cut_module.cut_bias = self.cut_module.cut_bias[selected_idx].detach()
        #FIXME we only need 1 instead of 2 for the first dimension
        self.cut_module.general_beta[self.net.final_name] = beta[:, :, :, selected_idx].detach()
        self.net.cut_beta_params = []
        selected_idx_list = [idx for idx in range(num_cuts) if selected_idx[idx]]
        for i, start_node in enumerate(self.start_nodes):
            self.net.cut_beta_params.append(self.cut_module.general_beta[start_node.name])
            active_cuts = [selected_idx_list.index(idx)
                 for idx in self.cut_module.active_cuts[start_node.name] if selected_idx[idx]] 
            self.cut_module.active_cuts[start_node.name] = torch.tensor(active_cuts, device=self.device)
        if split_history:
            assert isinstance(split_history['general_betas'], torch.Tensor)
            split_history['general_betas'] = split_history['general_betas'][:, :, :, selected_idx]

        self.refined = True
        # For now, we will fix the cuts at this time and no longer optimize them in bab.
        # Do we need to optimize them in bab?
        self.opt = False   
        # TODO clip further if there are still too many cuts left

        print(f'{len(self.cuts)} cuts out of {num_cuts} retained')


# Temporary
OptimizedCutter2 = OptimizedCutter
