#### preprocessor-hint: private-file
import torch
import numpy as np
import arguments
from auto_LiRPA.patches import Patches, patches_to_matrix

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

PLOT_DIST = False

@torch.no_grad()
def add_input_cuts(model, A, number_cuts=10, device='cuda', mask=None):
    diff = ((model.x.ptb.x_U - model.x.ptb.x_L) / 2.0).flatten()
    center = ((model.x.ptb.x_U + model.x.ptb.x_L) / 2.0).flatten()

    input_node_name = model.net.input_name[0]
    unstable_idx_list = []
    lower, upper, this_lA, this_uA, this_lb, this_ub = [], [], [], [], [], []
    for layer_idx in range(len(model.net.relus)):
    # for layer_idx in [0]:

        # merge all A matrix and bias across layers
        if isinstance(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'], Patches):
            # convert patches to matrix
            _A = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA']
            _A = patches_to_matrix(_A.patches, model.x.shape, _A.stride, _A.padding, _A.output_shape, _A.unstable_idx)
            A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'] = _A

            _A = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA']
            _A = patches_to_matrix(_A.patches, model.x.shape, _A.stride, _A.padding, _A.output_shape, _A.unstable_idx)
            A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA'] = _A

            # convert unstable_idx from n-d list to flatten idx
            output = model.net.relus[layer_idx].inputs[0].lower
            # print(output.shape)
            idx = torch.arange(output.numel()).reshape(output[0].shape)
            _unstable_idx = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['unstable_idx']
            _unstable_idx = idx[_unstable_idx]
            A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['unstable_idx'] = _unstable_idx.to(
                device)

        # unstable_idx = (mask[layer_idx][0] == 1).nonzero().flatten()
        unstable_idx = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['unstable_idx'].to(torch.long)
        if unstable_idx is None:
            # dense A matrix, we generate unstable_idx here
            _lower = model.net.relus[layer_idx].inputs[0].lower.flatten()
            _upper = model.net.relus[layer_idx].inputs[0].upper.flatten()
            unstable_idx = torch.logical_and(_lower < 0, _upper > 0).float()
            unstable_idx = (unstable_idx == 1).nonzero().flatten()

        assert len(unstable_idx) > 0, print('No unstable neuron in this layer')
        print('number of unstable neurons in {}th layer: {}'.format(layer_idx, len(unstable_idx)))
        unstable_idx_list.append(unstable_idx)

        lower.append(model.net.relus[layer_idx].inputs[0].lower.flatten()[unstable_idx])
        upper.append(model.net.relus[layer_idx].inputs[0].upper.flatten()[unstable_idx])

        _this_lA = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'][0].flatten(1).to(torch.get_default_dtype())
        assert _this_lA.shape[0] >= len(unstable_idx), print('The length of A matrix should be larger than unstable_idx')
        if _this_lA.shape[0] > len(unstable_idx):
            this_lA.append(_this_lA[unstable_idx])
            this_lb.append(
                A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lbias'][:, unstable_idx])
            this_uA.append(
                A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA'][0].flatten(1)[unstable_idx].to(torch.get_default_dtype()))
            this_ub.append(
                A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['ubias'][:, unstable_idx])
        else:
            this_lA.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'][0].flatten(1).to(torch.get_default_dtype()))
            this_lb.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lbias'])

            this_uA.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA'][0].flatten(1).to(torch.get_default_dtype()))
            this_ub.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['ubias'])

        # assert this_lA[-1].shape[0] == len(unstable_idx), print('for sparse_intermediate_bounds only')

        lb = this_lA[-1].matmul(center) - this_lA[-1].abs().matmul(diff) + this_lb[-1]
        ub = this_uA[-1].matmul(center) + this_uA[-1].abs().matmul(diff) + this_ub[-1]
        print('max diff of lb:', (lower[-1] - lb).abs().max())
        print('max diff of ub:', (upper[-1] - ub).abs().max())

    lower = torch.cat(lower)
    upper = torch.cat(upper)
    this_lA = torch.cat(this_lA)
    this_lb = torch.cat(this_lb, dim=-1)
    this_uA = torch.cat(this_uA)
    this_ub = torch.cat(this_ub, dim=-1)
    unstable_idx = torch.cat(unstable_idx_list)
    unstable_idx_list_len = torch.cumsum(torch.tensor([len(i) for i in unstable_idx_list], device=device), dim=0)

    lower = (this_lA.matmul(center) - this_lA.abs().matmul(diff) + this_lb).flatten()
    upper = (this_uA.matmul(center) + this_uA.abs().matmul(diff) + this_ub).flatten()
    assert (lower < 0).all() and (upper > 0).all()  # make sure we only have unstable neurons here
    # assert torch.allclose(lower, lb)

    last_A = A[model.net.output_name[0]][input_node_name]['lA'].to(torch.get_default_dtype())
    # extract the slice of last A matrix with the worst lower bound's label
    worst_idx = model.net[model.net.output_name[0]].lower.argmin()
    last_A = last_A[0, worst_idx].view(1, -1)
    last_b = A[model.net.output_name[0]][input_node_name]['lbias'][:, worst_idx]
    assert last_A.matmul(center) - last_A.abs().matmul(diff) + last_b < 0

    # cstr = constraint
    cstr_lower, cstr_upper = pair_heuristic(model, this_lA, this_lb, this_uA, this_ub, center, diff, top_k=arguments.Config["bab"]["cut"]["topk_cuts_in_filter"],
                                            device=device, return_cstr_only=True)

    ############# lower bound < 0 #############
    cstr_lA = this_lA[cstr_lower]

    # bias terms of constraints
    cstr_lb = this_lb[0, cstr_lower]

    # filter by checking whether (cstr_lA * primal_last_l + cstr_lb < 0)
    primal_last_l = get_primal(model, last_A, -1)
    check_unsat = (cstr_lA.matmul(primal_last_l.t()).t() + cstr_lb > 0).nonzero()
    print('{} of {} are filtered out by check primal already satisfied (cx + d < 0)'.format(len(cstr_lA) - len(check_unsat), len(cstr_lA)))
    cstr_lower = cstr_lower[check_unsat[:, 1]]  # only keep unsat indexes
    # extract again
    cstr_lA = this_lA[cstr_lower]
    cstr_lb = this_lb[0, cstr_lower]

    # get the primal for upper bound
    primal_u = get_primal(model, cstr_lA, +1)
    # max value that Ax + b can get
    best_c_l = cstr_lA.matmul(primal_u.t()).diag() + cstr_lb - 1

    ############# upper bound > 0 #############
    cstr_uA = this_uA[cstr_upper]

    # bias terms of constraints
    cstr_ub = this_ub[0, cstr_upper]

    # filter by checking whether (cstr_uA * primal_last_l + cstr_ub > 0)
    check_unsat = (cstr_uA.matmul(primal_last_l.t()).t() + cstr_ub < 0).nonzero()
    print('{} of {} are filtered out by check primal already satisfied (cx + d > 0)'.format(len(cstr_uA) - len(check_unsat), len(cstr_uA)))
    cstr_upper = cstr_upper[check_unsat[:, 1]]  # only keep unsat indexes
    # extract again
    cstr_uA = this_uA[cstr_upper]
    cstr_ub = this_ub[0, cstr_upper]

    # get the primal for lower bound
    primal_l = get_primal(model, cstr_uA, -1)
    # min value that Ax + b can get
    best_c_u = cstr_uA.matmul(primal_l.t()).diag() + cstr_ub - 1

    x_decision = [[-1, i] for i in range(center.shape[0])]
    ret, cut_ret = [], []

    for i, idx in enumerate(cstr_lower):
        layer_idx = torch.searchsorted(unstable_idx_list_len, idx, right=True).item()
        ReLU_idx = unstable_idx[idx].item()
        best_C = best_c_l[i].item()
        c = cstr_lA[i]
        bias = this_lb[0, idx].item()
        cut_info = ['cx + d <= z + z * C with ReLU idx: [{}, {}], '.format(layer_idx, ReLU_idx)]
        if cut_info in ret:
            continue
        print(cut_info)
        ret.append(cut_info)
        cut_ret.append({"x_decision": x_decision, "x_coeffs": c.tolist(),
                        "relu_decision": [], "relu_coeffs": [],
                        "arelu_decision": [[layer_idx, ReLU_idx]],
                        "arelu_coeffs": [-1.0 - best_C], "pre_decision": [], "pre_coeffs": [], "bias": -bias, "c": -1})

    for i, idx in enumerate(cstr_upper):
        layer_idx = torch.searchsorted(unstable_idx_list_len, idx, right=True).item()
        ReLU_idx = unstable_idx[idx].item()
        best_C = best_c_u[i].item()
        c = cstr_uA[i]
        bias = -this_ub[0, idx].item() + 1 + best_C
        cut_info = ['cx + d >= 1 - z + (1 - z) * C with ReLU idx: [{}, {}], '.format(layer_idx, ReLU_idx)]
        if cut_info in ret:
            continue
        print(cut_info)
        ret.append(cut_info)
        cut_ret.append({"x_decision": x_decision, "x_coeffs": c.tolist(),
                        "relu_decision": [], "relu_coeffs": [],
                        "arelu_decision": [[layer_idx, ReLU_idx]],
                        "arelu_coeffs": [1.0 + best_C], "pre_decision": [], "pre_coeffs": [], "bias": bias, "c": 1})

    return cut_ret[:number_cuts]

@torch.no_grad()
def add_implied_cuts(model, A, number_cuts=10, device='cuda', mask=None, return_unsort=True):
    # example of adding implied bound cuts on the second relu layer

    diff = ((model.x.ptb.x_U - model.x.ptb.x_L) / 2.0).flatten()
    center = ((model.x.ptb.x_U + model.x.ptb.x_L) / 2.0).flatten()

    input_node_name = model.net.input_name[0]
    unstable_idx_list = []
    lower, upper, this_lA, this_uA, this_lb, this_ub = [], [], [], [], [], []
    for layer_idx in range(len(model.net.relus)):

        # merge all A matrix and bias across layers
        if isinstance(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'], Patches):
            # convert patches to matrix
            _A = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA']
            _A = patches_to_matrix(_A.patches, model.x.shape, _A.stride, _A.padding, _A.output_shape, _A.unstable_idx)
            A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'] = _A

            _A = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA']
            _A = patches_to_matrix(_A.patches, model.x.shape, _A.stride, _A.padding, _A.output_shape, _A.unstable_idx)
            A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA'] = _A

            # convert unstable_idx from n-d list to flatten idx
            output = model.net.relus[layer_idx].inputs[0].lower
            # print(output.shape)
            idx = torch.arange(output.numel()).reshape(output[0].shape)
            _unstable_idx = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['unstable_idx']
            _unstable_idx = idx[_unstable_idx]
            A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['unstable_idx'] = _unstable_idx.to(device)

        # unstable_idx = (mask[layer_idx][0] == 1).nonzero().flatten()
        unstable_idx = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['unstable_idx']
        if unstable_idx is None or type(unstable_idx) == tuple:
            # dense A matrix, we generate unstable_idx here
            _lower = model.net.relus[layer_idx].inputs[0].lower.flatten()
            _upper = model.net.relus[layer_idx].inputs[0].upper.flatten()
            unstable_idx = torch.logical_and(_lower < 0, _upper > 0).float()
            unstable_idx = (unstable_idx == 1).nonzero().flatten()

        assert len(unstable_idx) > 0, print('No unstable neuron in this layer')
        print('number of unstable neurons in {}th layer: {}'.format(layer_idx, len(unstable_idx)))
        unstable_idx_list.append(unstable_idx)

        lower.append(model.net.relus[layer_idx].inputs[0].lower.flatten()[unstable_idx])
        upper.append(model.net.relus[layer_idx].inputs[0].upper.flatten()[unstable_idx])

        _this_lA = A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'][0].flatten(1)
        assert _this_lA.shape[0] >= len(unstable_idx), print('The length of A matrix should be larger than unstable_idx')
        if _this_lA.shape[0] > len(unstable_idx):
            this_lA.append(_this_lA[unstable_idx])
            this_lb.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lbias'][:, unstable_idx])
            this_uA.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA'][0].flatten(1)[unstable_idx])
            this_ub.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['ubias'][:, unstable_idx])
        else:
            this_lA.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lA'][0].flatten(1))
            this_lb.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['lbias'])

            this_uA.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['uA'][0].flatten(1))
            this_ub.append(A[model.net.relus[layer_idx].inputs[0].name][input_node_name]['ubias'])

        # assert this_lA[-1].shape[0] == len(unstable_idx), print('for sparse_intermediate_bounds only')

        lb = this_lA[-1].matmul(center) - this_lA[-1].abs().matmul(diff) + this_lb[-1]
        ub = this_uA[-1].matmul(center) + this_uA[-1].abs().matmul(diff) + this_ub[-1]
        print('max diff of lb:', (lower[-1] - lb).abs().max())
        print('max diff of ub:', (upper[-1] - ub).abs().max())

    # lower = torch.cat(lower)
    # upper = torch.cat(upper)
    this_lA = torch.cat(this_lA)
    this_lb = torch.cat(this_lb, dim=-1)
    this_uA = torch.cat(this_uA)
    this_ub = torch.cat(this_ub, dim=-1)
    unstable_idx = torch.cat(unstable_idx_list)
    unstable_idx_list_len = np.cumsum([len(i) for i in unstable_idx_list])

    lower = (this_lA.matmul(center) - this_lA.abs().matmul(diff) + this_lb).flatten()
    upper = (this_uA.matmul(center) + this_uA.abs().matmul(diff) + this_ub).flatten()
    # assert (lower < 0).all() and (upper > 0).all()  # make sure we only have unstable neurons here
    # assert torch.allclose(lower, lb)

    cstr_lower, cstr_upper, obj_A_l2l_idx, obj_A_u2l_idx, obj_A_l2u_idx, obj_A_u2u_idx, obj_len_l2l, obj_len_l2u, obj_len_u2l, obj_len_u2u = \
        pair_heuristic(model, this_lA, this_lb, this_uA, this_ub, center, diff, top_k=arguments.Config["bab"]["cut"]["topk_cuts_in_filter"], device=device)

    # # the index of lower bounds constraint
    # cstr_lower = torch.tensor(list(pair_dict['lower'].keys()))
    # # the index of upper bounds constraint
    # cstr_upper = torch.tensor(list(pair_dict['upper'].keys()))

    loss1_diff, loss2_diff, loss3_diff, loss4_diff = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    loss1_diff_flipped, loss2_diff_flipped, loss3_diff_flipped, loss4_diff_flipped = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    beta1_total = beta1_unimproved = beta2_total = beta2_unimproved = beta3_total = beta3_unimproved = beta4_total = beta4_unimproved = None

    if len(cstr_lower) > 0:
        # A matrix of optimize lower bounds s.t. lower bound < 0
        cstr_A_l2l = torch.repeat_interleave(this_lA[cstr_lower], obj_len_l2l, dim=0)
        # bias terms of constraints
        cstr_b_l2l = torch.repeat_interleave(this_lb[0, cstr_lower], obj_len_l2l, dim=0)
        # A matrix of objective
        # obj_A_l2l_idx = torch.cat([i[0] for i in pair_dict['lower'].values()])
        obj_A_l2l = this_lA[obj_A_l2l_idx]
        # bias term of objective
        obj_b_l2l = this_lb[0, obj_A_l2l_idx]
        # Improve lower bound, with another neuron's lower bound <= 0 (inactive).
        best_obj1, best_beta1 = fast_solve(cstr_A_l2l, obj_A_l2l, cstr_b_l2l + cstr_A_l2l.matmul(center), diff)
        beta1_total = best_beta1.size(0)
        beta1_unimproved = beta1_total - best_beta1.count_nonzero().item()
        del best_beta1, cstr_A_l2l, cstr_b_l2l
        # add missed constant terms Ax_0 + b
        best_obj1 += obj_A_l2l.matmul(center) + obj_b_l2l
        loss1_diff = best_obj1 - lower[obj_A_l2l_idx]
        loss1_diff_flipped = loss1_diff.clone()
        loss1_diff_flipped[best_obj1 < 0] = 0  # we filter out lb < 0, only keep (always active) cases
        del obj_A_l2l

        # A matrix of optimize upper bounds s.t. lower bound < 0
        cstr_A_u2l = torch.repeat_interleave(this_lA[cstr_lower], obj_len_u2l, dim=0)
        # bias terms of constraints
        cstr_b_u2l = torch.repeat_interleave(this_lb[0, cstr_lower], obj_len_u2l, dim=0)
        # A matrix of objective
        # obj_A_u2l_idx = torch.cat([i[1] for i in pair_dict['lower'].values()])
        obj_A_u2l = this_uA[obj_A_u2l_idx]
        # bias term of objective
        obj_b_u2l = this_ub[0, obj_A_u2l_idx]
        # Improve upper bound, with another neuron's lower bound <= 0 (inactive).
        best_obj2, best_beta2 = fast_solve(cstr_A_u2l, -obj_A_u2l, cstr_b_u2l + cstr_A_u2l.matmul(center), diff)
        beta2_total = best_beta2.size(0)
        beta2_unimproved = beta2_total - best_beta2.count_nonzero().item()
        del best_beta2, cstr_A_u2l, cstr_b_u2l
        # add missed constant terms Ax_0 + b
        # we changed min to max, so the true objective are negative
        best_obj2 = -best_obj2 + obj_A_u2l.matmul(center) + obj_b_u2l
        loss2_diff = upper[obj_A_u2l_idx] - best_obj2
        loss2_diff_flipped = loss2_diff.clone()
        loss2_diff_flipped[best_obj2 > 0] = 0  # we filter out ub > 0, only keep (always inactive) cases
        del obj_A_u2l

    if len(cstr_upper) > 0:
        # A matrix of optimize upper bounds s.t. upper bound > 0
        cstr_A_l2u = - torch.repeat_interleave(this_uA[cstr_upper], obj_len_l2u, dim=0)  # Negative.
        # bias terms of constraints
        cstr_b_l2u = - torch.repeat_interleave(this_ub[0, cstr_upper], obj_len_l2u, dim=0)
        # A matrix of objective
        # obj_A_l2u_idx = torch.cat([i[0] for i in pair_dict['upper'].values()])
        obj_A_l2u = this_lA[obj_A_l2u_idx]
        # bias term of objective
        obj_b_l2u = this_lb[0, obj_A_l2u_idx]
        del this_lA, this_lb
        # Improve lower bound, with another neuron's upper bound >= 0 (active).
        best_obj3, best_beta3 = fast_solve(cstr_A_l2u, obj_A_l2u, cstr_b_l2u + cstr_A_l2u.matmul(center), diff)
        beta3_total = best_beta3.size(0)
        beta3_unimproved = beta3_total - best_beta3.count_nonzero().item()
        del best_beta3, cstr_A_l2u, cstr_b_l2u
        # add missed constant terms Ax_0 + b
        best_obj3 += obj_A_l2u.matmul(center) + obj_b_l2u
        loss3_diff = best_obj3 - lower[obj_A_l2u_idx]
        loss3_diff_flipped = loss3_diff.clone()
        loss3_diff_flipped[best_obj3 < 0] = 0  # we filter out lb < 0, only keep (always active) cases
        del obj_A_l2u

        # A matrix of optimize upper bounds s.t. upper bound > 0
        cstr_A_u2u = - torch.repeat_interleave(this_uA[cstr_upper], obj_len_u2u, dim=0)  # Negative.
        # bias terms of constraints
        cstr_b_u2u = - torch.repeat_interleave(this_ub[0, cstr_upper], obj_len_u2u, dim=0)
        # A matrix of objective
        # obj_A_u2u_idx = torch.cat([i[1] for i in pair_dict['upper'].values()])
        obj_A_u2u = this_uA[obj_A_u2u_idx]
        # bias term of objective
        obj_b_u2u = this_ub[0, obj_A_u2u_idx]
        del this_uA, this_ub
        # Improve upper bound, with another neuron's upper bound >= 0 (active).
        best_obj4, best_beta4 = fast_solve(cstr_A_u2u, -obj_A_u2u, cstr_b_u2u + cstr_A_u2u.matmul(center), diff)
        beta4_total = best_beta4.size(0)
        beta4_unimproved = beta4_total - best_beta4.count_nonzero().item()
        del best_beta4, cstr_A_u2u, cstr_b_u2u
        # add missed constant terms Ax_0 + b
        # we changed min to max, so the true objective are negative
        best_obj4 = -best_obj4 + obj_A_u2u.matmul(center) + obj_b_u2u
        loss4_diff = upper[obj_A_u2u_idx] - best_obj4
        loss4_diff_flipped = loss4_diff.clone()
        loss4_diff_flipped[best_obj4 > 0] = 0  # we filter out ub > 0, only keep (always inactive) cases
        del obj_A_u2u

    # warp-up results
    # some obj and cstr are actually same A in u2l and l2u case, so we exclude them
    print('Results in unimproved/total: l2l: {}/{}, u2l: {}/{}, l2u: {}/{}, u2u: {}/{}'.format(beta1_unimproved, beta1_total,
           beta2_unimproved, beta2_total-len(cstr_lower), beta3_unimproved, beta3_total-len(cstr_upper), beta4_unimproved, beta4_total))

    if return_unsort:
        unstable_idx_list_len = torch.tensor(unstable_idx_list_len, device=device)

        def find_idx(cstr_idx, obj_len, obj_idx, best_obj, loss_diff, obj_lower=True, threshold=1e-3):
            cstr_neuron_idx = unstable_idx[cstr_idx].repeat_interleave(obj_len)
            cstr_layer_idx = torch.searchsorted(unstable_idx_list_len, cstr_idx, right=True).repeat_interleave(obj_len)
            obj_neuron_idx = unstable_idx[obj_idx]
            obj_layer_idx = torch.searchsorted(unstable_idx_list_len, obj_idx, right=True)
            if obj_lower:
                flip_idx = torch.where(torch.logical_and(best_obj > 0, (best_obj - loss_diff) < 0))[0]
            else:
                flip_idx = torch.where(torch.logical_and(best_obj < 0, (best_obj + loss_diff) > 0))[0]
            improve_idx = torch.where(loss_diff > threshold)[0]

            flipped_ret = torch.stack([obj_layer_idx[flip_idx], obj_neuron_idx[flip_idx], cstr_layer_idx[flip_idx], cstr_neuron_idx[flip_idx], best_obj[flip_idx]])
            improved_ret = torch.stack([obj_layer_idx[improve_idx], obj_neuron_idx[improve_idx], cstr_layer_idx[improve_idx], cstr_neuron_idx[improve_idx], best_obj[improve_idx]])

            # remove obj and cstr are from the same A matrix
            self_pair = (torch.logical_and(flipped_ret[0] == flipped_ret[2], flipped_ret[1] == flipped_ret[3]) == 1).nonzero().squeeze()
            if len(self_pair) > 0:
                idx = torch.arange(flipped_ret.shape[1], device=device)
                combined = torch.cat((idx, self_pair))
                uniques, counts = combined.unique(return_counts=True)
                difference = uniques[counts == 1]
                flipped_ret = flipped_ret[:, difference]

            self_pair = (torch.logical_and(improved_ret[0] == improved_ret[2], improved_ret[1] == improved_ret[3]) == 1).nonzero().squeeze()
            if len(self_pair) > 0:
                idx = torch.arange(improved_ret.shape[1], device=device)
                combined = torch.cat((idx, self_pair))
                uniques, counts = combined.unique(return_counts=True)
                difference = uniques[counts == 1]
                improved_ret = improved_ret[:, difference]

            return {'flipped': flipped_ret, 'improved': improved_ret}

        final_ret = {}
        final_ret['lb improved s.t. lb < 0'] = find_idx(cstr_lower, obj_len_l2l, obj_A_l2l_idx, best_obj1, loss1_diff, obj_lower=True)
        final_ret['ub improved s.t. lb < 0'] = find_idx(cstr_lower, obj_len_u2l, obj_A_u2l_idx, best_obj2, loss2_diff, obj_lower=False)
        final_ret['lb improved s.t. ub > 0'] = find_idx(cstr_upper, obj_len_l2u, obj_A_l2u_idx, best_obj3, loss3_diff, obj_lower=True)
        final_ret['ub improved s.t. ub > 0'] = find_idx(cstr_upper, obj_len_u2u, obj_A_u2u_idx, best_obj4, loss4_diff, obj_lower=False)

        return final_ret

    unstable_idx_list_len = unstable_idx_list_len.cpu().numpy()

    sorted_improve_flipped = torch.sort(torch.cat([loss1_diff_flipped, loss3_diff_flipped, loss2_diff_flipped, loss4_diff_flipped]), descending=True)
    sorted_improve_all = torch.sort(torch.cat([loss1_diff, loss3_diff, loss2_diff, loss4_diff]), descending=True)
    score_length = torch.cumsum(torch.tensor([len(loss1_diff), len(loss3_diff), len(loss2_diff), len(loss4_diff)], device=device), dim=0)

    if PLOT_DIST:
        import matplotlib.pyplot as plt
        loss1_improve = (-loss1_diff/lower[obj_A_l2l_idx]).clamp(0, 1)
        loss2_improve = (loss2_diff/upper[obj_A_u2l_idx]).clamp(0, 1)
        loss3_improve = (-loss3_diff/lower[obj_A_l2u_idx]).clamp(0, 1)
        loss4_improve = (loss4_diff/upper[obj_A_u2u_idx]).clamp(0, 1)

        sorted_improve_ratio = torch.sort(torch.cat([loss1_improve, loss3_improve, loss2_improve, loss4_improve]), descending=True)
        data = sorted_improve_ratio.values.cpu().numpy()
        # data = sorted_improve_all.values.cpu().numpy()
        data = data[data > 0][:5000]
        plt.plot(data)  # density=False would make counts
        plt.grid(True)
        plt.ylabel('Improvement')
        # plt.xlabel('{}, mean {}'.format(arguments.Config['model']['name'], data.mean()))
        plt.xlabel('Sorted neuron index')
        # plt.legend()
        plt.annotate('Number of flipped neurons: {}'.format(sum(data == 1)), xy=(0.3, 0.95), xycoords='axes fraction')
        plt.savefig('plots/{}_mean_{}.pdf'.format(arguments.Config['model']['name'], data.mean()))
        plt.close()
        raise NotImplementedError

    group0 = obj_len_l2l.cumsum(0)
    group1 = obj_len_l2u.cumsum(0)
    group2 = obj_len_u2l.cumsum(0)
    group3 = obj_len_u2u.cumsum(0)

    ret = []
    cut_ret = []
    def find_cuts(sorted_list, flipped=False, this_number_cuts=50):
        for i, idx in enumerate(sorted_list.indices[:this_number_cuts]):
            group = torch.searchsorted(score_length, idx, right=True)
            if (not flipped and sorted_list.values[i] < 1e-3) or (flipped and sorted_list.values[i] <= 0):
                break

            # ret format: [explanation, layer_idx of obj, improved neuron idx, layer_idx of constraint, constraint neuron idx, final objective, improvement]
            if group == 0:
                real_idx = idx
                idx2 = obj_A_l2l_idx[real_idx]
                idx1 = torch.searchsorted(group0, real_idx, right=True)
                idx_cstr = unstable_idx[cstr_lower[idx1].item()].item()
                idx_obj = unstable_idx[idx2.item()].item()
                layer_cstr_idx = np.searchsorted(unstable_idx_list_len, cstr_lower[idx1].item(), side='right')
                layer_obj_idx = np.searchsorted(unstable_idx_list_len, idx2.item(), side='right')
                if layer_obj_idx == layer_cstr_idx and idx_obj == idx_cstr:
                    continue
                cut_info = ['lb idx: [{}, {}] improved from {:.5f} to {:.5f} s.t. lb with idx: [{}, {}] < 0'.format(
                    layer_obj_idx, idx_obj, lower[idx2].cpu().item(), best_obj1[real_idx].cpu().item(), layer_cstr_idx, idx_cstr)]
                if cut_info in ret:
                    continue  # some cuts in sorted_improve_flipped may already added in
                ret.append(cut_info)
                if flipped:
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [],  "arelu_decision": [[layer_obj_idx, idx_obj], [layer_cstr_idx, idx_cstr]],
                                    "arelu_coeffs": [1.0, 1.0], "pre_decision": [], "pre_coeffs": [], "bias": 1.0, "c": 1})
                else:
                    assert best_obj1[real_idx] < 0
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [],
                                    "arelu_decision": [[layer_cstr_idx, idx_cstr], [layer_cstr_idx, idx_cstr]],
                                    "arelu_coeffs": [best_obj1[real_idx].cpu().abs().item(), lower[idx2].cpu().item()],
                                    "pre_decision": [[layer_obj_idx, idx_obj]], "pre_coeffs": [-1.0],
                                    "bias": best_obj1[real_idx].cpu().abs().item(), "c": -1})

            elif group == 1:
                real_idx = idx - score_length[group - 1]
                idx2 = obj_A_l2u_idx[real_idx]
                idx1 = torch.searchsorted(group1, real_idx, right=True)
                idx_cstr = unstable_idx[cstr_upper[idx1].item()].item()
                idx_obj = unstable_idx[idx2.item()].item()
                layer_cstr_idx = np.searchsorted(unstable_idx_list_len, cstr_upper[idx1].item(), side='right')
                layer_obj_idx = np.searchsorted(unstable_idx_list_len, idx2.item(), side='right')
                if layer_obj_idx == layer_cstr_idx and idx_obj == idx_cstr:
                    continue
                cut_info = ['lb idx: [{}, {}] improved from {:.5f} to {:.5f} s.t. ub with idx: [{}, {}] > 0'.format(
                    layer_obj_idx, idx_obj, lower[idx2].cpu().item(), best_obj3[real_idx].cpu().item(), layer_cstr_idx, idx_cstr)]
                if cut_info in ret:
                    continue  # some cuts in sorted_improve_flipped may already added in
                ret.append(cut_info)
                if flipped:
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [], "arelu_decision": [[layer_cstr_idx, idx_cstr], [layer_obj_idx, idx_obj]],
                                "arelu_coeffs": [1.0, -1.0], "pre_decision": [], "pre_coeffs": [], "bias": 0, "c": -1})
                else:
                    assert best_obj3[real_idx] < 0
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [],
                                    "arelu_decision": [[layer_cstr_idx, idx_cstr], [layer_cstr_idx, idx_cstr]],
                                    "arelu_coeffs": [lower[idx2].cpu().abs().item(), best_obj3[real_idx].cpu().item()],
                                    "pre_decision": [[layer_obj_idx, idx_obj]], "pre_coeffs": [-1.0],
                                    "bias": lower[idx2].cpu().abs().item(), "c": -1})
            elif group == 2:
                real_idx = idx - score_length[group - 1]
                idx2 = obj_A_u2l_idx[real_idx]
                idx1 = torch.searchsorted(group2, real_idx, right=True)
                idx_cstr = unstable_idx[cstr_lower[idx1].item()].item()
                idx_obj = unstable_idx[idx2.item()].item()
                layer_cstr_idx = np.searchsorted(unstable_idx_list_len, cstr_lower[idx1].item(), side='right')
                layer_obj_idx = np.searchsorted(unstable_idx_list_len, idx2.item(), side='right')
                if layer_obj_idx == layer_cstr_idx and idx_obj == idx_cstr:
                    continue
                cut_info = ['ub idx: [{}, {}] improved from {:.5f} to {:.5f} s.t. lb with idx: [{}, {}] < 0'.format(
                    layer_obj_idx, idx_obj, upper[idx2].cpu().item(), best_obj2[real_idx].cpu().item(), layer_cstr_idx, idx_cstr)]
                if cut_info in ret:
                    continue  # some cuts in sorted_improve_flipped may already added in
                ret.append(cut_info)
                if flipped:
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [], "arelu_decision": [[layer_cstr_idx, idx_cstr], [layer_obj_idx, idx_obj]],
                                "arelu_coeffs": [1.0, -1.0], "pre_decision": [], "pre_coeffs": [], "bias": 0.0, "c": 1})
                else:
                    assert best_obj2[real_idx] > 0
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [],
                                    "arelu_decision": [[layer_cstr_idx, idx_cstr], [layer_cstr_idx, idx_cstr]],
                                    "arelu_coeffs": [-best_obj2[real_idx].cpu().item(), upper[idx2].cpu().item()],
                                    "pre_decision": [[layer_obj_idx, idx_obj]], "pre_coeffs": [-1.0],
                                    "bias": -best_obj2[real_idx].cpu().item(), "c": 1})

            elif group == 3:
                real_idx = idx - score_length[group - 1]
                idx2 = obj_A_u2u_idx[real_idx]
                idx1 = torch.searchsorted(group3, real_idx, right=True)
                idx_cstr = unstable_idx[cstr_upper[idx1].item()].item()
                idx_obj = unstable_idx[idx2.item()].item()
                layer_cstr_idx = np.searchsorted(unstable_idx_list_len, cstr_upper[idx1].item(), side='right')
                layer_obj_idx = np.searchsorted(unstable_idx_list_len, idx2.item(), side='right')
                if layer_obj_idx == layer_cstr_idx and idx_obj == idx_cstr:
                    continue
                cut_info = ['ub idx: [{}, {}] improved from {:.5f} to {:.5f} s.t. ub with idx: [{}, {}] > 0'.format(
                    layer_obj_idx, idx_obj, upper[idx2].cpu().item(), best_obj4[real_idx].cpu().item(), layer_cstr_idx, idx_cstr)]
                if cut_info in ret:
                    continue  # some cuts in sorted_improve_flipped may already added in
                ret.append(cut_info)
                if flipped:
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [], "arelu_decision": [[layer_obj_idx, idx_obj], [layer_cstr_idx, idx_cstr]],
                                "arelu_coeffs": [1.0, 1.0], "pre_decision": [], "pre_coeffs": [], "bias": 1.0, "c": -1})
                else:
                    assert best_obj4[real_idx] > 0
                    cut_ret.append({"relu_decision": [], "relu_coeffs": [],
                                    "arelu_decision": [[layer_cstr_idx, idx_cstr], [layer_cstr_idx, idx_cstr]],
                                    "arelu_coeffs": [-upper[idx2].cpu().item(), best_obj4[real_idx].cpu().item()],
                                    "pre_decision": [[layer_obj_idx, idx_obj]], "pre_coeffs": [-1.0],
                                    "bias": -upper[idx2].cpu().item(), "c": 1})

    find_cuts(sorted_improve_flipped, flipped=True, this_number_cuts=number_cuts)  # first selected flipped cuts
    number_cuts -= len(cut_ret)
    # flipped = len(cut_ret)
    # print('!!!number of flipped:', flipped)
    # print('!!!number of improved:', sum(sorted_improve_all.values > 1e-5) - flipped)

    if number_cuts > 0:
        find_cuts(sorted_improve_all, flipped=False, this_number_cuts=number_cuts)  # then add rest cuts
    print('add {} cuts:'.format(len(ret)))
    for i in ret: print(i)
    return cut_ret


def pair_heuristic(model, this_lA, this_lb, this_uA, this_ub, center, diff,  top_k=100, device='cuda', return_cstr_only=False):
    """
    l2l: means optimize lower obj s.t lower < 0
    l2u: means optimize lower obj s.t upper > 0
    u2l: means optimize upper obj s.t lower < 0
    u2u: means optimize upper obj s.t upper > 0
    """

    # calculate distance of Ax + b to x0
    lower_distance = (this_lA.matmul(center) + this_lb).squeeze().abs() - (diff.view(1, -1) * this_lA).norm(dim=1, p=1)
    upper_distance = (this_uA.matmul(center) + this_ub).squeeze().abs() - (diff.view(1, -1) * this_uA).norm(dim=1, p=1)
    # lower_distance = ((this_lA.matmul(center) + this_lb) / this_lA.norm(dim=1, p=2)).squeeze().abs() - diff[0]
    # upper_distance = ((this_uA.matmul(center) + this_ub) / this_uA.norm(dim=1, p=2)).squeeze().abs() - diff[0]

    # top_k candidates which have closest distance
    cstr_lower = lower_distance.argsort()[: min(sum(lower_distance < 0), top_k)]  # we only keep the top_k candidates
    cstr_upper = upper_distance.argsort()[: min(sum(upper_distance < 0), top_k)]  # we only keep the top_k candidates

    if return_cstr_only:
        return cstr_lower, cstr_upper

    batch_size = arguments.Config["bab"]["cut"]["batch_size_primal"]

    def filter_primal(cstr, bias, A_sign, zero_A_sign):
        # Check feasibility based on current primal variables of the objective. When objective has zero coefficients we look at the sign of the constraints.
        total_batch = len(cstr) // batch_size if len(this_lA) > batch_size else 1
        obj_opt = []
        for i in range(total_batch):
            this_batch = cstr[i * batch_size: (i + 1) * batch_size]
            if A_sign == -1 and zero_A_sign == -1:
                primal = get_primal_based_on_cstr(model, this_lA, -1, zero_A_sign=-1, cstr_A=this_lA[this_batch])
                obj_opt.append((this_lA[this_batch].unsqueeze(1).bmm(primal.permute(1, 2, 0)).squeeze(1) + bias[0, this_batch].unsqueeze(-1)))  # optimal lower obj s.t. lower bound < 0

            elif A_sign == -1 and zero_A_sign == +1:
                primal = get_primal_based_on_cstr(model, this_lA, -1, zero_A_sign=+1, cstr_A=this_uA[this_batch])
                obj_opt.append((this_uA[this_batch].unsqueeze(1).bmm(primal.permute(1, 2, 0)).squeeze(1) + bias[0, this_batch].unsqueeze(-1)))  # optimal lower obj s.t. upper bound > 0

            elif A_sign == +1 and zero_A_sign == -1:
                primal = get_primal_based_on_cstr(model, this_uA, +1, zero_A_sign=-1, cstr_A=this_lA[this_batch])
                obj_opt.append((this_lA[this_batch].unsqueeze(1).bmm(primal.permute(1, 2, 0)).squeeze(1) + bias[0, this_batch].unsqueeze(-1)))  # optimal upper obj s.t. lower bound < 0

            elif A_sign == +1 and zero_A_sign == +1:
                primal = get_primal_based_on_cstr(model, this_uA, +1, zero_A_sign=+1, cstr_A=this_uA[this_batch])
                obj_opt.append((this_uA[this_batch].unsqueeze(1).bmm(primal.permute(1, 2, 0)).squeeze(1) + bias[0, this_batch].unsqueeze(-1)))  # optimal upper obj s.t. upper bound > 0
        obj_opt = torch.cat(obj_opt, dim=0)

        del primal
        # constraints that still unsatisfied yet by given objective primal solutions
        if zero_A_sign == 1:
            unsatisfied = (obj_opt < 0).nonzero()
        else:
            unsatisfied = (obj_opt > 0).nonzero()

        obj_A_idx = unsatisfied[:, 1]
        # _, obj_len = unsatisfied[:, 0].unique(return_counts=True)
        unsatisfied = unsatisfied.cpu().numpy()
        obj_len = [(unsatisfied[:, 0] == i).sum() for i in range(len(cstr))]
        obj_len = torch.tensor(obj_len, device=device)

        return obj_A_idx, obj_len

    obj_A_l2l_idx, obj_len_l2l = filter_primal(cstr_lower, this_lb, -1, -1)
    obj_A_l2u_idx, obj_len_l2u = filter_primal(cstr_upper, this_ub, -1, +1)
    obj_A_u2l_idx, obj_len_u2l = filter_primal(cstr_lower, this_lb, +1, -1)
    obj_A_u2u_idx, obj_len_u2u = filter_primal(cstr_upper, this_ub, +1, +1)

    return cstr_lower, cstr_upper, obj_A_l2l_idx, obj_A_u2l_idx, obj_A_l2u_idx, obj_A_u2u_idx, obj_len_l2l, obj_len_l2u, obj_len_u2l, obj_len_u2u


@torch.no_grad()
@torch.jit.script
def get_primal_based_on_cstr_impl(x_lb, x_ub, A, cstr_A, sign : int, zero_A_sign : int):
    mask_A = (A * sign > 0).to(dtype = A.dtype)
    mask_special_case = (A == 0).to(dtype = A.dtype) * (zero_A_sign * cstr_A > 0).to(dtype = A.dtype)
    mask = torch.clamp(mask_A + mask_special_case, max=1.)
    return x_lb * (1. - mask) + x_ub * mask
    

@torch.no_grad()
def get_primal_based_on_cstr(model, A, sign, zero_A_sign=None, cstr_A=None):

    assert sign in [1, -1] and zero_A_sign in [None, 1, -1]

    assert model.x.ptb.norm == np.inf, print('we only support to get primals for Linf norm perturbation so far')
    assert sign in [-1, 1]
    batch_obj = A.shape[0]  # number of unstable neurons
    batch_cstr = cstr_A.shape[0]  # number of constraint

    x_lb, x_ub = model.x.ptb.x_L, model.x.ptb.x_U
    x_lb = x_lb.flatten(1).expand(batch_obj, batch_cstr, -1)  # n, topK, input
    x_ub = x_ub.flatten(1).expand(batch_obj, batch_cstr, -1)  # n, topK, input
    expand_A = A.unsqueeze(1).expand(-1, batch_cstr, -1)  # n, topK, input
    expand_cstr_A = cstr_A.unsqueeze(0).expand(batch_obj, -1, -1)  # n, topK, input
    input_primal = get_primal_based_on_cstr_impl(x_lb, x_ub, expand_A, expand_cstr_A, sign, zero_A_sign)
    return input_primal


def solve_by_gradient_decent(center, diff, cstr_len_l2l, cstr_len_l2u, cstr_len_u2l, cstr_len_u2u, obj_A_l2l, cstr_lA_l2l,
                             obj_lb_l2l, cstr_lb_l2l, obj_A_l2u, cstr_lA_l2u, obj_lb_l2u, cstr_lb_l2u, obj_A_u2l, cstr_uA_u2l,
                             obj_ub_u2l, cstr_ub_u2l, obj_A_u2u, cstr_uA_u2u, obj_ub_u2u, cstr_ub_u2u, iteration=100):
    rho_lower_l2l = torch.zeros(size=(sum(cstr_len_l2l), 1), requires_grad=True)
    rho_lower_l2u = torch.zeros(size=(sum(cstr_len_l2u), 1), requires_grad=True)

    rho_upper_u2l = torch.zeros(size=(sum(cstr_len_u2l), 1), requires_grad=True)
    rho_upper_u2u = torch.zeros(size=(sum(cstr_len_u2u), 1), requires_grad=True)

    opt = torch.optim.Adam([rho_lower_l2l, rho_lower_l2u, rho_upper_u2l, rho_upper_u2u], lr=0.1)

    for i in range(iteration):
        # optimize lower bounds
        # pair2 lb < 0
        loss1 = (obj_A_l2l + rho_lower_l2l * cstr_lA_l2l).matmul(center) - \
                (obj_A_l2l + rho_lower_l2l * cstr_lA_l2l).abs().matmul(diff) + \
                obj_lb_l2l + rho_lower_l2l.squeeze(-1) * cstr_lb_l2l

        # pair2 ub > 0
        loss2 = (obj_A_l2u - rho_lower_l2u * cstr_lA_l2u).matmul(center) - \
                (obj_A_l2u - rho_lower_l2u * cstr_lA_l2u).abs().matmul(diff) + \
                obj_lb_l2u - rho_lower_l2u.squeeze(-1) * cstr_lb_l2u

        # optimize upper bounds
        # pair2 lb < 0
        loss3 = (obj_A_u2l + rho_upper_u2l * cstr_uA_u2l).matmul(center) + \
                (obj_A_u2l + rho_upper_u2l * cstr_uA_u2l).abs().matmul(diff) + \
                obj_ub_u2l + rho_upper_u2l.squeeze(-1) * cstr_ub_u2l

        # pair2 ub > 0
        loss4 = (obj_A_u2u - rho_upper_u2u * cstr_uA_u2u).matmul(center) + \
                (obj_A_u2u - rho_upper_u2u * cstr_uA_u2u).abs().matmul(diff) + \
                obj_ub_u2u - rho_upper_u2u.squeeze(-1) * cstr_ub_u2u

        loss = - loss1.sum() - loss2.sum() + loss3.sum() + loss4.sum()
        print(i, loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        rho_lower_l2l.data = torch.clamp(rho_lower_l2l, min=0)
        rho_lower_l2u.data = torch.clamp(rho_lower_l2u, min=0)
        rho_upper_u2l.data = torch.clamp(rho_upper_u2l, min=0)
        rho_upper_u2u.data = torch.clamp(rho_upper_u2u, min=0)

    return loss1, loss2, loss3, loss4, rho_lower_l2l, rho_lower_l2u, rho_upper_u2l, rho_upper_u2u


def fast_solve(a, c, d, epsilon):
    def f(beta):
        return - (epsilon.view(1, 1, -1) * (a.unsqueeze(1) * beta.unsqueeze(-1) + c.unsqueeze(1))).abs().sum(-1) + d.unsqueeze(-1) * beta

    def f2(beta):
        return - (epsilon * (a * beta.unsqueeze(-1) + c)).abs().sum(-1) + d * beta

    # print(a[a != 0].abs().min(), c[c != 0].abs().min())
    # unexpect = torch.logical_and(a == 0, c == 0)
    # if ((unexpect == 0).sum(1) == 0).any():
    #     import pdb; pdb.set_trace()

    a_copy = a.clone()  # clone a for inplace operations.
    a_copy[a_copy == 0] = 1e-9
    q = -c / a_copy

    ''' enumerate_solve
    beta = torch.cat((torch.zeros(size=(q.size(0), 1)), q.clamp(min=0)), dim=1)  # Add 0 at the beginning.
    obj = f(beta)
    best_obj1, best_idx1 = obj.max(dim=-1)
    best_beta1 = beta[range(obj.size(0)), best_idx1].clamp(min=0)

    if (f2(best_beta1 * 2 + 1) > f2(best_beta1)).any():
        print(f'Objective is unbounded')
        raise ValueError

    return best_obj1, best_beta1
    '''

    batch_size = a.size(0)
    order_idx = torch.argsort(q, dim=-1)  # Dominates time complexity.
    sorted_a = a_copy.mul_(epsilon)  # inplace operation.
    row_indices = torch.arange(batch_size).view(-1, 1).expand(-1, a.size(1))
    sorted_a = sorted_a[row_indices, order_idx]

    sum_a_neg = -sorted_a.abs().cumsum(-1)
    total_a = -sum_a_neg[:, -1].unsqueeze(-1)  # sorted_a.abs().sum()
    sum_a_pos = total_a + sum_a_neg
    # Supergradient at the i-th crossing-zero point is in range [super_gradients[i-1], super_gradients[i]]
    # For i = 0, super_gradients[-1] = total_a * epsilon but we don't need to explicitly compute it - we compare to f(0) below.
    super_gradients = (sum_a_pos + sum_a_neg) + d.unsqueeze(-1)
    # Search where the supergradient contain 0, which is the point of maximum.
    best_idx = torch.searchsorted(-super_gradients, torch.zeros(size=(batch_size, 1), device=a.device), right=True).squeeze(1)

    # best_idx = torch.searchsorted(-super_gradients, 0, right=True)
    if (best_idx >= a.size(1)).any():
        # This should not happen in our case, if our constraints are from unstable neurons.
        print('Objective is unbounded.')
        raise ValueError
    else:
        best_beta = q[range(order_idx.size(0)), order_idx[range(order_idx.size(0)), best_idx]].clamp(min=0)
        best_obj = f2(best_beta)
        # We still need to compare to f(0), which is an additional end point.
        f0 = (-epsilon * c.abs()).sum(-1)
        cond = best_obj < f0
        if cond.any():
            best_obj[cond] = f0[cond]
            best_beta[cond] = 0
        # print(f'best obj is {best_obj.item()}, best beta is {best_beta}, idx {best_idx}')
        return best_obj, best_beta
