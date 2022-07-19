#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""alpha-beta-CROWN verifier with interface to handle vnnlib specifications used in VNN-COMP (except for ACASXu and nn4sys)."""

import socket
import random
import pickle
import os
import time
import gc
import csv
import torch
import numpy as np
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_min
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver import LiRPAConvNet
from lp_mip_solver import FSB_score
from utils import parse_model_shape_vnnlib, process_vnn_lib_attack, save_cex, reshape_bounds, attack
from nn4sys_verification import nn4sys_verification
from batch_branch_and_bound import relu_bab_parallel
from batch_branch_and_bound_input_split import input_bab_parallel

from read_vnnlib import batch_vnnlib
from cut_utils import terminate_mip_processes, terminate_single_mip_process

def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20).', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])

    arguments.Config.add_argument("--csv_name", type=str, default=None, help='Name of .csv file containing a list of properties to verify (VNN-COMP specific).', hierarchy=h + ["csv_name"])
    arguments.Config.add_argument("--results_file", type=str, default=None, help='Path to results file.', hierarchy=h + ["results_file"])
    arguments.Config.add_argument("--root_path", type=str, default='', help='Root path of VNN-COMP benchmarks (VNN-COMP specific).', hierarchy=h + ["root_path"])

    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="mnist_9_200", help='Model name.', hierarchy=h + ["name"])
    arguments.Config.add_argument("--onnx_path", type=str, default=None, help='Path to .onnx model file.', hierarchy=h + ["onnx_path"])
    arguments.Config.add_argument("--onnx_path_prefix", type=str, default='', help='Add a prefix to .onnx model path to correct malformed csv files.', hierarchy=h + ["onnx_path_prefix"])
    arguments.Config.add_argument("--onnx_optimization_flags", choices=["merge_bn", "merge_linear", "merge_gemm", "none"], default="none", help='Onnx graph optimization config.', hierarchy=h + ["onnx_optimization_flags"])

    h = ["specification"]
    arguments.Config.add_argument("--vnnlib_path", type=str, default=None, help='Path to .vnnlib specification file.', hierarchy=h + ["vnnlib_path"])
    arguments.Config.add_argument("--vnnlib_path_prefix", type=str, default='', help='Add a prefix to .vnnlib specs path to correct malformed csv files.', hierarchy=h + ["vnnlib_path_prefix"])

    h = ["data"]
    arguments.Config.add_argument("--dataset", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "CIFAR_SDP_FULL", "CIFAR_RESNET", "CIFAR_SAMPLE", "MNIST_SAMPLE", "CIFAR_ERAN", "MNIST_ERAN",
                                 "MNIST_ERAN_UN", "MNIST_SDP", "MNIST_MADRY_UN", "CIFAR_SDP", "CIFAR_UN", "NN4SYS", "TEST"], help='Dataset name. Dataset must be defined in utils.py.', hierarchy=h + ["dataset"])
    arguments.Config.add_argument("--filter_path", type=str, default=None, help='A filter in pkl format contains examples that will be skipped (not used).', hierarchy=h + ["data_filter_path"])

    h = ["attack"]
    arguments.Config.add_argument("--mip_attack", action='store_true', help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.', hierarchy=h + ["enable_mip_attack"])
    arguments.Config.add_argument('--cex_path', type=str, default='./test_cex.txt', help='Save path for counter-examples.', hierarchy=h + ["cex_path"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", None], help='Debugging option. Do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()


def get_labels(model_ori, x, vnnlib):
    for prop_mat, prop_rhs in vnnlib[0][1]:
        if len(prop_rhs) > 1:
            # we only verify the easiest one
            output = model_ori(x).detach().cpu().numpy().flatten()
            print(output)
            vec = prop_mat.dot(output)
            selected_prop = prop_mat[vec.argmax()]
            y = int(np.where(selected_prop == 1)[0])  # true label
            pidx = int(np.where(selected_prop == -1)[0])  # target label
            # FIXME arguments.Config should be read-only
            arguments.Config["bab"]["decision_thresh"] = prop_rhs[vec.argmax()]
        else:
            assert len(prop_mat) == 1
            y = np.where(prop_mat[0] == 1)[0]
            if len(y) != 0:
                y = int(y)
            else:
                y = None
            pidx = np.where(prop_mat[0] == -1)[0]  # target label
            pidx = int(pidx) if len(pidx) != 0 else None  # Fix constant specification with no target label.
            if y is not None and pidx is None: y, pidx = pidx, y  # Fix vnnlib with >= const property.
            # FIXME arguments.Config should be read-only
            arguments.Config["bab"]["decision_thresh"] = prop_rhs[0]
        if pidx == y:
            raise NotImplementedError
    return y


def incomplete_verifier(model_ori, data, y=None, data_ub=None, data_lb=None, eps=0.0, vnnlib=None):
    norm = arguments.Config["specification"]["norm"]
    # LiRPA wrapper
    num_outputs = arguments.Config["data"]["num_outputs"]
    if vnnlib:
        # Generally, c should be constructed from vnnlib
        assert len(vnnlib) == 1
        vnnlib = vnnlib[0]
        c = torch.tensor([item[0] for item in vnnlib[1]]).to(data)
        if c.shape[0] != 1:
            # TODO need a more general solution
            c = c.transpose(0, 1)
        arguments.Config["bab"]["decision_thresh"] = torch.tensor([item[1] for item in vnnlib[1]]).to(data)
    else:
        print("this option not supported!")
        exit()
        if y is not None:
            labels = torch.tensor([y]).long()
            if num_outputs == 1:
                # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
                c = (float(y) - 0.5) * 2 * torch.ones(size=(data.size(0), 1, 1))
            else:
                # Building a spec for all target labels.
                c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
                I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
                # Remove spec to self.
                c = (c[I].view(data.size(0), num_outputs - 1, num_outputs))
        else:
            c = None

    model = LiRPAConvNet(model_ori, y, None, in_size=data.shape, c=c)
    print('Model prediction is:', model.net(data))
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    bound_prop_method = arguments.Config["solver"]["bound_prop_method"]

    _, global_lb, _, _, _, mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_images = model.build_the_model(
            domain, x, data_lb, data_ub, vnnlib, stop_criterion_func=stop_criterion_min(arguments.Config["bab"]["decision_thresh"]))

    if (global_lb > arguments.Config["bab"]["decision_thresh"]).all():
        print("verified with init bound!")
        return "safe-incomplete", None, None, None, None

    if (arguments.Config["attack"]["pgd_order"] == "middle"):
        if (attack_images is not None): return "unsafe-pgd", None, None, None, None

    # Save the alpha variables during optimization. Here the batch size is 1.
    saved_slopes = defaultdict(dict)
    for m in model.net.relus:
        for spec_name, alpha in m.alpha.items():
            # each slope size is (2, spec, 1, *shape); batch size is 1.
            saved_slopes[m.name][spec_name] = alpha.detach().clone()

    if bound_prop_method == 'alpha-crown':
        # obtain and save relu alphas
        activation_opt_params = dict([(relu.name, relu.dump_optimized_params()) for relu in model.net.relus])
    else:
        activation_opt_params = None

    if y is not None and num_outputs > 1:
        # For the last layer, since we missed one label, we add them back here.
        assert lower_bounds[-1].size(0) == 1  # this function only handles batchsize = 1.
        lower_bounds, upper_bounds, global_lb = reshape_bounds(lower_bounds, upper_bounds, y, global_lb)
        saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)
    else:
        saved_bounds = (model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA)

    return "unknown", global_lb, saved_bounds, saved_slopes, activation_opt_params

def mip(saved_bounds, y, labels_to_verify=None):

    lirpa_model, lower_bounds, upper_bounds, mask, pre_relu_indices, lA = saved_bounds
    refined_betas = None

    if arguments.Config["general"]["complete_verifier"] == "mip":
        mip_global_lb, mip_global_ub, mip_status, mip_adv = lirpa_model.build_the_model_mip(labels_to_verify=labels_to_verify, save_adv=True)

        if mip_global_lb.ndim == 1:
            mip_global_lb = mip_global_lb.unsqueeze(0)  # Missing batch dimension.
        if mip_global_ub.ndim == 1:
            mip_global_ub = mip_global_ub.unsqueeze(0)  # Missing batch dimension.
        print(f'MIP solved lower bound: {mip_global_lb}')
        print(f'MIP solved upper bound: {mip_global_ub}')

        verified_status = "safe-mip"
        # Batch size is always 1.
        labels_to_check = labels_to_verify if labels_to_verify is not None else range(len(mip_status))
        for pidx in labels_to_check:
            if mip_global_lb[0, pidx] >=0:
                # Lower bound > 0, verified.
                continue
            # Lower bound < 0, now check upper bound.
            if mip_global_ub[0, pidx] <=0:
                # Must be 2 cases: solved with adv example, or early terminate with adv example.
                assert mip_status[pidx] in [2, 15]
                print("verified unsafe-mip with init mip!")
                return "unsafe-mip", mip_global_lb, None, None, None
            # Lower bound < 0 and upper bound > 0, must be a timeout.
            assert mip_status[pidx] == 9 or mip_status[pidx] == -1, "should only be timeout for label pidx"
            verified_status = "unknown-mip"

        print(f"verified {verified_status} with init mip!")
        return verified_status, mip_global_lb, None, None, None

    elif arguments.Config["general"]["complete_verifier"] == "bab-refine":
        print("Start solving intermediate bounds with MIP...")
        score = FSB_score(lirpa_model.net, lower_bounds, upper_bounds, mask, pre_relu_indices, lA, bound_reshaped=False if y is None else True)

        refined_lower_bounds, refined_upper_bounds, refined_betas = lirpa_model.build_the_model_mip_refine(lower_bounds, upper_bounds,
                            score=score, stop_criterion_func=stop_criterion_min(1e-4))
        if arguments.Config["data"]["num_outputs"] > 1:
            lower_bounds, upper_bounds, _ = reshape_bounds(refined_lower_bounds, refined_upper_bounds, y)
        else:
            lower_bounds, upper_bounds, = refined_lower_bounds, refined_upper_bounds
        refined_global_lb = lower_bounds[-1]
        print("refined global lb:", refined_global_lb, "min:", refined_global_lb.min())
        if refined_global_lb.min()>=0:
            print("Verified safe using alpha-CROWN with MIP improved bounds!")
            return "safe-incomplete-refine", refined_global_lb, lower_bounds, upper_bounds, None

        return "unknown", refined_global_lb, lower_bounds, upper_bounds, refined_betas
    else:
        return "unknown", -float("inf"), lower_bounds, upper_bounds, refined_betas


def bab(unwrapped_model, data, targets, y, eps=None, data_ub=None, data_lb=None,
        lower_bounds=None, upper_bounds=None, reference_slopes=None,
        attack_images=None, c=None, all_prop=None, cplex_processes=None,
        activation_opt_params=None, reference_lA=None, rhs=None, 
        model_incomplete=None, timeout=None, refined_betas=None):
    # FIXME robustness_verifier.py is broken as there is no `c`.
    # c seems to have duplicate information as targets and y.

    norm = arguments.Config["specification"]["norm"]
    if arguments.Config["specification"]["type"] == 'lp':
        if norm == np.inf:
            if data_ub is None:
                data_ub = data + eps
                data_lb = data - eps
            elif eps is not None:
                data_ub = torch.min(data + eps, data_ub)
                data_lb = torch.max(data - eps, data_lb)
        else:
            data_ub = data_lb = data
            assert torch.unique(eps).numel() == 1  # For other norms, the eps must be the same for each channel.
            eps = torch.mean(eps).item()
    elif arguments.Config["specification"]["type"] == 'bound':
        # Use data_lb and data_ub directly.
        assert norm == np.inf
    else:
        raise ValueError(f'Unsupported perturbation type {arguments.Config["specification"]["type"]}')

    # This will use the refined bounds if the complete verifier is "bab-refine".
    # FIXME do not repeatedly create LiRPAConvNet which creates a new BoundedModule each time.
    model = LiRPAConvNet(
        unwrapped_model, y, targets,
        in_size=data.shape if not targets.size > 1 else [len(targets)] + list(data.shape[1:]),
        c=c, cplex_processes=cplex_processes)

    data = data.to(model.device)
    data_lb, data_ub = data_lb.to(model.device), data_ub.to(model.device)
    output = model.net(data).flatten()
    print('Model prediction is (first 10):', output[:10])

    if arguments.Config['attack']['check_clean']:
        clean_rhs = c.matmul(output)
        print(f'Clean RHS: {clean_rhs}')
        if (clean_rhs < rhs).any():
            return -np.inf, np.inf, None, None, 'unsafe'

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    if cut_enabled:
        model.set_cuts(model_incomplete.A_saved, x, lower_bounds, upper_bounds)

    if arguments.Config["bab"]["branching"]["input_split"]["enable"]:
        min_lb, min_ub, glb_record, nb_states, verified_ret = input_bab_parallel(
            model, domain, x, model_ori=unwrapped_model, all_prop=all_prop,
            rhs=rhs, timeout=timeout, branching_method=arguments.Config["bab"]["branching"]["method"])
    else:
        min_lb, min_ub, glb_record, nb_states, verified_ret = relu_bab_parallel(
            model, domain, x,
            refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
            activation_opt_params=activation_opt_params, reference_lA=reference_lA,
            reference_slopes=reference_slopes, attack_images=attack_images,
            targets=targets, timeout=timeout, refined_betas=refined_betas, rhs=rhs)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    if min_lb is None:
        min_lb = -np.inf
    if isinstance(min_ub, torch.Tensor):
        min_ub = min_ub.item()
    if min_ub is None:
        min_ub = np.inf

    return min_lb, min_ub, nb_states, glb_record, verified_ret


def update_parameters(model, data_min, data_max):
    if 'vggnet16_2022' in arguments.Config['general']['root_path']:
        perturbed = (data_max - data_min > 0).sum()
        if perturbed > 10000:
            print('WARNING: prioritizing attack due to too many perturbed pixels on VGG')
            print('Setting arguments.Config["attack"]["pgd_order"] to "before"')
            arguments.Config['attack']['pgd_order'] = 'before'

    # Legacy code
    if False:
        MODEL_LAYER_THRESHOLD = 3
        if sum(p.numel() for p in model.parameters()) < 15:
            return        
        if sum([type(m) == torch.nn.Sigmoid for m in list(model.modules())]) > 0:
            # if there is Sigmoid in model
            print('arguments.Config["general"]["loss_reduction_func"] change: {} -> {}'.format(arguments.Config["general"]["loss_reduction_func"], 'min'))
            arguments.Config["general"]["loss_reduction_func"] = 'min'
            print('arguments.Config["solver"]["alpha-crown"]["iteration"] change: {} -> {}'.format(arguments.Config["solver"]["alpha-crown"]["iteration"], 1000))
            arguments.Config["solver"]["alpha-crown"]["iteration"] = 1000
            print('arguments.Config["solver"]["alpha-crown"]["lr_decay"] change: {} -> {}'.format(arguments.Config["solver"]["alpha-crown"]["lr_decay"], 0.999))
            arguments.Config["solver"]["alpha-crown"]["lr_decay"] = 0.999
            print('arguments.Config["solver"]["beta-crown"]["lr_decay"] change: {} -> {}'.format(arguments.Config["solver"]["beta-crown"]["lr_decay"], 0.999))
            arguments.Config["solver"]["beta-crown"]["lr_decay"] = 0.999
            print('arguments.Config["solver"]["early_stop_patience"] change: {} -> {}'.format(arguments.Config["solver"]["early_stop_patience"], 1000))
            arguments.Config["solver"]["early_stop_patience"] = 1000

            if arguments.Config["attack"]["pgd_order"] != 'skip':
                # It may be set to "skip" when testing on unsafe examples,
                # to check whether verification and attack are contradictory.
                print('arguments.Config["attack"]["pgd_order"] change: {} -> {}'.format(arguments.Config["attack"]["pgd_order"], 'before'))
                arguments.Config["attack"]["pgd_order"] = 'before'
            if not arguments.Config["bab"]["branching"]["input_split"]["enable"]:
                print('arguments.Config["general"]["complete_verifier"] change: {} -> {}'.format(arguments.Config["general"]["complete_verifier"], 'skip'))
                arguments.Config["general"]["complete_verifier"] = 'skip'
            return

        if sum([type(m) == torch.nn.ReLU for m in list(model.modules())]) < MODEL_LAYER_THRESHOLD and not arguments.Config["bab"]["branching"]["input_split"]["enable"]:
            # if the number of ReLU layers < 3
            print('arguments.Config["general"]["complete_verifier"] change: {} -> {}'.format(arguments.Config["general"]["complete_verifier"], 'mip'))
            arguments.Config["general"]["complete_verifier"] = 'mip'
            return


def sort_target_labels(properties, lower_bounds):
    # need to sort pidx such that easier first according to initial alpha crown
    y = int(np.where(properties[1][0][0][0] == 1)[0])
    target_label_arrays = list(properties[1])
    # target_label_arrays does not have true label entity
    target_label_arrays.insert(y, None)
    init_lbs = lower_bounds[-1].reshape(-1).detach()
    sorted_pidx = init_lbs.argsort(descending=True).tolist()
    target_label_arrays = [target_label_arrays[spidx] for spidx in sorted_pidx]
    target_label_arrays.remove(None)
    print("Using cplex cuts, sort target labels [easier->harder]:", sorted_pidx)
    return target_label_arrays

def sort_targets_cls(batched_vnnlib, init_global_lb, reverse=False):
    # To sort targets, this must be a classification task, and initial_max_domains
    # is set to 1.
    num_classes = init_global_lb.shape[1]
    assert len(batched_vnnlib) == num_classes - 1
    scores = init_global_lb[0]
    # For a vnnlib item vl, vl[1][0][3] is the target
    batched_vnnlib = sorted(batched_vnnlib, key=lambda vl: scores[vl[1][0][3]], reverse=reverse)
    return batched_vnnlib

def complete_verifier(
        model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
        init_global_lb, lower_bounds, upper_bounds, index,
        timeout_threshold, bab_ret=None, lA=None, cplex_processes=None,
        reference_slopes=None, activation_opt_params=None, refined_betas=None):
    start_time = time.time()
    cplex_cuts = arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]
    sort_targets = arguments.Config["bab"]["sort_targets"]

    if sort_targets:
        batched_vnnlib = sort_targets_cls(batched_vnnlib, init_global_lb)
    if cplex_cuts:
        # need to sort pidx such that easier first according to initial alpha crown
        batched_vnnlib = sort_targets_cls(batched_vnnlib, init_global_lb, reverse=True)

    for property_idx, properties in enumerate(batched_vnnlib):  # loop of x
        # batched_vnnlib: [x, [(c, rhs, y, pidx)]]
        print(f'\nProperties batch {property_idx}, size {len(properties[0])}')
        timeout = timeout_threshold - (time.time() - start_time)
        print(f'Remaining timeout: {timeout}')
        start_time_bab = time.time()

        x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
        data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
        data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
        x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.

        target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)

        assert len(target_label_arrays) == 1
        c, rhs, y, pidx = target_label_arrays[0]

        print('##### [{}] True label: {}, Tested against: {}, thresh: {} ######'.format(
            index, y.flatten(), pidx.flatten(), rhs.flatten()))

        if np.array(pidx == y).any():
            raise NotImplementedError

        torch.cuda.empty_cache()
        gc.collect()

        c = torch.tensor(c, dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
        rhs = torch.tensor(rhs, dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])

        # extract cplex cut filename
        if cplex_cuts:
            assert arguments.Config["bab"]["initial_max_domains"] == 1

        # Complete verification (BaB, BaB with refine, or MIP).
        if arguments.Config["general"]["enable_incomplete_verification"]:
            assert arguments.Config["bab"]["branching"]["input_split"]["enable"] is False
            # Reuse results from incomplete results, or from refined MIPs.
            # skip the prop that already verified
            # TODO need a more general way to handle here, since the pidx and y are meaningless sometimes. (y > constant)
            rlb, rub = list(lower_bounds), list(upper_bounds)
            if arguments.Config["data"]["num_outputs"] != 1:
                rlb[-1] = rlb[-1][0, pidx]
                init_verified_cond = (rlb[-1] > rhs).flatten()
                init_verified_idx = pidx.flatten()[torch.where(init_verified_cond)[0].cpu()]
                if init_verified_idx.size > 0:
                    print(f"Init opt crown verified for label {init_verified_idx} with bound {init_global_lb[0, init_verified_idx]}.")
                    l, ret = init_global_lb[0, init_verified_idx].cpu().numpy().tolist(), 'safe'
                    bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                init_failure_idx = pidx.flatten()[torch.where(~init_verified_cond)[0].cpu()]
                if init_failure_idx.size == 0:
                    # This batch of x verified by init opt crown
                    continue
                print(f"Remaining labels {init_failure_idx} with bounds {init_global_lb[0, init_failure_idx]} need to verify.")
                assert len(np.unique(y)) == 1 and len(rhs.unique()) == 1
            else:
                init_verified_cond, init_failure_idx, y = torch.tensor([1]), np.array(0), np.array(0)

            # TODO change index [0:1] to [torch.where(~init_verified_cond)[0]] can handle more general vnnlib for multiple x
            l, u, nodes, glb_record, ret = bab(
                model_ori, x[0:1], init_failure_idx, y=np.unique(y),
                data_ub=data_max[0:1], data_lb=data_min[0:1],
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                c=c[torch.where(~init_verified_cond)[0]],
                reference_slopes=reference_slopes, cplex_processes=cplex_processes, rhs=rhs[0:1],
                activation_opt_params=activation_opt_params, reference_lA=lA,
                model_incomplete=model_incomplete, timeout=timeout, refined_betas=refined_betas)
            bab_ret.append([index, float(l), nodes, time.time() - start_time_bab, init_failure_idx.tolist()])
        else:
            assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
            # input split also goes here directly
            l, u, nodes, _, ret = bab(
                model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min, c=c,
                all_prop=target_label_arrays, cplex_processes=cplex_processes,
                rhs=rhs, timeout=timeout)
            bab_ret.append([index, l, nodes, time.time() - start_time_bab, pidx])

        # terminate the corresponding cut inquiry process if exists
        if cplex_cuts:
            terminate_single_mip_process(cplex_processes, pidx)

        timeout = timeout_threshold - (time.time() - start_time)
        if ret == 'unsafe':
            return 'unsafe-bab'
        if ret == 'unknown' or timeout < 0:
            return 'unknown'
        if ret != 'safe':
            raise ValueError(f'Unknown return value of bab: {ret}')
    else:
        return 'safe'


def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    torch.set_printoptions(precision=8)
    device = arguments.Config["general"]["device"]
    if device != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config["general"]["seed"])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if arguments.Config["general"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    if arguments.Config["general"]["double_fp"]:
        torch.set_default_dtype(torch.float64)
    
    precompile_jit_kernels() 
    
    if arguments.Config["specification"]["norm"] != np.inf and arguments.Config["attack"]["pgd_order"] != "skip":
        print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
        arguments.Config["attack"]["pgd_order"] = "skip"

    if arguments.Config["general"]["csv_name"] is not None and arguments.Config["model"]["onnx_path"] is None \
            and arguments.Config["specification"]["vnnlib_path"] is None:
        file_root = arguments.Config["general"]["root_path"]

        with open(os.path.join(file_root, arguments.Config["general"]["csv_name"]), newline='') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            csv_file = []
            for row in reader:
                csv_file.append(row)

        save_path = 'vnn-comp_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_initial_max_domains={}.npz'. \
            format(os.path.splitext(arguments.Config["general"]["csv_name"])[0], arguments.Config["data"]["start"],  arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"], arguments.Config["solver"]["beta-crown"]["batch_size"],
                   arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"], arguments.Config["bab"]["branching"]["reduceop"],
                   arguments.Config["bab"]["branching"]["candidates"], arguments.Config["solver"]["alpha-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"],
                   arguments.Config["bab"]["initial_max_domains"])
        print(f'saving results to {save_path}')

        arguments.Config["data"]["end"] = min(arguments.Config["data"]["end"], reader.line_num)
        if arguments.Config["data"]["start"] != 0 or arguments.Config["data"]["end"] != reader.line_num:
            assert arguments.Config["data"]["start"]>=0 and arguments.Config["data"]["start"]<=reader.line_num and arguments.Config["data"]["end"]>arguments.Config["data"]["start"],\
                "start or end sample error: {}, {}, {}".format(arguments.Config["data"]["end"], arguments.Config["data"]["start"], reader.line_num)
            print("customized start/end sample from {} to {}".format(arguments.Config["data"]["start"], arguments.Config["data"]["end"]))
        else:
            print("no customized start/end sample, testing for all samples")
            arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, reader.line_num
    else:
        # run in .sh
        arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, 1
        csv_file = [(arguments.Config["model"]["onnx_path"], arguments.Config["specification"]["vnnlib_path"], arguments.Config["bab"]["timeout"])]
        save_path = arguments.Config["general"]["results_file"]
        file_root = ''

    verification_summary = defaultdict(list)
    time_per_sample_list = []
    status_per_sample_list = []
    bab_ret = []
    cnt = 0  # Number of examples in this run.
    bnb_ids = csv_file[arguments.Config["data"]["start"]:arguments.Config["data"]["end"]]
    select_instance = arguments.Config["data"]["select_instance"]

    for new_idx, csv_item in enumerate(bnb_ids):
        arguments.Globals["example_idx"] = new_idx
        vnnlib_id = new_idx + arguments.Config["data"]["start"]

        # Select some instances to verify
        if select_instance and not vnnlib_id in select_instance:
            continue

        start_time = time.time()
        print(f'\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx: {new_idx}, vnnlib ID: {vnnlib_id} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        onnx_path, vnnlib_path, arguments.Config["bab"]["timeout"] = csv_item
        onnx_path = os.path.join(arguments.Config["model"]["onnx_path_prefix"], onnx_path.strip())
        vnnlib_path = os.path.join(arguments.Config["specification"]["vnnlib_path_prefix"], vnnlib_path.strip())
        print(f'Using onnx {onnx_path}')
        print(f'Using vnnlib {vnnlib_path}')

        arguments.Config["bab"]["timeout"] = float(arguments.Config["bab"]["timeout"])
        if arguments.Config["bab"]["timeout_scale"] != 1:
            new_timeout = arguments.Config["bab"]["timeout"] * arguments.Config["bab"]["timeout_scale"]
            print(f'Scaling timeout: {arguments.Config["bab"]["timeout"]} -> {new_timeout}')
            arguments.Config["bab"]["timeout"] = new_timeout
        if arguments.Config["bab"]["override_timeout"] is not None:
            new_timeout = arguments.Config["bab"]["override_timeout"]
            print(f'Overriding timeout: {new_timeout}')
            arguments.Config["bab"]["timeout"] = new_timeout
        timeout_threshold = arguments.Config["bab"]["timeout"]  # In case arguments.Config["bab"]["timeout"] is changed later.

        ### preprocessor-hint: private-section-start
        if arguments.Config["bab"]["cut"]["_eran_cuts"] is not None:
            # FIXME: Remove this dirty option.
            arguments.Config["bab"]["cut"]["_tmp_cuts"] = os.path.join(
                arguments.Config["bab"]["cut"]["_eran_cuts"], f'{new_idx + arguments.Config["data"]["start"]}.krelu.pt')
        ### preprocessor-hint: private-section-end

        model_ori, is_channel_last, shape, vnnlib = parse_model_shape_vnnlib(file_root, onnx_path, vnnlib_path)

        if arguments.Config["data"]["dataset"] == 'NN4SYS':
            verified_status = res = nn4sys_verification(model_ori, vnnlib, onnx_path=os.path.join(file_root, onnx_path))
            print(res)
        else:
            model_ori.eval()

            # All other models.
            if is_channel_last:
                vnnlib_shape = shape[:1] + shape[2:] + shape[1:2]
                print(f'Notice: this ONNX file has NHWC order. We assume the X in vnnlib is also flattend in in NHWC order {vnnlib_shape}')
            else:
                vnnlib_shape = shape

            # FIXME attack and initial_incomplete_verification only works for assert len(vnnlib) == 1
            x_range = torch.tensor(vnnlib[0][0], dtype=torch.get_default_dtype())
            data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
            data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
            x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
            if is_channel_last:
                # The VNNlib file has X in NHWC order. We always use NCHW order.
                data_min = data_min.permute(0, 3, 1, 2).contiguous()
                data_max = data_max.permute(0, 3, 1, 2).contiguous()
                x = x.permute(0, 3, 1, 2).contiguous()

            # auto tune args
            update_parameters(model_ori, data_min, data_max)

            model_ori = model_ori.to(device)
            x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)

            verified_status = "unknown"
            verified_success = False                

            if arguments.Config["attack"]["pgd_order"] == "before":
                verified_status, verified_success, attack_images = attack(
                    model_ori, x, data_min, data_max, vnnlib,
                    verified_status, verified_success)

            init_global_lb = saved_bounds = saved_slopes = y = lower_bounds = upper_bounds = None
            activation_opt_params = model_incomplete = lA = cplex_processes = None

            # Incomplete verification is enabled by default. The intermediate lower
            # and upper bounds will be reused in bab and mip.
            if (not verified_success and (
                    arguments.Config["general"]["enable_incomplete_verification"]
                    or arguments.Config["general"]["complete_verifier"] == "bab-refine")):
                assert len(vnnlib) == 1
                y = None
                if arguments.Config["data"]["num_outputs"] > 1:
                    y = get_labels(model_ori, x, vnnlib)
                verified_status, init_global_lb, saved_bounds, saved_slopes, activation_opt_params = \
                    incomplete_verifier(
                        model_ori, x, y=y, 
                        data_ub=data_max, data_lb=data_min, vnnlib=vnnlib) 
                verified_success = verified_status != "unknown"
                if not verified_success:
                    model_incomplete, lower_bounds, upper_bounds = saved_bounds[:3]
                    lA = saved_bounds[-1]

            if not verified_success and arguments.Config["attack"]["pgd_order"] == "after":
                verified_status, verified_success, attack_images = attack(
                    model_ori, x, data_min, data_max, vnnlib,
                    verified_status, verified_success)

            # MIP or MIP refined bounds.
            refined_betas = None
            if not verified_success and (arguments.Config["general"]["complete_verifier"] == "mip" or arguments.Config["general"]["complete_verifier"] == "bab-refine"):
                verified_status, init_global_lb, lower_bounds, upper_bounds, refined_betas = mip(saved_bounds=saved_bounds, y=y)
                verified_success = verified_status != "unknown"

            # extract the process pool for cut inquiry
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                if saved_bounds is not None:
                    # use nullity of saved_bounds as an indicator of whether cut processes are launched
                    # saved_bounds[0] is the AutoLiRPA model instance
                    cplex_processes = saved_bounds[0].processes
                    mip_building_proc = saved_bounds[0].mip_building_proc

            # BaB bounds. (not do bab if unknown by mip solver for now)
            if not verified_success and arguments.Config["general"]["complete_verifier"] != "skip" and verified_status != "unknown-mip":
                batched_vnnlib = batch_vnnlib(vnnlib)
                verified_status = complete_verifier(
                    model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
                    init_global_lb, lower_bounds, upper_bounds, new_idx,
                    timeout_threshold=timeout_threshold - (time.time() - start_time),
                    bab_ret=bab_ret, lA=lA, cplex_processes=cplex_processes,
                    reference_slopes=saved_slopes, activation_opt_params=activation_opt_params,
                    refined_betas=refined_betas)

            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and saved_bounds is not None:
                terminate_mip_processes(mip_building_proc, cplex_processes)
                del cplex_processes

            del init_global_lb, saved_bounds, saved_slopes

        # Summarize results.
        if arguments.Config["model"]["onnx_path"] is not None and arguments.Config["specification"]["vnnlib_path"] is not None:
            # run in .sh
            if ('unknown' in verified_status or
                    'timeout' in verified_status or
                    'timed out' in verified_status):
                verified_status = 'timeout'
            elif 'unsafe' in verified_status:
                verified_status = 'sat'
            elif 'safe' in verified_status:
                verified_status = 'unsat'
            else:
                raise ValueError(f'Unknown verified_status {verified_status}')

            print('Result:', verified_status)
            with open(save_path, "w") as file:
                file.write(verified_status)
                if arguments.Config["general"]["save_adv_example"]:
                    if verified_status == 'sat':
                        file.write('\n')
                        with open(arguments.Config["attack"]["cex_path"], "r") as adv_example:
                            file.write(adv_example.read())
                file.flush()
        else:
            cnt += 1
            if time.time() - start_time > timeout_threshold:
                if 'unknown' not in verified_status:
                    verified_status += ' (timed out)'
            verification_summary[verified_status].append(new_idx)
            status_per_sample_list.append(verified_status)
            time_per_sample_list.append(time.time() - start_time)
            with open(save_path, "wb") as f:
                pickle.dump({"summary": verification_summary, "results": status_per_sample_list, "time": time_per_sample_list, "bab_ret": bab_ret}, f)
            print(f"Result: {verified_status} in {time_per_sample_list[-1]:.4f} seconds")

    if arguments.Config["general"]["csv_name"] is not None and arguments.Config["model"]["onnx_path"] is None and arguments.Config["specification"]["vnnlib_path"] is None:
        # Finished all examples.
        num_timeout = sum("unknown" in s for s in status_per_sample_list)
        num_verified = sum("safe" in s and "unsafe" not in s for s in status_per_sample_list)
        num_unsafe = sum("unsafe" in s for s in status_per_sample_list)

        with open(save_path, "wb") as f:
            pickle.dump({"summary": verification_summary, "results": status_per_sample_list, "time": time_per_sample_list, "bab_ret": bab_ret}, f)

        print("############# Summary #############")
        print("Final verified acc: {}% [total {} examples]".format(num_verified / len(bnb_ids) * 100., len(bnb_ids)))
        print("Total verification count:", num_verified + num_unsafe + num_timeout, ", total verified safe:", num_verified,
              ", verified unsafe:", num_unsafe, ", timeout:", num_timeout)
        if len(bab_ret) > 0:
            time_sum = sum([item[3] for item in bab_ret])
            print("mean time (bab) [total:{}]: {}".format(len(bnb_ids), time_sum / float(len(bnb_ids))))
        print(f"mean time [{num_verified + num_timeout}]", np.sum(time_per_sample_list)/(num_verified + num_timeout), "max time", np.max(time_per_sample_list))
        for k, v in verification_summary.items():
            print(f"{k} (total {len(v)}):", v)


if __name__ == "__main__":
    config_args()
    main()
