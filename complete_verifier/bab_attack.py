#### preprocessor-hint: private-file
"""Branch and bound based adversarial attacks."""

import copy
import time
from collections import defaultdict, Counter
import numpy as np
import torch
import arguments
from attack_pgd import pgd_attack
from branching_domains import pick_out_batch, ReLUDomain, SortedList, DFS_SortedList, merge_domains_params
from branching_heuristics import choose_node_parallel_kFSB
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedModule, BoundedTensor


### preprocessor-hint: private-section-start
diving_Visited = 0  # FIXME (10/21): Do not use global variable here. This will not be cleaned across examples.
### preprocessor-hint: private-section-end

def history_to_splits(history):
    splits = []
    coeffs = []
    for layer_idx, layer_history in enumerate(history):
        for s, c in zip(*layer_history):
            splits.append([layer_idx, s])
            coeffs.append(c)
    return splits, coeffs


def find_promising_domains(net, adv_pool, dive_domains, candidates, start_iter, max_dive_fix, min_local_free):
    find_promising_domains.counter += 1
    all_splits = []
    all_coeffs = []
    all_advs = []
    all_act_patterns = []
    # Skip the earliest a few iterations.
    if find_promising_domains.counter < start_iter:
        print(f'Current iteration {find_promising_domains.counter}, MIP will start at iteration {start_iter}')
        return all_splits, all_coeffs, all_advs, all_act_patterns
    n_domains = candidates - 1
    counter = find_promising_domains.counter - start_iter
    if find_promising_domains.current_method == 'top-down':
        # automatically adjust top down max_dive_fix_ratio
        max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
        if max_dive_fix_ratio > 0 and find_promising_domains.topdown_status == "infeas":
            # still run top down for next sub-mip round because the previous one has too large dive and inf sub-mips
            find_promising_domains.current_method = "bottom-up"
            print(f"orig max_dive_fix_ratio: {max_dive_fix_ratio}, fix: {max_dive_fix}")
            new_max_dive_fix_ratio = max(max_dive_fix_ratio - 0.2, 0)
            max_dive_fix = int(max_dive_fix / max_dive_fix_ratio * new_max_dive_fix_ratio)
            arguments.Config["bab"]["attack"]["max_dive_fix_ratio"] = new_max_dive_fix_ratio
            print(f"### topdown most inf! reduce max_dive_fix_ratio to {new_max_dive_fix_ratio}, now fix: {max_dive_fix}")
        if max_dive_fix_ratio < 1 and find_promising_domains.topdown_status == "timeout":
            # wait for next round topdown dive with the increased max_dive_fix_ratio
            print(f"orig max_dive_fix_ratio: {max_dive_fix_ratio}, fix: {max_dive_fix}")
            new_max_dive_fix_ratio = min(max_dive_fix_ratio + 0.2, 1.)
            arguments.Config["bab"]["attack"]["max_dive_fix_ratio"] = new_max_dive_fix_ratio
            print(f"### topdown most timeout! increase max_dive_fix_ratio to {new_max_dive_fix_ratio}")
        # set back status to normal
        find_promising_domains.topdown_status = "normal"

    if find_promising_domains.current_method == 'bottom-up':
        # automatically adjust bottom up min_local_free_ratio
        min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
        if min_local_free_ratio > 0. and find_promising_domains.bottomup_status == "timeout":
            # rerun bottomup search with the decreased min_local_free_ratio
            print(f"orig min_local_free_ratio: {min_local_free_ratio}, fix: {min_local_free}")
            new_min_local_free_ratio = max(min_local_free_ratio - 0.1, 0.01)
            min_local_free = int(min_local_free / min_local_free_ratio * new_min_local_free_ratio)
            arguments.Config["bab"]["attack"]["min_local_free_ratio"] = new_min_local_free_ratio
            print(f"### bottom-up most timeout! decrease min_local_free_ratio to {new_min_local_free_ratio}, fix: {min_local_free}")
        find_promising_domains.bottomup_status = "normal"

    if find_promising_domains.current_method == 'top-down':
        # Try adversarial example local search.
        # Common adversarial pattern. Always used.
        find_promising_domains.current_method = 'bottom-up'
        print('Bottom-Up: Constructing sub-MIPs from current adversarial examples.')
        adv_split, adv_coeff = adv_pool.get_activation_pattern_from_pool()
        all_splits = [adv_split.tolist()]
        all_coeffs = [adv_coeff.tolist()]
        all_advs = [adv_pool.adv_pool[0].x.unsqueeze(0)]
        all_act_patterns = [[p.unsqueeze(0) for p in adv_pool.adv_pool[0].activation_pattern]]
        for i in range(n_domains - 1):
            # Add an adversarial example.
            if i < len(adv_pool.adv_pool):
                uncommon_split, _ = adv_pool.get_ranked_activation_pattern(n_activations=min_local_free, find_uncommon=True, random_keep=True)
                adv_s, adv_c = adv_pool.get_activation_pattern(adv_pool.adv_pool[i], blacklist=uncommon_split)
                all_splits.append(adv_s)
                all_coeffs.append(adv_c)
                all_advs.append(adv_pool.adv_pool[i].x.unsqueeze(0))
                all_act_patterns.append([p.unsqueeze(0) for p in adv_pool.adv_pool[i].activation_pattern])
    else:
        find_promising_domains.current_method = 'top-down'
        print('Top-Down: Constructing sub-MIPs from beam search domains.')
        diving_fix = []
        # Try diving domains.
        for i in range(n_domains):
            # Add this domain for MIP solving.
            if i < len(dive_domains):
                s, c = history_to_splits(dive_domains[i].history)
                diving_fix.append(max(max_dive_fix - len(s), 0))
                # Add adv diving.
                if max_dive_fix > len(s):
                    adv_s, adv_c = adv_pool.get_ranked_activation_pattern(n_activations=max_dive_fix - len(s), blacklist=s, random_keep=True)
                    s = torch.cat([torch.tensor(s), adv_s], dim=0)
                    c = torch.cat([torch.tensor(c), adv_c], dim=0)
                all_splits.append(s)
                all_coeffs.append(c)
                all_advs.append(None)
                all_act_patterns.append(None)
        print(f"{diving_fix} neurons fixed by diving.")
    print(f"Generating sub-MIPs with {[len(s) for s in all_splits]} fixed neurons.")
    return all_splits, all_coeffs, all_advs, all_act_patterns


def beam_alpha_crown(net, splits, coeffs):

    batch = len(splits)
    # reset beta to None
    for mi, m in enumerate(net.net.relus):
        m.sparse_beta = None
        m.sparse_beta_loc = None
        m.sparse_beta_sign = None
    
    # set slope alpha
    slope = net.refined_slope
    spec_name = net.net.final_name
    for m in net.net.relus:
        for spec_name in slope[m.name].keys():
            m.alpha[spec_name] = slope[m.name][spec_name].repeat(1, 1, batch, *([1] * (slope[m.name][spec_name].ndim - 3))).detach().requires_grad_()

    # repeat lower and upper bounds according to batch size
    lower_bounds, upper_bounds = [], []
    for refined_lower_bounds, refined_upper_bounds in zip(net.refined_lower_bounds, net.refined_upper_bounds):
        if refined_lower_bounds.ndim == 4:
            lower_bounds.append(refined_lower_bounds.repeat(batch,1,1,1).detach())
            upper_bounds.append(refined_upper_bounds.repeat(batch,1,1,1).detach())
        else:
            lower_bounds.append(refined_lower_bounds.repeat(batch,1).detach())
            upper_bounds.append(refined_upper_bounds.repeat(batch,1).detach())

    # update bounds with splits
    for bi in range(batch):
        split, coeff = splits[bi], coeffs[bi]
        for s, c in zip(split, coeff):
            # splits for each batch
            relu_layer, neuron_idx = s
            if c == 1:
                lower_bounds[relu_layer].view(batch, -1)[bi, neuron_idx] = 0
            else:
                upper_bounds[relu_layer].view(batch, -1)[bi, neuron_idx] = 0

    new_interval, reference_bounds = {}, {}
    for i, layer in enumerate(net.net.relus):
        nd = layer.inputs[0].name
        if i == 0:
            print("new interval:", i, nd, net.refined_lower_bounds[i].shape, lower_bounds[i].shape)
            new_interval[nd] = [lower_bounds[i], upper_bounds[i]]
        else:
            print("reference bounds:", i, nd, net.refined_lower_bounds[i].shape, lower_bounds[i].shape)
            reference_bounds[nd] = [lower_bounds[i], upper_bounds[i]]

    ptb = PerturbationLpNorm(norm=net.x.ptb.norm, eps=net.x.ptb.eps,
                                x_L=net.x.ptb.x_L.repeat(batch, 1, 1, 1),
                                x_U=net.x.ptb.x_U.repeat(batch, 1, 1, 1))
    new_x = BoundedTensor(net.x.data.repeat(batch, 1, 1, 1), ptb)
    c = None if net.c is None else net.c.repeat(new_x.shape[0], 1, 1)

    lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
    init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
    share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
    no_joint_opt = arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]
    optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
    lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
    
    net.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': init_iteration, 'ob_beta': False, 'ob_alpha': True,
                                'ob_alpha_share_slopes': False, 'ob_optimizer': optimizer,
                                'ob_early_stop': False, 'ob_verbose': 0,
                                'ob_keep_best': True, 'ob_update_by_layer': True,
                                'ob_lr': lr_init_alpha, 'ob_init': False,
                                'ob_lr_decay': lr_decay}})
    lb, _ = net.net.compute_bounds(x=(new_x,), C=c, method='crown-optimized', 
                                new_interval=new_interval, reference_bounds=reference_bounds, 
                                bound_upper=False, needed_A_dict=net.needed_A_dict)
    lower_bounds_new, upper_bounds_new = net.get_candidate_parallel(net.net, lb, lb + 99, batch)

    return lower_bounds_new, upper_bounds_new


def beam_mip_attack(net, adv_pool, dive_domains, mip_dive_start, max_dive_fix, min_local_free, finalize=False, attack_args=None):
    def parse_results(res):
        solver_results = res.get()
        upper_bounds, lower_bounds, status, solutions = zip(*solver_results)
        solutions = torch.cat(solutions, dim=0)  # Each MIP worker may return 0 or 1 solution.
        print('Sub-MIP Method:', find_promising_domains.current_method)
        print('Got MIP ub:', [f"{ub:.5f}" for ub in upper_bounds])
        print('Got MIP lb:', [f"{lb:.5f}" for lb in lower_bounds])
        print('Got MIP status:', status)
        print('Got MIP solutions:', solutions.size())
        # collect status for submip
        mip_status = "normal"
        inf_cnt = 0
        timeout_cnt = 0
        for st in status:
            if st == 9:
                timeout_cnt += 1
            if st == 3:
                inf_cnt += 1
        if inf_cnt >= (len(upper_bounds) // 2):
            mip_status = "infeas"
        if timeout_cnt >= (len(upper_bounds) // 2):
            mip_status = "timeout"
        
        if find_promising_domains.current_method == "top-down":
            find_promising_domains.topdown_status = "normal"
            find_promising_domains.topdown_status = mip_status
            print(f"### topdown status {find_promising_domains.topdown_status}")
        else:
            find_promising_domains.bottomup_status = "normal"
            find_promising_domains.bottomup_status = mip_status
            print(f"### bottomup status {find_promising_domains.bottomup_status}")
        return min(upper_bounds), solutions
    mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
    # Wait for last batch of MIP to finish.
    attack_success = False
    min_bound = float("inf")
    solutions = None
    if finalize and beam_mip_attack.started:
        print('Waiting MIP Solver to finalize...')
        return parse_results(net.pool_result)
    else:
        if net.pool_result is not None:
            # Checking if last batch of MIP has finished.
            if not net.pool_result.ready():
                print('MIP solver still running. Waiting for the next iteration.')
                return float("inf"), None
            else:
                # Get results from the last batch and run a new batch.
                min_bound, solutions = parse_results(net.pool_result)
                attack_success = min_bound < 0
        
        if solutions is not None and solutions.size(0) != 0:
            # Add MIP solutions to the pool.
            solutions = solutions.to(net.net.device)
            mip_pred = net.net(solutions).cpu()
            mip_margins = mip_pred.matmul(net.c.cpu()[0].transpose(-1, -2)).squeeze(-1)
            # Convert to margin via the C matrix.
            #### DOUBLE CHECK PGD ATTACK RESTARTS
            # attack_args.update({'x': solutions, 'restarts': 1, 'initialization': 'none'})
            attack_args.update({'x': solutions, 'initialization': 'none'})
            attack_ret, attack_images, attack_margin = pgd_attack(**attack_args)
            adv_pool.add_adv_images(attack_images)
            adv_pool.print_pool_status()
            print(f'mip ub: {min(mip_margins.view(-1)).item()} -> mip ub (PGD): {min(attack_margin)}, best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')

        if not attack_success:
            # Run new MIPs based on selected domains.
            splits, coeffs, advs, act_patterns = find_promising_domains(net, adv_pool, dive_domains, mip_multi_proc, mip_dive_start, max_dive_fix, min_local_free)
            if len(splits) > 0:
                print('Start to run MIP!')
                refined_lower_bounds, refined_upper_bounds = None, None
                if arguments.Config["bab"]["attack"]["refined_mip_attacker"]:
                    refined_batch_size = arguments.Config["bab"]["attack"]["refined_batch_size"]
                    batch = len(splits)
                    if refined_batch_size is None or batch <= refined_batch_size:
                        refined_lower_bounds, refined_upper_bounds = beam_alpha_crown(net, splits, coeffs)
                    else:
                        refined_lower_bounds = [[] for _ in net.refined_lower_bounds]
                        refined_upper_bounds = [[] for _ in net.refined_upper_bounds]
                        start_batch, end_batch = 0, refined_batch_size
                        while start_batch < batch:
                            rlbs, rubs = beam_alpha_crown(net, splits[start_batch: end_batch], coeffs[start_batch: end_batch])
                            for relu_idx, (rlb, rub) in enumerate(zip(rlbs, rubs)):
                                refined_lower_bounds[relu_idx].append(rlb)
                                refined_upper_bounds[relu_idx].append(rub)
                            start_batch += refined_batch_size
                            end_batch += refined_batch_size

                        for relu_idx in range(len(refined_lower_bounds)):
                            refined_lower_bounds[relu_idx] = torch.cat(refined_lower_bounds[relu_idx])
                            refined_upper_bounds[relu_idx] = torch.cat(refined_upper_bounds[relu_idx])
                            assert refined_lower_bounds[relu_idx].size(0) == batch, f"refined_batch_size process wrong, {relu_idx}, {refined_lower_bounds[relu_idx].size(0)} != {batch}!"

                net.update_mip_model_fix_relu(splits, coeffs, target=None,
                        async_mip=True, best_adv=advs, adv_activation_pattern=act_patterns,
                        refined_lower_bounds=refined_lower_bounds, refined_upper_bounds=refined_upper_bounds)
                beam_mip_attack.started = True
    if min_bound > 0:
        # ObjBound was returned. This is not an upper bound.
        return float("inf"), solutions
    else:
        return min_bound, solutions


def probabilistic_select_domains(dive_domains, candidates_number):
    softmax_temperature = 0.1
    new_domains = type(dive_domains)()
    # Always Keey domains with non-zero priorities.
    removed_domains = []
    for i, d in enumerate(dive_domains):
        if d.priority > 0:
            # Shallow copy this domain.
            new_domains.add(copy.copy(d))
            # Make sure this domain will not be selected again later.
            removed_domains.append(i)
    for r in reversed(removed_domains):
        dive_domains.pop(r)
    lbs = torch.tensor([d.lower_bound for d in dive_domains])
    # Select candidates_number - domains with priority.
    remain_domains = min(len(dive_domains), candidates_number) - len(new_domains)
    if remain_domains > 0:
        # probs = -lbs / lbs.abs().sum()
        normalized_lbs = -lbs / lbs.neg().max()
        probs = torch.nn.functional.softmax(normalized_lbs / softmax_temperature, dim=0)
        # Choose domains based on sampling probability.
        selected_indices = probs.multinomial(remain_domains, replacement=False)
        for i in selected_indices:
            new_domains.add(dive_domains[i])
        print(f'Probabilistic domain selection: probability are {probs[len(probs)//100]}@0.01, '
              f'{probs[len(probs)//20]}@0.05, {probs[len(probs)//10]}@0.1, {probs[len(probs)//5]}@0.2, {probs[len(probs)//2]}@0.5')
    del dive_domains
    return new_domains


def bab_attack(dive_domains, net, batch, pre_relu_indices, growth_rate, layer_set_bound=True,
                       adv_pool=None, attack_args=None, max_dive_fix=float("inf"), min_local_free=0):

    decision_thresh = arguments.Config["bab"]["decision_thresh"]
    branching_method = arguments.Config['bab']['branching']['method']
    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    DFS_percent = arguments.Config["bab"]["dfs_percent"]
    num_dive_constraints = arguments.Config["bab"]["attack"]["num_dive_constraints"]
    candidates_number = arguments.Config["bab"]["attack"]["beam_candidates"]
    split_depth = arguments.Config["bab"]["attack"]["beam_depth"]
    mip_dive_start = arguments.Config["bab"]["attack"]["mip_start_iteration"]

    def merge_split_decisions(all_decisions, top_k=7):
        """Merge a list of list of decisions, and pick the top-k of decisions."""
        n_examples = len(all_decisions)
        flat_decisions = [tuple(decision) for example_decisions in all_decisions for decision in example_decisions]
        counter = Counter(flat_decisions)
        return [[list(c[0]) for c in counter.most_common(top_k)]] * n_examples

    # dive_domains store the dive domains
    global diving_Visited   # FIXME (10/21): Do not use global variables.

    total_time = time.time()

    pickout_time = time.time()
    print(f"iteration starts with dive domain length: {len(dive_domains)}")
    dive_domains = probabilistic_select_domains(dive_domains, candidates_number)
    print(f"prune dive domains to be length {len(dive_domains)}")
    # pickout the worst candidates_number domains
    domains_params = pick_out_batch(dive_domains, batch=candidates_number, device=net.x.device)
    mask, lAs, orig_lbs, orig_ubs, slopes, betas, _, selected_domains = domains_params

    ###### Maybe we can apply integer fix here #######

    # throw away all the rest dive domains
    dive_domains = DFS_SortedList() if DFS_percent > 0 else SortedList()
    pickout_time = time.time() - pickout_time
    
    # for each domain in dive_domains, select k (7 or 10) split decisions with kfsb
    decision_time = time.time()
    history = [sd.history for sd in selected_domains]
    split_history = [sd.split_history for sd in selected_domains]
    assert branching_method == "kfsb"
    # we need to select k decisions for each dive domain
    dive_decisions, _ = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                    branching_candidates=split_depth,
                                    branching_reduceop=branching_reduceop,
                                    slopes=slopes, betas=betas, history=history,
                                    keep_all_decision=True,
                                    prioritize_slopes="none")   # Change to "negative" to branch on upper bound first.
    # In beam search, we use the same branching decisions for all nodes, so we merge branching decisions for different nodes into 1.
    merged_decisions = merge_split_decisions(dive_decisions, len(dive_decisions[0]))
    print('splitting decisions: {}'.format(merged_decisions[0]))
    if len(merged_decisions[0]) == 0:
        raise RuntimeError("Attack success might be possible, but lb and ub do not match. Need to fix!")
    decision_time = time.time() - decision_time

    dive_time = time.time()
    # fill up the dive domains to be max_dive_domains (1024)
    # TODO: we should allow a loop here to run very large batch size. This is useful for CIFAR.
    domains_params = add_dive_domain_from_dive_decisions(selected_domains, merged_decisions, mask=mask, device=net.x.device)
    mask, orig_lbs, orig_ubs, orig_slopes, orig_betas, selected_domains = domains_params
    orig_history = [sd.history for sd in selected_domains]
    orig_split_history = [[] for i in range(len(orig_history))]  # This is not used in diving.
    dive_time = time.time() - dive_time

    solve_time = time.time()
    dom_ub, dom_lb, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas = [], [], [], [], [], [], [], []
    primals = None
    # Divide all domains into multiple batches.
    batch_boundaries = torch.arange(0, len(orig_betas), batch).tolist() + [len(orig_betas)]
    for i in range(len(batch_boundaries) - 1):
        batch_start, batch_end = batch_boundaries[i], batch_boundaries[i+1]
        # orig_lbs and orig_ubs are organized per layer.
        batch_orig_lbs = [single_lbs[batch_start:batch_end] for single_lbs in orig_lbs]
        batch_orig_ubs = [single_ubs[batch_start:batch_end] for single_ubs in orig_ubs]

        batch_slopes = defaultdict.fromkeys(orig_slopes.keys(), None)
        for m_name in list(orig_slopes.keys()):
            for spec_name in orig_slopes[m_name]:
                batch_slopes[m_name] = {spec_name: orig_slopes[m_name][spec_name][:, :, batch_start:batch_end]}

        batch_history = orig_history[batch_start:batch_end]
        batch_split_history = orig_split_history[batch_start:batch_end]
        batch_betas = orig_betas[batch_start:batch_end]
        batch_split = {'decision': [], 'coeffs': [], 'diving': len(batch_history)}
        ret = net.get_lower_bound(batch_orig_lbs, batch_orig_ubs, batch_split, slopes=batch_slopes,
                                  history=batch_history, split_history=batch_split_history, betas=batch_betas,
                                  layer_set_bound=layer_set_bound, single_node_split=True, intermediate_betas=None)
        batch_dom_ub, batch_dom_lb, _, batch_lAs, batch_dom_lb_all, batch_dom_ub_all, batch_slopes, batch_split_history, batch_betas, _, batch_primals = ret
        for full_list, partial_list in ((dom_ub, batch_dom_ub), (dom_lb, batch_dom_lb), (lAs, batch_lAs), (dom_lb_all, batch_dom_lb_all),
                (dom_ub_all, batch_dom_ub_all), (slopes, batch_slopes), (split_history, batch_split_history), (betas, batch_betas)):
            full_list.extend(partial_list)
        # "batch_primals" is a tensor.
        if primals is None:
            primals = batch_primals
        else:
            primals = torch.cat([primals, batch_primals], dim=0)

    if adv_pool is not None:
        adv_imgs = primals[np.argsort(dom_ub)[:50]]  # we only select best adv_imgs according to their upper bounds
        #### DOUBLE CHECK PGD ATTACK RESTARTS
        # attack_args.update({'x': adv_imgs, 'restarts': 1, 'initialization': 'none'})
        attack_args.update({'x': adv_imgs, 'initialization': 'none'})
        attack_ret, attack_images, attack_margin = pgd_attack(**attack_args)
        adv_pool.add_adv_images(attack_images)
        adv_pool.print_pool_status()
        print(f'ub: {min(dom_ub)} -> ub (PGD): {min(attack_margin)}, best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')

    solve_time = time.time() - solve_time
    add_time = time.time()

    # See how these neurons are set in adv. examples, and we keep that domain for searching.
    activations = adv_pool.find_most_likely_activation(merged_decisions[0])
    # Get all the domains with this specific activations.
    activations = [str((aa + 1) // 2) for aa in activations]  # convert to 1, 0 instead of +1, -1
    domain_idx = int("".join(activations), base=2)
    priorities = torch.zeros(len(dom_lb))
    priorities[domain_idx] = 1.0
    print(f'decision in adv example {activations}, domain size {len(dom_lb)}')

    # add all 1024 domains back into dive domains
    ####### will not add extra split constraints here by default, set add_constraints=True if want to ######
    diving_unsat_list = add_dive_domain_parallel(lA=lAs, lb=dom_lb, ub=dom_ub,
                                            lb_all=dom_lb_all, ub_all=dom_ub_all,
                                            dive_domains=dive_domains, selected_domains=selected_domains,
                                            slope=slopes, beta=betas, decision_thresh=decision_thresh,
                                            split_history=split_history, check_infeasibility=False,
                                            num_dive_constraints=num_dive_constraints, primals=primals,
                                            add_constraints=False, priorities=priorities)
                                            
    print('Diving domain [lb, ub] (depth):')
    for i in dive_domains[:50]:
        if i.priority != 0:
            prio = f'(prio={i.priority:.2f})'
        else:
            prio = ""
        print(f'[{i.lower_bound:.5f}, {i.upper_bound:.5f}] ({i.depth}){prio}', end=', ')
    print()
    
    diving_Visited += len(selected_domains)
    print('Current worst domains:', [i.lower_bound for i in dive_domains[:10]])
    add_time = time.time() - add_time

    total_time = time.time() - total_time
    print('length of dive domains:', len(dive_domains))
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t dive: {dive_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')
    
    dive_domains = probabilistic_select_domains(dive_domains, max(candidates_number, arguments.Config["solver"]["mip"]["parallel_solvers"]))
    print(f"prune dive domains to {len(dive_domains)} according to probabilistic_select_domains()")

    # Run MIP for final solving adversarial examples.
    mip_ub, solutions = beam_mip_attack(net, adv_pool, dive_domains, mip_dive_start, max_dive_fix, min_local_free, finalize=len(dive_domains) == 0, attack_args=attack_args)
    
    if len(dive_domains) > 0:
        global_lb = dive_domains[0].lower_bound
    else:
        print("No dive domains left, attack failed, please increase search candidates!")
        return torch.tensor(-np.inf), np.inf, dive_domains

    # check dom_ub for adv
    batch_ub = np.inf
    if get_upper_bound:
        if adv_pool is not None:
            batch_ub = min(min(dom_ub), mip_ub, adv_pool.adv_pool[0].obj)
        else:
            batch_ub = min(dom_ub)
        print(f"Current lb:{global_lb}, ub:{batch_ub}")
    else:
        print(f"Current lb:{global_lb}")

    print('{} diving domains visited'.format(diving_Visited))

    return global_lb, batch_ub, dive_domains


def count_domain_unstable(domain):
    unstable_cnt = []
    for relu_idx, (lb, ub) in enumerate(zip(domain.lower_all[:-1],  domain.upper_all[:-1])):
        unstable_cnt.append(torch.logical_and(lb < 0, ub > 0).sum().item())
    print("remaining unstable neurons:", unstable_cnt)


def random_heuristic(dive_domain, num_dive_constraints):

    lower_bounds, upper_bounds = dive_domain.lower_all, dive_domain.upper_all
    device = lower_bounds[0].device
    decision = []
    coeffs = []
    all_layer_unstable_idx = []
    all_layer_neurons = []

    for relu_idx, (lb, ub) in enumerate(zip(lower_bounds[:-1], upper_bounds[:-1])):
        neurons = lb.view(-1).shape[0]
        unstable_idx = torch.logical_and(lb < 0, ub > 0).view(-1).nonzero()
        unstable_idx = torch.cat([torch.zeros(unstable_idx.shape,\
                    device=device) + relu_idx, unstable_idx], dim=1)
        all_layer_unstable_idx.append(unstable_idx)
        all_layer_neurons.append(neurons)

    all_layer_unstable_idx = torch.cat(all_layer_unstable_idx, dim=0)
    # print(f"unstable neurons {all_layer_unstable_idx.shape[0]}/{sum(all_layer_neurons)}")

    if all_layer_unstable_idx.shape[0] >= num_dive_constraints:
        random_idx = torch.randint(low=0, high=all_layer_unstable_idx.shape[0],
                    size=(num_dive_constraints,), device=device)
    else:
        # number of unstable neurons are not enough compared to num_dive_constraints, take all the rest
        random_idx = torch.arange(all_layer_unstable_idx.shape[0], device=device)
    decision = all_layer_unstable_idx[random_idx].long()
    coeffs = torch.randint(low=0, high=2, size=(decision.shape[0],), device=device) * 2 - 1
    return {"decision": decision, "coeffs": coeffs}


def adv_heuristic(dive_domain, attack_images, net, mask):

    net.net(attack_images)
    decision = []
    coeffs = []

    for i, layer in enumerate(net.net.relus):
        activation = layer.inputs[0].forward_value.flatten(1)
        active_idx = ((torch.all(activation > 0, dim=0) == True) * mask[i]).nonzero(as_tuple=False)
        active_idx[:, 0] = i
        inactive_idx = ((torch.all(activation <= 0, dim=0) == True) * mask[i]).nonzero(as_tuple=False)
        inactive_idx[:, 0] = i

        for d in active_idx.unsqueeze(1):
            decision.append(d)
            coeffs.append(1)
        for d in inactive_idx.unsqueeze(1):
            decision.append(d)
            coeffs.append(-1)

    decision = torch.cat(decision, dim=0).long()
    coeffs = torch.tensor(coeffs)

    return {"decision": decision, "coeffs": coeffs}


def integer_heuristic(dive_domain, num_dive_constraints=None, source='dual'):

    if source == 'dual' or source == 'both':
        # Recover slope sides from primal solutions.
        decision = []
        coeffs = []
        # Threshold for determining setting z to 0 or 1.
        threshold = 1e-3
        lower_bounds, upper_bounds, lAs, slopes = dive_domain.lower_all, dive_domain.upper_all, dive_domain.lA, dive_domain.slope
        batch_size = lAs[0].size(0)
        assert batch_size == 1
        device = lAs[0].device
        # FIXME: for lower and upper bounds, we also need a dictionary rather than a list, otherwise they cannot easily match lAs.
        layer_names = sorted([int(s[1:]) for s in slopes.keys() if isinstance(s, str)])
        slope_list = []
        # FIXME: this is very dirty hack. We must change the lower_all and upper_all types.
        for layer in layer_names:
            str_name = f"/{layer}"
            keys = list(slopes[str_name].keys())
            assert len(keys) == 1
            slope_list.append(slopes[str_name][keys[0]][0,0])  # lower bound dimension, only 1 output specification.
        # Go over all layers:
        for relu_idx in range(len(lAs)):
            alpha = slope_list[relu_idx].view(-1)
            lA_mask = (lAs[relu_idx] > 0.1).view(-1)  # lA is positive, and lower bound is selected.
            unstable_mask = torch.logical_and(lower_bounds[relu_idx] < 0, upper_bounds[relu_idx] > 0).view(-1)
            dive_mask = torch.logical_and(lA_mask, unstable_mask)
            # Set these neurons to always active: alpha=1 so the \hat{x} >= x bound is binding, or alpha != 0 (both are binding).
            set_z0 = torch.logical_and(alpha <= threshold, dive_mask).nonzero()
            # Set these neurons to always inactive: alpha=0 so the \hat{x} = 0 bound is binding, or alpha != 0 (both are binding).
            set_z1 = torch.logical_and(alpha > 1.0 - threshold, dive_mask).nonzero()
            set_z0_idx = torch.cat([torch.zeros(set_z0.shape, device=device) + relu_idx, set_z0], dim=1)
            set_z1_idx = torch.cat([torch.zeros(set_z1.shape, device=device) + relu_idx, set_z1], dim=1)
            decision.append(set_z1_idx)
            decision.append(set_z0_idx)
            coeffs = coeffs + [1] * set_z1_idx.size(0) + [-1] * set_z0_idx.size(0)

        decision = dual_decision = torch.cat(decision, dim=0).long()
        coeffs = dual_coeffs = torch.tensor(coeffs, device=device)

    if source == 'primal' or source == 'both':
        # Recover slope sides from primal solutions.
        if dive_domain.primals is None:
            raise RuntimeError("No primal values available, please enable get_upper_bound!")
        decision = []
        coeffs = []

        ###### Customize the integer heuristic according to solve_diving_lp function
        ###### make sure the beta crown outputs will not change!!!!

        lAs, primal_values, integer_values = dive_domain.lA, dive_domain.primals["p"], dive_domain.primals["z"]
        # the first and last primal values are input and output, the rest are pre_relu primals
        primal_values = primal_values[1:-1]
        for relu_idx in range(len(lAs)):
            pv = primal_values[relu_idx].view(-1)
            iv = integer_values[relu_idx].view(-1)
            lA = lAs[relu_idx].view(-1).to(iv.device)

            unstable_mask = (iv != -1)
            # unstable and primal on lower bounds
            lower_unstable_mask = torch.logical_and(lA > 0, unstable_mask)

            # we can set either 0 or 1 for pv == 0 and not changing the output
            set_z1 = lower_unstable_mask.logical_and(pv >= 0).nonzero()
            set_z0 = lower_unstable_mask.logical_and(pv < 0).nonzero()

            set_z1_idx = torch.cat([torch.zeros(set_z1.shape, device=iv.device) + relu_idx, set_z1], dim=1)
            set_z0_idx = torch.cat([torch.zeros(set_z0.shape, device=iv.device) + relu_idx, set_z0], dim=1)
            decision.append(set_z1_idx)
            decision.append(set_z0_idx)
            coeffs = coeffs + [1] * set_z1_idx.size(0) + [-1] * set_z0_idx.size(0)

            # print(f"relu_idx {relu_idx}, set {set_z1_idx.size(0)} z to 1 and {set_z0_idx.size(0)} z to 0,"\
            #         f" total set unstable {set_z1_idx.size(0)+set_z0_idx.size(0)}/{unstable_mask.sum().item()}")

        decision = primal_decision = torch.cat(decision, dim=0).long()
        coeffs = primal_coeffs = torch.tensor(coeffs)

    if source == 'both' and False:
        # compare both results.
        primal_decision_list = primal_decision.tolist()
        dual_decision_list = dual_decision.tolist()
        difference_detected = False
        for i, d in enumerate(primal_decision_list):
            c = primal_coeffs[i]
            try:
                idx = dual_decision_list.index(d)
            except ValueError:
                print(f"decision {d} not in dual decision list!")
                difference_detected = True
            else:
                if c != dual_coeffs[idx]:
                    print(f"coeffs for decision {d} is different: primal {c}, dual {dual_coeffs[idx].item()}")
                    difference_detected = True
        if difference_detected:
            pass

    return {"decision": decision, "coeffs": coeffs}


def dive_heuristic(dive_domain, num_dive_constraints=0, method="integer", attack_images=None, net=None, mask=None):
    # given a domain, return the list of nodes need to be split
    if method == "random":
        dive_splits = random_heuristic(dive_domain, num_dive_constraints)
    
    elif method == 'adv':
        # dive_splits = integer_heuristic(dive_domain, num_dive_constraints)
        # add_dive_constraints(dive_domain, dive_splits)
        dive_splits = adv_heuristic(dive_domain, attack_images, net, mask)

    elif method == "integer":
        dive_splits = integer_heuristic(dive_domain, num_dive_constraints)
        # new_num_dive_constraints = dive_splits["coeffs"].size(0)
        # if new_num_dive_constraints < num_dive_constraints:
        #     add_dive_constraints(dive_domain, dive_splits)
        #     dive_splits = random_heuristic(dive_domain, num_dive_constraints - new_num_dive_constraints)
    
    else:
        print("dive heuristic method not supported yet!")
        exit()
    return dive_splits


def add_dive_domains_from_domains(domains, dive_domains, num_dive_constraints=50):
    for d in domains:
        assert d.valid
        dive_d = d.clone_to_dive()
        dive_splits = dive_heuristic(dive_d, num_dive_constraints)
        add_dive_constraints(dive_d, dive_splits)
        # count_domain_unstable(dive_d)

        # customize the sort function later
        dive_domains.add(dive_d)


def add_dive_domain_from_adv(dive_domains, candidate_domain, attack_images, net, mask):
    # construct dive domain from attack_images, only support one candidate_domain

    dive_d = candidate_domain.clone_to_dive()
    dive_splits = dive_heuristic(dive_d, method='adv', attack_images=attack_images, net=net, mask=mask)

    add_dive_constraints(dive_d, dive_splits)
    # print("adv dive unstable count:")
    # count_domain_unstable(dive_d)

    dive_domains.add(dive_d)


def bfs_splits_coeffs(num_splits):
    # parse all possible coeffs combinations consider the number of splits
    return [[int((float(c) - 0.5) * 2) for c in f"{{:0{num_splits}b}}".format(i)] for i in range(2**num_splits)]


def add_dive_domain_from_dive_decisions(dive_domains, dive_decisions, DFS_percent=0., mask=None, device='cuda'):
    new_dive_domains = []  # DFS_SortedList() if DFS_percent > 0 else SortedList()

    dive_coeffs = {}

    # merged_new_lAs = []
    merged_lower_bounds = []
    merged_upper_bounds = []
    betas_all = []
    slopes_all = []
    ret_s = defaultdict.fromkeys(dive_domains[0].slope.keys(), None)
    # intermediate_betas_all = []

    for di, dive_d in enumerate(dive_domains):
        decision = torch.tensor(dive_decisions[di], device='cpu').long()
        num_splits = len(dive_decisions[di])
        repeats = 2 ** num_splits

        expand_lb = [dive_d.lower_all[i].repeat(repeats, *([1] * (dive_d.lower_all[i].ndim - 1))) for i in range(len(dive_d.lower_all))]
        expand_ub = [dive_d.upper_all[i].repeat(repeats, *([1] * (dive_d.upper_all[i].ndim - 1))) for i in range(len(dive_d.upper_all))]

        if num_splits not in dive_coeffs:
            dive_coeffs[num_splits] = bfs_splits_coeffs(num_splits)
        # Generate beta. All subdomains generated by this domain has the same beta.
        # All subdomains also share the same decisions. They just have different coeffs in history.
        if dive_d.beta is None:
            # No existing beta, so generate a new one.
            new_beta = [torch.zeros(size=(0,), device=device) for _ in range(len(dive_d.history))]
        else:
            # Reuse existing beta in dive_d. This should be a cuda tensor.
            new_beta = dive_d.beta
        # For all subdomains we add the same decisions.
        decision_to_add_per_layer = []
        # Store which neurons are selected per layer.
        layer_idx_mask = []
        for layer_idx in range(len(dive_d.history)):
            # Adding new decisions for this layer.
            idx_mask = (decision[:, 0] == layer_idx)
            # Save this mask to be used later.
            layer_idx_mask.append(idx_mask)
            if idx_mask.size(0) == 0:
                # no node selected in this layer
                decision_to_add_per_layer.append(None)
                continue
            # Finding the location of splits in this layer.
            dive_loc = decision[idx_mask][:, 1].long()
            decision_to_add_per_layer.append(dive_d.history[layer_idx][0] + dive_loc.tolist())
            # Adding zeros to beta.
            new_beta[layer_idx] = torch.cat([new_beta[layer_idx], torch.zeros(dive_loc.size(0), device=device)])
        # Repeat beta for each layer, and add views to the list.
        new_beta = [b.view(1, -1).repeat(repeats, 1) for b in new_beta]
        for i in range(repeats):
            betas_all.append([b[i] for b in new_beta])  # This is just a view and will be added very quickly.

        # Deal with split history. This has to be done per-domain, howere there are very few tensor operations here.
        dive_coeffs_t = torch.tensor(dive_coeffs[num_splits], device='cpu')
        for i in range(repeats):
            new_dive_d = dive_d.clone_to_dive(beam_search=True)  # This will copy nothing.
            # This is just for generating the history. In each subdomain, only the history is different.
            for layer_idx in range(len(dive_d.history)):
                idx_mask = layer_idx_mask[layer_idx]
                if idx_mask.size(0) == 0:
                    # no node selected in this layer
                    continue
                this_layer_dive_coeffs = dive_coeffs_t[i][idx_mask]
                # add new dive constraints to dive domain history.
                new_dive_d.history[layer_idx][0] = decision_to_add_per_layer[layer_idx]
                new_dive_d.history[layer_idx][1] = dive_d.history[layer_idx][1] + this_layer_dive_coeffs.tolist()
            new_dive_d.depth += decision.size(0)
            new_dive_domains.append(new_dive_d)

        coeffs = np.array(dive_coeffs[num_splits])
        decisions = np.repeat(np.expand_dims(np.array(dive_decisions[di]), 0), repeats, axis=0)  # 1024, 10, 2
        zero_coeffs = np.argwhere(coeffs == -1)  # 5120, 2
        one_coeffs = np.argwhere(coeffs == 1)  # 5120, 2

        zero_idx = decisions[zero_coeffs[:, 0], zero_coeffs[:, 1]]  # 5120, 2
        one_idx = decisions[one_coeffs[:, 0], one_coeffs[:, 1]]  # 5120, 2
        for i in range(len(expand_lb)):
            selected_one_idx = np.argwhere(one_idx[:, 0] == i)
            selected_zero_idx = np.argwhere(zero_idx[:, 0] == i)

            if len(selected_one_idx) == 0 and len(selected_zero_idx) == 0:
                continue
            expand_lb[i].view(repeats, -1)[one_coeffs[:, 0][selected_one_idx.squeeze()], one_idx[selected_one_idx.squeeze()][:, 1]] = 0
            expand_ub[i].view(repeats, -1)[zero_coeffs[:, 0][selected_zero_idx.squeeze()], zero_idx[selected_zero_idx.squeeze()][:, 1]] = 0

        merged_lower_bounds.append(expand_lb)
        merged_upper_bounds.append(expand_ub)

        # for j in range(len(new_dive_d.lA)):
        #     new_lAs.append(new_dive_d.lA[j].repeat(repeats, *([1] * (len(new_dive_d.lA[j].shape) - 1))))
        # merged_new_lAs.append(new_lAs)

        assert isinstance(new_dive_d.slope, dict)
        tmp_slope = defaultdict.fromkeys(new_dive_d.slope.keys(), None)
        for m_name in list(new_dive_d.slope.keys()):
            for spec_name in new_dive_d.slope[m_name]:
                tmp_slope[m_name] = {spec_name: new_dive_d.slope[m_name][spec_name].repeat(1, 1, repeats,  *([1] * (new_dive_d.slope[m_name][spec_name].ndim - 3)))}
        slopes_all.append(tmp_slope)

        # intermediate_betas_all += [new_dive_d.intermediate_betas] * repeats

    ret_lbs = []
    for j in range(len(merged_lower_bounds[0])):
        ret_lbs.append(torch.cat([merged_lower_bounds[i][j] for i in range(len(merged_lower_bounds))]))

    ret_ubs = []
    for j in range(len(merged_upper_bounds[0])):
        ret_ubs.append(torch.cat([merged_upper_bounds[i][j] for i in range(len(merged_upper_bounds))]))

    ret_lbs = [t.to(device=device, non_blocking=True) for t in ret_lbs]
    ret_ubs = [t.to(device=device, non_blocking=True) for t in ret_ubs]

    for m_name in list(slopes_all[0].keys()):
        for spec_name in slopes_all[0][m_name]:
            ret_s[m_name] = {spec_name: torch.cat([slopes_all[i][m_name][spec_name] for i in range(len(slopes_all))], dim=2)}

    # Recompute the mask on GPU.
    new_masks = []
    for j in range(len(ret_lbs) - 1):  # Exclude the final output layer.
        new_masks.append(
            torch.logical_and(ret_lbs[j] < 0, ret_ubs[j] > 0).view(ret_lbs[0].size(0), -1).float())
    print(f"expand original {len(dive_domains)} selected dive domains to {len(new_dive_domains)} with {num_splits} splits")

    return new_masks, ret_lbs, ret_ubs, ret_s, betas_all, new_dive_domains


def add_dive_domain_parallel(lA, lb, ub, lb_all, ub_all, dive_domains, selected_domains, slope, beta,
                        split_history=None, decision_thresh=0,
                        check_infeasibility=True, num_dive_constraints=50, primals=None, add_constraints=True, priorities=None,
                        cs=None):
    """
    add domains into dive domains from params with num_dive_constraints extra splits for each new domain
    """
    unsat_list = []
    batch = len(selected_domains)
    if primals.is_cuda:
        primals = primals.cpu()
    for i in range(batch):

        infeasible = False
        if lb[i] < decision_thresh:
            if check_infeasibility:
                for ii, (l, u) in enumerate(zip(lb_all[i][1:-1], ub_all[i][1:-1])):
                    if (l-u).max() > 1e-6:
                        infeasible = True
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        break

            if not infeasible:
                priority=0 if priorities is None else priorities[i].item()
                new_history = copy.deepcopy(selected_domains[i].history)
                dive_primals = primals[i] if primals is not None else None
                dive_d = ReLUDomain(lA[i], lb[i], ub[i], lb_all[i], ub_all[i], slope[i], beta[i],
                                  selected_domains[i].depth+0, split_history=split_history[i],  # depth has been added during diving.
                                  history=new_history,
                                  primals=dive_primals, priority=priority,
                                  c=cs[i:i+1] if cs is not None else None)

                if add_constraints:
                    # add extra constraints to each new dive domain
                    dive_constraints = dive_heuristic(dive_d, num_dive_constraints)
                    add_dive_constraints(dive_d, dive_constraints)
                    # count_domain_unstable(dive_d)

                dive_domains.add(dive_d)

    return unsat_list


def add_dive_constraints(dive_domain, dive_splits, ignore_beta=False):
    # dive_domain.history = ...
    # dive beta to be 0 for the new constraints
    # dive domain lbs and ubs new constraints need to be set 0 accordingly
    device = dive_domain.lA[0].device
    if dive_splits is None:
        print("Warning: no dive splits applied")
        return

    if dive_domain.beta is None and not ignore_beta:
        dive_domain.beta = [torch.zeros(size=(0,), device=device) for _ in range(len(dive_domain.history))]
    
    for layer_idx in range(len(dive_domain.history)):
        idx_mask = (dive_splits["decision"][:, 0] == layer_idx)
        if idx_mask.size(0) == 0: 
            # no node selected in this layer
            continue
        dive_loc = dive_splits["decision"][idx_mask][:, 1].long()
        dive_coeffs = dive_splits["coeffs"][idx_mask]

        # add new dive constraints to dive domain history
        dive_domain.history[layer_idx][0] += dive_loc.tolist()
        dive_domain.history[layer_idx][1] += dive_coeffs.tolist()

        # set new dive constrainst betas to be 0
        if not ignore_beta:
            dive_domain.beta[layer_idx] = torch.cat([dive_domain.beta[layer_idx],\
                        torch.zeros(dive_loc.size(0), device=device)])

        # will set bounds outside
        # # set upper and lower bounds
        # coeff_zero_mask = (dive_coeffs == -1)
        # coeff_one_mask = (dive_coeffs == 1)
        # if coeff_zero_mask.sum() > 0:
        #     # the assert make sure the constraints are only applied on unstable nodes
        #     # and the constraints will not be added repeatedly on the same node
        #     assert dive_domain.upper_all[layer_idx].view(-1)[dive_loc[coeff_zero_mask]].min() > 0,\
        #                 f"{dive_domain.upper_all[layer_idx].view(-1)[dive_loc[coeff_zero_mask]].min().item()} > 0"
        #     dive_domain.upper_all[layer_idx].view(-1)[dive_loc[coeff_zero_mask]] = 0
        # if coeff_one_mask.sum() > 0:
        #     assert dive_domain.lower_all[layer_idx].view(-1)[dive_loc[coeff_one_mask]].max() < 0,\
        #             f"{dive_domain.lower_all[layer_idx].view(-1)[dive_loc[coeff_one_mask]].max().item()} < 0"
        #     dive_domain.lower_all[layer_idx].view(-1)[dive_loc[coeff_one_mask]] = 0

    dive_domain.depth += dive_splits['decision'].size(0)
    

def initial_attack_criterion(ubs, rhs):
    """ check whether attack successful """
    attacked_idx = torch.all((ubs - rhs) <= 0, dim=-1)  # in vnnlib, adv examples are defined by <= or >= only
    if attacked_idx.any():  # check if there is any input can be attacked
        print('Adversarial example found! ub - rhs:', (ubs - rhs)[attacked_idx], 'index:', attacked_idx)
        return True
    else:
        return False
