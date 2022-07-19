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
import time
import random
import numpy as np
import torch
import copy
from collections import defaultdict

from auto_LiRPA.utils import stop_criterion_sum, stop_criterion_batch_any, stop_criterion_batch_topk
from branching_domains import merge_domains_params, SortedReLUDomainList, BatchedReLUDomainList
### preprocessor-hint: private-section-start
# from branching_domains import add_cut_domain
### preprocessor-hint: private-section-end
from branching_heuristics import choose_node_parallel_FSB, choose_node_parallel_crown, choose_node_parallel_kFSB

import arguments

### preprocessor-hint: private-section-start
from branching_domains import select_batch
from adv_domains import AdvExamplePool
from bab_attack import add_dive_domain_parallel, beam_mip_attack, find_promising_domains, bab_attack, add_dive_domain_from_adv, add_dive_domains_from_domains
from cut_utils import fetch_cut_from_cplex, generate_cplex_cuts, clean_net_mps_process, cplex_update_general_beta
### preprocessor-hint: private-section-end

Visited, Flag_first_split = 0, True
Use_optimized_split = False
all_node_split = False
DFS_enabled = False
diving_Visited = 0
total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0


def build_history(history, split, orig_lbs, orig_ubs):
    '''
    Generate fake history and fake lower and upper bounds for new domains
    history: [num_domain], history of the input domains
    split: [num_copy * num_domain], split decision for each new domain.
    orig_lbs, orig_ubs: [num_relu_layer, num_copy, num_domain, relu_layer.shape]
    '''
    new_history = []
    num_domain = len(history)
    num_split = len(split)//num_domain

    num_layer = len(orig_lbs)

    def generate_history(heads, splits, orig_lbs, orig_ubs, domain_idx):
        '''
        Generate [num_copy] fake history and fake lower and upper bounds for an input domain.
        '''
        for pos in range(num_split-1):
            num_history = len(heads)
            for i in range(num_history):
                decision_layer = splits[pos*num_domain+domain_idx][0][0]
                decision_index = splits[pos*num_domain+domain_idx][0][1]

                for l in range(num_layer):
                    orig_ubs[l][num_history+i][domain_idx] = orig_ubs[l][i][domain_idx]
                    orig_lbs[l][num_history+i][domain_idx] = orig_lbs[l][i][domain_idx]

                orig_lbs[decision_layer][i][domain_idx].view(-1)[decision_index] = 0.0
                heads[i][decision_layer][0].append(decision_index)
                heads[i][decision_layer][1].append(1.0)
                heads.append(copy.deepcopy(heads[i]))
                orig_ubs[decision_layer][num_history+i][domain_idx].view(-1)[decision_index] = 0.0
                heads[-1][decision_layer][1][-1] = -1.0
        return heads
    new_history_list = []
    for i in range(num_domain):
        new_history_list.append(generate_history([history[i]], split, orig_lbs, orig_ubs, i))

    for i in range(len(new_history_list[0])):
        for j in range(num_domain):
            new_history.append(new_history_list[j][i])
    # num_copy * num_domain
    return new_history, orig_lbs, orig_ubs


def batch_verification(d, net, batch, pre_relu_indices, growth_rate, layer_set_bound=True,
                       dive_domains=None, adv_pool=None, stop_func=stop_criterion_sum, multi_spec_keep_func=lambda x: torch.all(x, dim=-1)):
    ### preprocessor-hint: private-section-start
    global diving_Visited
    ### preprocessor-hint: private-section-end
    global Visited, Flag_first_split
    global Use_optimized_split
    global DFS_enabled
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time
    
    opt_intermediate_beta = False
    ### preprocessor-hint: private-section-start
    opt_intermediate_beta = arguments.Config["solver"]["intermediate_refinement"]["enabled"]
    opt_coeffs = arguments.Config["solver"]["intermediate_refinement"]["opt_coeffs"]
    opt_bias = arguments.Config["solver"]["intermediate_refinement"]["opt_bias"]
    ### preprocessor-hint: private-section-end
    branching_method = arguments.Config['bab']['branching']['method']
    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    DFS_percent = arguments.Config["bab"]["dfs_percent"]
    branching_candidates = arguments.Config["bab"]["branching"]["candidates"]
    ### preprocessor-hint: private-section-start
    # dive_domains store the dive domains
    # dive_rate: 0-1, the portion of dive domains out of a batch if the batch is full
    max_dive_domains = arguments.Config["bab"]["attack"]["max_dive_domains"]
    num_dive_constraints = arguments.Config["bab"]["attack"]["num_dive_constraints"]
    dive_rate = arguments.Config["bab"]["attack"]["dive_rate"] if max_dive_domains >= 0 else 0.
    print(f"remaining dive domains: {len(dive_domains)}/{max_dive_domains}, dive_rate:{dive_rate}")
    ### preprocessor-hint: private-section-end
    decision_thresh = arguments.Config["bab"]["decision_thresh"]

    total_time = time.time()

    pickout_time = time.time()
    #### add new domains into dive_domains
    ### preprocessor-hint: private-section-start
    if len(dive_domains) <= max_dive_domains:
        # clone the original domains in domains into dive domains which have extra num_dive_constraints constraints
        num_takein_dive_domains = max_dive_domains - len(dive_domains)
        # Take the worse num_takein_dive_domains domains to the diving domain list.
        add_dive_domains_from_domains(d.pick_out(num_takein_dive_domains), dive_domains, num_dive_constraints=num_dive_constraints)
        print(f"taken {num_takein_dive_domains} new domains, new dive domains length: {len(dive_domains)}, domains length: {len(d)}")
    if len(dive_domains) > 0:
        print(f'current dive domain:', dive_domains[0].depth, dive_domains[0].lower_bound)
    #### Add the most promising domain from the dive domain to BaB domain for DFS research.
    if len(dive_domains) > 0 and dive_domains[0].depth > 127:  # FIXME: change the hard-coded number.
        print('Adding the worst domain for searching!', dive_domains[0].lower_bound, dive_domains[0].depth)
        dd = dive_domains.pop(0)
        d.add(dd)
        DFS_enabled = True
        searching_dive_domain = True
    else:
        searching_dive_domain = False
    if searching_dive_domain:
        print(f'The worst domain has lb {d[0].lower_bound} depth {d[0].depth}')
    ### preprocessor-hint: private-section-end

    domains_params = d.pick_out(batch=batch * (1 - dive_rate), device=net.x.device)
    mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs = domains_params
    # print('-' * 20)
    # print('mask shape:', [x.shape for x in mask])
    # print('lAs shape:', [x.shape for x in lAs])
    # print('orig_lbs shape:', [x.shape for x in orig_lbs])
    # print('orig_ubs shape:', [x.shape for x in orig_ubs])
    # print('slopes shape:', len(slopes), '*' dict([(x, dict([(xx, yy.shape) for xx, yy in y.items()])) for x, y in slope[0].items()]))
    # print('Cs.shape:', cs.shape)
    # print('-' * 20)

    pickout_time = time.time() - pickout_time
    total_pickout_time += pickout_time

    ### preprocessor-hint: private-section-start
    # when cplex cut is enabled, for domains with general_beta created for outdated cuts, 
    # we need to rewrite it to general_beta for new cuts
    if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
        cplex_update_general_beta(net, selected_domains)
    ### preprocessor-hint: private-section-end

    if mask is not None:
        decision_time = time.time()

        history = [sd.history for sd in selected_domains]
        split_history = [sd.split_history for sd in selected_domains]

        ''' Here we check the length of current domain list.
            If the domain list is small, we can split more layers.
        '''
        min_batch_size = min(arguments.Config["solver"]["beta-crown"]["min_batch_size_ratio"]*arguments.Config["solver"]["beta-crown"]["batch_size"], batch)

        if orig_lbs[0].shape[0] < min_batch_size:
            # Split multiple levels, to obtain at least min_batch_size domains in this batch.
            split_depth = int(np.log(min_batch_size)/np.log(2))

            if orig_lbs[0].shape[0] > 0:
                split_depth = max(int(np.log(min_batch_size/orig_lbs[0].shape[0])/np.log(2)), 0)
            split_depth = max(split_depth, 1)
        else:
            split_depth = 1

        print("batch: ", orig_lbs[0].shape, "pre split depth: ", split_depth)
        # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
        branching_candidates = max(branching_candidates, split_depth)

        if branching_method == 'babsr':
            branching_decision, split_depth = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                                            batch=batch, branching_reduceop=branching_reduceop, split_depth=split_depth, cs=cs, rhs=rhs)
        elif branching_method == 'fsb':
            branching_decision, split_depth = choose_node_parallel_FSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                            branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                            slopes=slopes, betas=betas, history=history, split_depth=split_depth, cs=cs, rhs=rhs)
        elif branching_method.startswith('kfsb'):
            branching_decision, split_depth = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                            branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                            slopes=slopes, betas=betas, history=history, split_depth=split_depth, cs=cs, rhs=rhs,
                                            method=branching_method)
        else:
            raise NotImplementedError

        print("batch: ", orig_lbs[0].shape, "post split depth: ", split_depth)

        if len(branching_decision) < len(mask[0]):
            print('all nodes are split!!')
            print('{} domains visited'.format(Visited))
            global all_node_split
            all_node_split = True
            if not arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
                global_lb = selected_domains[0].lower_bound - selected_domains[0].threshold
                for i in range(1, len(selected_domains)):
                    if max(selected_domains[i].lower_bound - selected_domains[i].threshold) <= max(global_lb):
                        global_lb = selected_domains[i].lower_bound - selected_domains[i].threshold
                return global_lb, np.inf

        ### preprocessor-hint: private-section-start
        selected_dive_domains = []
        if len(dive_domains) > 0 and dive_rate > 0:
            # pick out dive domains and append it to the selected domains
            # extra constraints have already been added when a dive domains is added, so just pick out!
            dive_domains_params = dive_domains.pick_out(batch=round(batch * dive_rate), device=net.x.device)
            selected_dive_domains = dive_domains_params[-1]
            mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs = merge_domains_params(domains_params, dive_domains_params)
            # need to add the new split constraints for dive domains in history
            history = [sd.history for sd in selected_domains]
            split_history = [sd.split_history for sd in selected_domains]
        ### preprocessor-hint: private-section-end

        print('splitting decisions: ')
        for l in range(split_depth):
            print("split level {}".format(l), end=": ")
            for b in range(min(10, len(history))):
                print(branching_decision[l*len(history) + b], end=" ")
            print('')
        # print the first two split for first 10 domains.

        if not Use_optimized_split:
            split = {}
            # split["decision"]: selected domains (next batch/2)->node list->node: [layer, idx]
            split["decision"] = [[bd] for bd in branching_decision]
            # split["split"]: selected domains (next batch/2)->node list->float coefficients
            split["coeffs"] = [[1.] for i in range(len(branching_decision))]
        else:
            split = {}
            num_nodes = 3
            split["decision"] = [[[2, i] for i in range(num_nodes)] for bd in branching_decision]
            split["coeffs"] = [[random.random() * 0.001 - 0.0005 for j in range(num_nodes)] for i in
                               range(len(branching_decision))]
        ### preprocessor-hint: private-section-start
        # diving stores how many diving domains in this batch
        split["diving"] = len(selected_dive_domains) if selected_dive_domains else 0
        ### preprocessor-hint: private-section-end

        decision_time = time.time() - decision_time
        total_decision_time += decision_time

        solve_time = time.time()
        single_node_split = True
        ### preprocessor-hint: private-section-start
        single_node_split = not opt_coeffs and not opt_bias and not opt_intermediate_beta
        ### preprocessor-hint: private-section-end
        # copy the original lbs

        num_copy = (2**(split_depth-1))

        if num_copy > 1:
            orig_lbs = [lb.unsqueeze(0).repeat(num_copy, *[1]*len(lb.shape)) for lb in orig_lbs]
            orig_ubs = [ub.unsqueeze(0).repeat(num_copy, *[1]*len(ub.shape)) for ub in orig_ubs]
            # 4 * [num_copy, num_domain, xxx]

            num_domain = len(history)

            # create fake history for each branch.
            # TODO: set origlbs and orig_ubs
            history, orig_lbs, orig_ubs = build_history(history, split['decision'], orig_lbs, orig_ubs)
            # num_domains -> num_domains * (2**num_split_layer)

            # set the slopes for each branch
            for k, v in slopes.items():
                for kk, vv in v.items():
                    v[kk] = torch.cat([vv] * num_copy, dim=2)

            # create fake split_history for each branch.
            split_history = split_history * num_copy

            # cs needs to repeat
            cs = torch.cat([cs] * num_copy, dim=0)

            new_betas = []
            new_intermediate_betas = []
            for i in range(num_copy):
                for j in range(len(betas)):
                    new_betas.append(betas[j])
                    new_intermediate_betas.append(intermediate_betas[j])
            betas = new_betas
            intermediate_betas = new_intermediate_betas

            orig_lbs = [lb.view(-1, *lb.shape[2:]) for lb in orig_lbs]
            orig_ubs = [ub.view(-1, *ub.shape[2:]) for ub in orig_ubs]

            # create split for num_copy * num_domain
            # we only keep the last split since the first few ones has been split with build_history
            split['decision'] = split['decision'][-num_domain:] * num_copy
            split['coeffs'] = split['coeffs'][-num_domain:] * num_copy

            branching_decision = branching_decision[-num_domain:] * num_copy
            rhs = torch.cat([rhs] * num_copy, dim=0)

        # Caution: we use "all" predicate to keep the domain when multiple specs are present: all lbs should be <= threshold, otherwise pruned
        # maybe other "keeping" criterion needs to be passed here
        ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history,
                                  split_history=split_history, layer_set_bound=layer_set_bound, betas=betas,
                                  single_node_split=single_node_split, intermediate_betas=intermediate_betas, cs=cs, decision_thresh=rhs, rhs=rhs,
                                  stop_func=stop_func(torch.cat([rhs, rhs])), multi_spec_keep_func=multi_spec_keep_func)

        dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals, dom_cs = ret

        if adv_pool is not None:
            adv_pool.add_adv_images(primals)
            adv_pool.print_pool_status()

        solve_time = time.time() - solve_time
        total_solve_time += solve_time
        add_time = time.time()
        batch, diving_batch = len(branching_decision), split["diving"]
        # If intermediate layers are not refined or updated, we do not need to check infeasibility when adding new domains.
        check_infeasibility = not (single_node_split and layer_set_bound)

        depths = [domain.depth + split_depth - 1 for domain in selected_domains] * num_copy * 2

        old_d_len = len(d)
        if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
            for domain_idx in range(len(depths)):
                # get tot_ambi_nodes
                dlb, dub = [dlbs[domain_idx: domain_idx + 1] for dlbs in dom_lb_all],  [dubs[domain_idx: domain_idx + 1] for dubs in dom_ub_all]
                decision_threshold = rhs.to(dom_lb[0].device, non_blocking=True)[domain_idx if domain_idx < (len(dom_lb)//2) else domain_idx - (len(dom_lb)//2)]
                # print(depths[domain_idx] + 1, dlb[-1], decision_threshold, torch.all(dlb[-1] <= decision_threshold))
                if depths[domain_idx] + 1 == net.tot_ambi_nodes  and torch.all(dlb[-1] <= decision_threshold):
                    lp_status, dlb, adv = net.all_node_split_LP(dlb, dub, decision_threshold)
                    print(f"using lp to solve all split node domain {domain_idx}/{len(dom_lb)}, results {dom_lb[domain_idx]} -> {dlb}, {lp_status}")
                    # import pdb; pdb.set_trace()
                    if lp_status == "unsafe":
                        # unsafe cases still needed to be handled! set to be unknown for now!
                        all_node_split = True
                        return dlb, np.inf
                    dom_lb_all[-1][domain_idx] = dlb
                    dom_lb[domain_idx] = dlb

        d.add(lAs, dom_lb, dom_ub, dom_lb_all, dom_ub_all, history, depths, slopes, betas, split_history,
              branching_decision, rhs, intermediate_betas, check_infeasibility, dom_cs, (2*num_copy)*batch)
        # unsat_list = add_domain_parallel(lA=lAs[:(2*num_copy)*batch], lb=dom_lb[:(2*num_copy)*batch], ub=dom_ub[:(2*num_copy)*batch], lb_all=dom_lb_all[:(2*num_copy)*batch], up_all=dom_ub_all[:(2*num_copy)*batch],
        #                                  domains=d, history=history, depth=depths, slope=slopes[:(2*num_copy)*batch], beta=betas[:(2*num_copy)*batch],
        #                                  growth_rate=growth_rate, branching_decision=branching_decision, decision_thresh=rhs,
        #                                  split_history=split_history[:(2*num_copy)*batch], intermediate_betas=intermediate_betas[:(2*num_copy)*batch],
        #                                  check_infeasibility=check_infeasibility, primals=primals[:(2*num_copy)*batch] if primals is not None else None,
        #                                  cs=dom_cs[:(2*num_copy)*batch])
        ### preprocessor-hint: private-section-start
        if searching_dive_domain:
            print(f'The worst domain has lb {dom_lb[0]} {dom_lb[batch]} after split')
        if diving_batch > 0:
            # FIXME (09/28): We should not furter dive inside this add_dive_domain_parallel function. This is not aligned with this function name and what it is supposed to do.
            # add extra constraints to the tested dive domains and put it back to dive_domains
            diving_unsat_list = add_dive_domain_parallel(lA=lAs[2*batch:], lb=dom_lb[2*batch:], ub=dom_ub[2*batch:],
                                            lb_all=dom_lb_all[2*batch:], ub_all=dom_ub_all[2*batch:],
                                            dive_domains=dive_domains, selected_domains=selected_domains[batch:],
                                            slope=slopes[2*batch:], beta=betas[2*batch:], decision_thresh=rhs,
                                            split_history=split_history[2*batch:], intermediate_betas=intermediate_betas[2*batch:],
                                            check_infeasibility=check_infeasibility, num_dive_constraints=num_dive_constraints, primals=primals[2*batch:] if primals is not None else None,
                                            cs=dom_cs[2*batch:])
            diving_Visited += (diving_batch - len(diving_unsat_list))  # one unstable neuron split to two nodes
            print('Diving domain [lb, ub] (depth):')
            for i in dive_domains[:50]:
                print(f'[{i.lower_bound:.5f}, {i.upper_bound:.5f}] ({i.depth})', end=', ')
            print()
            dive_domains.clear()
        ### preprocessor-hint: private-section-end

        Visited += (len(selected_domains) * num_copy) * 2 - (len(d) - old_d_len)
        if len(d) > 0:
            if get_upper_bound:
                print('Current worst splitting domains [lb, ub] (depth):')
            else:
                print('Current worst splitting domains lb-rhs (depth):')
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
                printed_d = d.get_min_domain(20, rev_order=True)
            else:
                printed_d = d.get_min_domain(20)
            for i in printed_d:
                if get_upper_bound:
                    print(f'[{(i.lower_bound - i.threshold).max():.5f}, {(i.upper_bound - i.threshold).min():5f}] ({i.depth})', end=', ')
                else:
                    print(f'{(i.lower_bound - i.threshold).max():.5f} ({i.depth})', end=', ')
            print()
            if hasattr(d, 'sublist'):
                print(f'Max depth domain: [{d.sublist[0].domain.lower_bound}, {d.sublist[0].domain.upper_bound}] ({d.sublist[0].domain.depth})')
        add_time = time.time() - add_time
        total_add_time += add_time

        total_time = time.time() - total_time
        print('length of domains:', len(d))
        print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')
        print(f'Accumulated time:\t pickout: {total_pickout_time:.4f}\t decision: {total_decision_time:.4f}\t get_bound: {total_solve_time:.4f}\t add_domain: {total_add_time:.4f}')

    if len(d) > 0:
        if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
            worst_domain = d.get_min_domain(1 ,rev_order=True)
            global_lb = worst_domain[-1].lower_bound - worst_domain[-1].threshold
        else:
            worst_domain = d.get_min_domain(1 ,rev_order=False)
            global_lb = worst_domain[0].lower_bound - worst_domain[0].threshold
    else:
        print("No domains left, verification finished!")
        print('{} domains visited'.format(Visited))
        return torch.tensor(arguments.Config["bab"]["decision_thresh"] + 1e-7), np.inf

    batch_ub = np.inf
    if get_upper_bound:
        batch_ub = min(dom_ub)
        print(f"Current (lb-rhs): {global_lb.max()}, ub:{batch_ub}")
    else:
        print(f"Current (lb-rhs): {global_lb.max()}")

    print('{} domains visited'.format(Visited))
    ### preprocessor-hint: private-section-start
    if dive_rate > 0:
        print('{} diving domains visited'.format(diving_Visited))
    ### preprocessor-hint: private-section-end

    return global_lb, batch_ub


### preprocessor-hint: private-section-start
def cut_verification(d, net, pre_relu_indices, layer_set_bound=True):
    decision_thresh = arguments.Config["bab"]["decision_thresh"]
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    cplex_cuts = arguments.Config["bab"]["cut"]["cplex_cuts"]

    # construct the cut splits
    # change to only create one domain and make sure the other is infeasible
    split = {}
    if cplex_cuts:
        generate_cplex_cuts(net)

    if net.cutter.cuts is not None:
        split["cut"] = net.cutter.cuts
        split["cut_timestamp"] = net.cutter.cut_timestamp
    else:
        print('Cut is not present from cplex or predefined cut yet, direct return from cut init')
        return None, None
    return None, None

    # total_time = time.time()
    # pickout_time = time.time()
    #
    # assert len(d) == 1, "only apply cut to the first initial domain for now"
    #
    # domains_params = pick_out_batch(d, batch=1, device=net.x.device)
    # # mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs = domains_params
    # mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs = domains_params
    #
    # history = [sd.history for sd in selected_domains]
    # split_history = [sd.split_history for sd in selected_domains]
    #
    # pickout_time = time.time() - pickout_time
    #
    # # gurobi lp solver cut
    # if lp_cut_enabled:
    #     gurobi_time = time.time()
    #     glb = net.update_the_model_cut( net.cutter.cuts, orig_lbs, orig_ubs)
    #     gurobi_time = time.time() - gurobi_time
    #
    # solve_time = time.time()
    # if getattr(net.cutter, 'opt', False):
    #     print('get_lower_bound with cut optimization!')
    # else:
    #     print('get_lower_bound without cut optimization!')
    # ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history,
    #                         split_history=split_history, layer_set_bound=layer_set_bound, betas=betas,
    #                         intermediate_betas=intermediate_betas)
    # dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, cut_slopes, split_history, betas, intermediate_betas, primals, _ = ret
    #
    # solve_time = time.time() - solve_time
    # add_cut_domain(
    #     lA=lAs, lb=dom_lb, ub=dom_ub, lb_all=dom_lb_all, ub_all=dom_ub_all,
    #     domains=d, selected_domains=selected_domains, slope=cut_slopes, beta=betas,
    #     decision_thresh=decision_thresh, split_history=split_history,
    #     intermediate_betas=intermediate_betas,
    #     check_infeasibility=True,
    #     primals=primals if primals is not None else None,
    #     cs=cs)
    # total_time = time.time() - total_time
    # print(f'Total time: {total_time:.4f}')
    #
    # if lp_cut_enabled:
    #     glb = net.update_the_model_cut(net.cutter.cuts, dom_lb_all[0], dom_ub_all[0])
    #
    # if len(d) > 0:
    #     if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
    #         global_lb = d[-1].lower_bound
    #     else:
    #         global_lb = d[0].lower_bound
    # else:
    #     print("No domains left, verification finished!")
    #     return torch.tensor(arguments.Config["bab"]["decision_thresh"] + 1e-7), np.inf
    #
    # batch_ub = np.inf
    # if get_upper_bound:
    #     batch_ub = min(dom_ub)
    #     print(f"Current lb:{global_lb}, ub:{batch_ub}")
    # else:
    #     print(f"Current lb:{global_lb}")
    # return global_lb, batch_ub


def relu_bab_parallel(net, domain, x, use_neuron_set_strategy=False, refined_lower_bounds=None,
                      refined_upper_bounds=None, activation_opt_params=None,
                      reference_slopes=None, reference_lA=None, attack_images=None,
                      targets=None, timeout=None, refined_betas=None, rhs=0):
    # the crown_lower/upper_bounds are present for initializing the unstable indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN process again which is slightly slower
    start = time.time()
    # All supported arguments.
    global Visited, Flag_first_split, all_node_split, DFS_enabled
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0

    opt_intermediate_beta = False
    timeout = timeout or arguments.Config["bab"]["timeout"]
    max_domains = arguments.Config["bab"]["max_domains"]
    batch = arguments.Config["solver"]["beta-crown"]["batch_size"]
    # TODO: arguments.Config["bab"]["decision_thresh"] sometimes has conflict rhs parsed by vnnlib
    decision_thresh = arguments.Config["bab"]["decision_thresh"] if (arguments.Config["bab"]["decision_thresh"] == rhs).all() else rhs
    record = arguments.Config["general"]["record_bounds"]
    ### preprocessor-hint: private-section-start
    opt_intermediate_beta = arguments.Config["solver"]["intermediate_refinement"]["enabled"]
    intermediate_refinement_layers = arguments.Config["solver"]["intermediate_refinement"]["layers"]
    ### preprocessor-hint: private-section-end
    lp_test = arguments.Config["debug"]["lp_test"]
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    DFS_percent = arguments.Config["bab"]["dfs_percent"]
    ### preprocessor-hint: private-section-start
    max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
    min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
    ### preprocessor-hint: private-section-end
    ### preprocessor-hint: private-section-start
    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    ### preprocessor-hint: private-section-end
    use_batched_domain = arguments.Config["bab"]["batched_domain_list"]

    multi_targets = targets if targets is not None and targets.size > 1 else None
    attack_args = getattr(arguments, 'attack_args', None)

    if isinstance(decision_thresh, torch.Tensor) and decision_thresh.shape[-1] > 1:
        # multi-bounds output for one C matrix
        # any spec >= rhs, then this sample can be stopped; if all samples can be stopped, stop = True, o.w., False
        stop_criterion = stop_criterion_batch_any
        multi_spec_keep_func = lambda x: torch.all(x, dim=-1)
    else:
        decision_thresh = torch.tensor(decision_thresh, dtype=torch.get_default_dtype(), device=x.device)
        assert len(decision_thresh.unique()) == 1
        decision_thresh = decision_thresh.unique().view(1, 1)
        stop_criterion = stop_criterion_sum
        # will not call multi_spec_keep_func
        multi_spec_keep_func = None

    # if arguments.Config["data"]["dataset"] == 'Carvana':
    #     stop_criterion = stop_criterion_batch_topk

    Visited, Flag_first_split, global_ub = 0, True, np.inf
    DFS_enabled = False
    adv_pool = None
    betas = None
    if arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, None, None, stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=None,
            targets=targets, cutter=net.cutter)
    elif refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config["general"]["enable_incomplete_verification"] is False
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_image = net.build_the_model(
            domain, x, stop_criterion_func=stop_criterion(decision_thresh), multi_targets=multi_targets)
    else:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds, activation_opt_params, reference_lA=reference_lA,
            stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=reference_slopes,
            targets=targets, cutter=net.cutter, refined_betas=refined_betas)
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()

    if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
        timeout = arguments.Config["bab"]["timeout"]
        # mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
        # mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
        # solver_pkg = arguments.Config["solver"]["intermediate_refinement"]["solver_pkg"]
        # adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
        net.build_solver_model(timeout, model_type="lp")

    ### preprocessor-hint: private-section-start
    if arguments.Config["bab"]["attack"]["enabled"]:
        # Beam dive enabled. We need to construct the MIP model.
        print('Building MIP for beam search...')
        out_vars = net.build_solver_model(
                    timeout=arguments.Config["bab"]["attack"]["mip_timeout"],
                    mip_multi_proc=arguments.Config["solver"]["mip"]["parallel_solvers"],
                    mip_threads=arguments.Config["solver"]["mip"]["solver_threads"],
                    model_type="mip")
        # setting model_type to lp, just to check solution!
        # import gurobipy as grb
        # for obj in out_vars:
        #     guro_start = time.time()
        #     net.net.model.setObjective(obj, grb.GRB.MINIMIZE)
        #     try:
        #         net.net.model.optimize()
        #     except grb.GurobiError as e:
        #         handle_gurobi_error(e.message)

        #     status = net.net.model.status
        #     assert status == 2, f"LP wasn't optimally solved status:{status}"
        #     print(f"[{obj}]- status: {status}, time: {time.time() - guro_start}")
        #     glb = obj.X if status != 3 else None
        # exit()
    ### preprocessor-hint: private-section-end

    # print('*' * 20)
    # print('c.shape:', net.c.shape)
    # print('global_ub:', global_ub)
    # print('global_lb:', global_lb)
    # print('primals:', primals)
    # print('updated_mask shape:', [x.shape for x in updated_mask])
    # print('lA shape:', [x.shape for x in lA[0]])
    # print('lower_bounds shape:', [x.shape for x in lower_bounds])
    # print('upper_bounds shape:', [x.shape for x in upper_bounds])
    # print('pre_relu_indices:', pre_relu_indices)
    # print('slope shape:', [dict([(x, dict([(xx, yy.shape) for xx, yy in y.items()])) for x, y in slopeitem.items()]) for slopeitem in slope])
    # print('history:', history)
    # print('*' * 20)

    all_label_global_lb = global_lb
    # if isinstance(all_label_global_lb, torch.Tensor) and all_label_global_lb.numel() > 1:
    all_label_global_lb = torch.min(all_label_global_lb - decision_thresh).item()
    all_label_global_ub = global_ub
    # if isinstance(all_label_global_ub, torch.Tensor) and all_label_global_ub.numel() > 1:
    all_label_global_ub = torch.max(all_label_global_ub - decision_thresh).item()

    if lp_test in ["LP", "MIP"]:
        return all_label_global_lb, all_label_global_ub, [[time.time()-start, global_lb]], 0, 'unknown'

    if stop_criterion(decision_thresh)(global_lb).all():
        return all_label_global_lb, all_label_global_ub, [[time.time()-start, global_lb]], 0, 'safe'

    ### preprocessor-hint: private-replacement \tif True:
    ### preprocessor-hint: private-section-start
    if not opt_intermediate_beta:
    ### preprocessor-hint: private-section-end
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        if not arguments.Config['solver']['beta-crown'].get('enable_opt_interm_bounds', False):
            # new_slope shape: [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}] for each sample in batch]
            new_slope = {}
            kept_layer_names = [net.net.final_name]
            kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
            print(f'Keeping slopes for these layers: {kept_layer_names}')
            for relu_layer, alphas in slope.items():
                new_slope[relu_layer] = {}
                for layer_name in kept_layer_names:
                    if layer_name in alphas:
                        new_slope[relu_layer][layer_name] = alphas[layer_name]
                    else:
                        print(f'Layer {relu_layer} missing slope for start node {layer_name}')
        else:
            new_slope = slope
    ### preprocessor-hint: private-section-start
    else:
        # We keep only these alphas for layers that will be optimized.
        # First get the layer names for refinement.
        refinement_layer_names = [net.net.relus[il].input_name[0] for il in intermediate_refinement_layers]
        # new_slope shape: [dict[relu_layer_name, dict[dest_layer_name, torch.tensor storing alpha]] for each sample in batch]
        new_slope = {}
        # Output layer alpha is always included.
        refinement_layer_names.append(net.net.final_name)
        for relu_layer, alphas in slope.items():
            new_slope[relu_layer] = {}
            for spec_name, v in alphas.items():
                if spec_name in refinement_layer_names:
                    new_slope[relu_layer][spec_name] = v
                else:
                    print(f'Removing slopes for layer {spec_name} as it is not being refined.')
    ### preprocessor-hint: private-section-end

    if use_batched_domain:
        DomainClass = BatchedReLUDomainList
    else:
        DomainClass = SortedReLUDomainList

    # This is the first (initial) domain.
    if multi_targets is None:
        domains = DomainClass([item[0:1] for item in lA], global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                              copy.deepcopy(history), [0], net.c, decision_thresh, betas, 1,
                              interm_transfer=arguments.Config["bab"]["interm_transfer"])

        if arguments.Config["bab"]["attack"]["enabled"]:
            dive_domains = DomainClass([item[0:1] for item in lA], global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                                       copy.deepcopy(history), [0], net.c, decision_thresh, betas, 1,
                                       interm_transfer=arguments.Config["bab"]["interm_transfer"])
        else:
            dive_domains = []
    else:
        domains = DomainClass(lA, global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                              copy.deepcopy(history), [0 for _ in multi_targets], net.c,
                              decision_thresh,
                              betas, len(multi_targets),
                              interm_transfer=arguments.Config["bab"]["interm_transfer"])

        if arguments.Config["bab"]["attack"]["enabled"]:
            dive_domains = DomainClass(lA, global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                                       copy.deepcopy(history), [0 for _ in multi_targets], net.c,
                                       decision_thresh,
                                       betas, len(multi_targets),
                                       interm_transfer=arguments.Config["bab"]["interm_transfer"])
        else:
            dive_domains = []
    # domains = DFS_SortedList() if DFS_percent > 0 else SortedList()
    # dive_domains = SortedList()
    # for candidate_domain in candidate_domains:
    #     domains.add(candidate_domain)
    ## preprocessor-hint: private-section-start
    if arguments.Config["bab"]["attack"]["adv_dive"]:
        for candidate_domain_idx in len(domains):
            add_dive_domain_from_adv(dive_domains, domains[candidate_domain_idx], attack_images, net, updated_mask)
    ## preprocessor-hint: private-section-end

    if not arguments.Config["bab"]["interm_transfer"]:
        # tell the AutoLiRPA class not to transfer intermediate bounds to save time
        net.interm_transfer = arguments.Config["bab"]["interm_transfer"]

    # after domains are added, we replace global_lb, global_ub with the multile targets "real" global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub

    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = [mask[0:1] for mask in updated_mask]
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable

    print(f'-----------------\n# of unstable neurons: {tot_ambi_nodes}\n-----------------\n')
    net.tot_ambi_nodes = tot_ambi_nodes

    ### preprocessor-hint: private-section-start
    if cut_enabled:
        print('======================Cut verification begins======================')
        start_cut = time.time()
        # enable lp solver
        if lp_cut_enabled:
            glb = net.build_the_model_lp()
        if arguments.Config["bab"]["cut"]["cplex_cuts"]:
            time.sleep(arguments.Config["bab"]["cut"]["cplex_cuts_wait"])
        global_lb_from_cut, batch_ub_from_cut = cut_verification(domains, net, pre_relu_indices, layer_set_bound=not opt_intermediate_beta)
        if global_lb_from_cut is None and batch_ub_from_cut is None:
            # no available cut present --- we don't refresh global_lb and global_ub
            pass
        else:
            global_lb, batch_ub = global_lb_from_cut, batch_ub_from_cut
        print('Cut bounds before BaB:', float(global_lb))
        if len(domains) >= 1 and getattr(net.cutter, 'opt', False):
            # beta will be reused from split_history
            assert len(domains) == 1
            assert isinstance(domains[0].split_history['general_betas'], torch.Tensor)
            net.cutter.refine_cuts(split_history=domains[0].split_history)
        print('Cut time:', time.time() - start_cut)
        print('======================Cut verification ends======================')        
    ### preprocessor-hint: private-section-end

    ### preprocessor-hint: private-section-start
    # Max number of fixed neurons during diving.
    max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
    min_local_free = int(min_local_free_ratio * tot_ambi_nodes)

    if arguments.Config["bab"]["attack"]["enabled"]:
        adv_pool = AdvExamplePool(net.net, updated_mask, C=net.c)
        adv_pool.add_adv_images(attack_images)
        print(f'best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')
        adv_pool.print_pool_status()
        find_promising_domains.counter = 0
        # find_promising_domains.current_method = "bottom-up"
        find_promising_domains.current_method = "top-down"
        find_promising_domains.topdown_status = "normal"
        find_promising_domains.bottomup_status = "normal"
        beam_mip_attack.started = False
        global_ub = min(all_label_global_ub, adv_pool.adv_pool[0].obj)
    ### preprocessor-hint: private-section-end

    glb_record = [[time.time()-start, global_lb]]
    stop_condition = len(domains) > 0
    ### preprocessor-hint: private-section-start
    stop_condition = (len(domains) > 0 if not arguments.Config["bab"]["attack"]["enabled"]
                                    else len(dive_domains) > 0)
    ### preprocessor-hint: private-section-end

    while stop_condition:
        ### preprocessor-hint: private-section-start
        if opt_intermediate_beta:
            if len(domains) > arguments.Config["solver"]["intermediate_refinement"]["max_domains"]:
                # reach refinement max domains, disable refinement and set back to regular batch size
                arguments.Config["solver"]["intermediate_refinement"]["layers"] = []
                batch = arguments.Config["solver"]["beta-crown"]["batch_size"]
            else:
                batch = arguments.Config["solver"]["intermediate_refinement"]["batch_size"]
        ### preprocessor-hint: private-section-end

        ### preprocessor-hint: private-section-start
        max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
        min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
        max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
        min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
        if arguments.Config["bab"]["attack"]["enabled"]:
            global_lb, batch_ub, dive_domains = bab_attack(dive_domains.to_sorted_list(), net, batch, pre_relu_indices, 0,
                                            layer_set_bound=not opt_intermediate_beta,
                                            adv_pool=adv_pool, attack_args=attack_args,
                                            max_dive_fix=max_dive_fix, min_local_free=min_local_free)
        else:
            # cut is enabled
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                fetch_cut_from_cplex(net)
            ### preprocessor-hint: private-section-end
            ### preprocessor-hint: private-replacement \t\tif True:
            if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:  # do two batch of neuron set bounds  per 10000 domains
                # neuron set  bounds cost more memory, we set a smaller batch here
                global_lb, batch_ub = batch_verification(domains, net, int(batch/2), pre_relu_indices, 0, layer_set_bound=False,
                                            dive_domains=dive_domains, adv_pool=adv_pool,
                                            stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func)
            else:
                global_lb, batch_ub = batch_verification(domains, net, batch, pre_relu_indices, 0,
                                            layer_set_bound=not opt_intermediate_beta,
                                            dive_domains=dive_domains, adv_pool=adv_pool,
                                            stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func)
        if get_upper_bound:
            print(f"Global ub: {global_ub}, batch ub: {batch_ub}")
        global_ub = min(global_ub, batch_ub)
        stop_condition = len(domains) > 0
        ### preprocessor-hint: private-section-start
        stop_condition = len(domains) > 0 if not arguments.Config["bab"]["attack"]["enabled"] else len(dive_domains) > 0
        ### preprocessor-hint: private-section-end

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        if isinstance(global_ub, torch.Tensor):
            global_ub = global_ub.min().item()

        if all_node_split:
            del domains
            all_node_split = False
            clean_net_mps_process(net)
            return global_lb, global_ub, glb_record, Visited, 'unknown'

        if len(domains) > max_domains:
            print("No enough memory for the domain list!!!!!!!!")
            del domains
            clean_net_mps_process(net)
            return global_lb, global_ub, glb_record, Visited, 'unknown'

        if get_upper_bound:
            if global_ub < decision_thresh:
                print("Attack success during bab!!!!!!!!")
                # Terminate MIP if it has been started.
                ### preprocessor-hint: private-section-start
                if arguments.Config["bab"]["attack"]["enabled"] and beam_mip_attack.started:
                    print('Terminating MIP processes...')
                    net.pool_termination_flag.value = 1
                ### preprocessor-hint: private-section-end
                del domains
                clean_net_mps_process(net)
                return global_lb, global_ub, glb_record, Visited, 'unsafe'

        if time.time() - start > timeout:
            print('Time out!!!!!!!!')
            ### preprocessor-hint: private-section-start
            if arguments.Config["bab"]["attack"]["enabled"] and beam_mip_attack.started:
                print('Terminating MIP processes...')
                net.pool_termination_flag.value = 1
            ### preprocessor-hint: private-section-end
            del domains
            clean_net_mps_process(net)
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb, global_ub, glb_record, Visited, 'unknown'

        if record:
            glb_record.append([time.time() - start, global_lb])
        print(f'Cumulative time: {time.time() - start}\n')

    del domains
    clean_net_mps_process(net)
    return global_lb, global_ub, glb_record, Visited, 'safe'
