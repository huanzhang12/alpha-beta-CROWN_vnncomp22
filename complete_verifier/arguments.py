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
"""
Arguments parser and config file loader.

When adding new commandline parameters, please make sure to provide a clear and descriptive help message and put it in under a related hierarchy.
"""

import re
import os
from secrets import choice
import sys
import yaml
import time
import argparse
from collections import defaultdict


class ConfigHandler:

    def __init__(self):
        self.config_file_hierarchies = {
                # Given a hierarchy for each commandline option. This hierarchy is used in yaml config.
                # For example: "batch_size": ["solver", "propagation", "batch_size"] will be an element in this dictionary.
                # The entries will be created in add_argument() method.
        }
        # Stores all arguments according to their hierarchy.
        self.all_args = {}
        # Parses all arguments with their defaults.
        self.defaults_parser = argparse.ArgumentParser()
        # Parses the specified arguments only. Not specified arguments will be ignored.
        self.no_defaults_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        # Help message for each configuration entry.
        self.help_messages = defaultdict(str)
        # Add all common arguments.
        self.add_common_options()

    def add_common_options(self):
        """
        Add all parameters that are shared by different front-ends.
        """

        # We must set how each parameter will be presented in the config file, via the "hierarchy" parameter.
        # Global Configurations, not specific for a particular algorithm.

        # The "--config" option does not exist in our parameter dictionary.
        self.add_argument('--config', type=str, help='Path to YAML format config file.', hierarchy=None)

        h = ["general"]
        self.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='Select device to run verifier, cpu or cuda (GPU).', hierarchy=h + ["device"])
        self.add_argument("--seed", type=int, default=100, help='Random seed.', hierarchy=h + ["seed"])
        self.add_argument("--conv_mode", default="patches", choices=["patches", "matrix"],
                help='Convolution mode during bound propagation: "patches" mode (default) is very efficient, but may not support all architecture; "matrix" mode is slow but supports all architectures.', hierarchy=h + ["conv_mode"])
        self.add_argument("--deterministic", action='store_true', help='Run code in CUDA deterministic mode, which has slower performance but better reproducibility.', hierarchy=h + ["deterministic"])
        self.add_argument("--double_fp", action='store_true',
                help='Use double precision floating point. GPUs with good double precision support are preferable (NVIDIA P100, V100, A100; AMD Radeon Instinc MI50, MI100).', hierarchy=h + ["double_fp"])
        self.add_argument("--loss_reduction_func", default="sum", help='When batch size is not 1, this reduction function is applied to reduce the bounds into a single number (options are "sum" and "min").', hierarchy=h + ["loss_reduction_func"])
        self.add_argument("--record_lb", action='store_true', help='Save lower bound during branch and bound. For debugging only.', hierarchy=h + ["record_bounds"])
        self.add_argument("--no_sparse_alpha", action='store_false', help='Sparse alpha (with bugs).', hierarchy=h + ["sparse_alpha"])
        self.add_argument("--save_adv_example", action='store_true', help='Save returned adversarial example in file.', hierarchy=h + ["save_adv_example"])

        h = ["model"]
        self.add_argument("--load", type=str, default=None, help='Load pretrained model from this specified path.', hierarchy=h + ["path"])
        self.add_argument("--cache_onnx_conversion", action='store_true', help='Cache the model converted from ONNX.', hierarchy=h + ["cache_onnx_conversion"])
        self.add_argument('--onnx_quirks', type=str, default=None, help="Load onnx model with quirks to workaround onnx model issue. "
                "This string will be passed to onnx2pytorch as the 'quirks' argument, and it is typically a literal of a python dict, e.g., \"{'Reshape': {'fix_batch_size: True'}}\".", hierarchy=h + ["onnx_quirks"])

        h = ["data"]
        self.add_argument("--start", type=int, default=0, help='Start from the i-th property in specified dataset.', hierarchy=h + ["start"])
        self.add_argument("--end", type=int, default=10000, help='End with the (i-1)-th property in the dataset.', hierarchy=h + ["end"])
        self.add_argument("--select_instance", type=int, nargs='+', default=None, help='Select a list of instances to verify.', hierarchy=h + ["select_instance"])
        self.add_argument('--num_outputs', type=int, default=10, help="Number of classes for classification problem.", hierarchy=h + ["num_outputs"])
        self.add_argument("--mean", nargs='+', type=float, default=0.0, help='Mean vector used in data preprocessing.', hierarchy=h + ["mean"])
        self.add_argument("--std", nargs='+', type=float, default=1.0, help='Std vector used in data preprocessing.', hierarchy=h + ["std"])
        self.add_argument('--pkl_path', type=str, default=None, help="Load properties to verify from a .pkl file (only used for oval20 dataset).", hierarchy=h + ["pkl_path"])

        h = ["specification"]
        self.add_argument("--spec_type", type=str, default='lp', choices=['lp', 'bound'], help='Type of verification specification. "lp" = L_p norm, "bounds" = element-wise lower and upper bound provided by dataloader.', hierarchy=h + ["type"])
        self.add_argument("--norm", type=float, default='inf', help='Lp-norm for epsilon perturbation in robustness verification (1, 2, inf).', hierarchy=h + ["norm"])
        self.add_argument("--epsilon", type=float, default=None, help='Set perturbation size (Lp norm). If not set, a default value may be used based on dataset loader.', hierarchy=h + ["epsilon"])

        h = ["solver"]
        self.add_argument("--float64_last_iter", default=True, action='store_false', help='Use double fp (float64) at the last iteration in alpha/beta CROWN.', hierarchy=h + ["no_float64_last_iter"])
        self.add_argument("--no_amp", action='store_true', help='Do not use amp (mix precision) in alpha/beta CROWN iterations.', hierarchy=h + ["no_amp"])
        self.add_argument('--early_stop_patience', type=int, default=10, help='Number of iterations that we will early stop if tracking no improvement.', hierarchy=h + ["early_stop_patience"])
        self.add_argument('--start_save_best', type=int, default=2, help='Start to save optimized best bounds when [i > iteration/start_save_best].', hierarchy=h + ["start_save_best"])
        self.add_argument('--bound_prop_method', default="alpha-crown", 
                        choices=["alpha-crown", "crown", "forward", "forward+crown", "alpha-forward", "crown-ibp", "init-crown"],
                        help='Bound propagation method used for incomplete verification and input split based branch and bound.', hierarchy=h + ["bound_prop_method"])        
        self.add_argument("--prune_after_crown", action='store_true', help='After CROWN pass, prune verified labels before starting the alpha-CROWN pass.', hierarchy=h + ["prune_after_crown"])

        h = ["solver", "crown"]
        self.add_argument('--crown_batch_size', type=int, default=int(1e9), help='Batch size in batched CROWN.', hierarchy=h + ["batch_size"])
        self.add_argument('--max_crown_size', type=int, default=int(1e9), help='Max output size in CROWN (when there are too many output neurons, only part of them will be bounded by CROWN).', hierarchy=h + ["max_crown_size"])

        h = ["solver", "alpha-crown"]
        self.add_argument('--no_alpha', action='store_false', dest='alpha', help='Disable/Enable alpha crown.', hierarchy=h + ["alpha"])
        self.add_argument("--lr_init_alpha", type=float, default=0.1, help='Learning rate for the optimizable parameter alpha in alpha-CROWN bound.', hierarchy=h + ["lr_alpha"])
        self.add_argument('--init_iteration', type=int, default=100, help='Number of iterations for alpha-CROWN incomplete verifier.', hierarchy=h + ["iteration"])
        self.add_argument("--share_slopes", action='store_true', help='Share some alpha variables to save memory at the cost of slightly looser bounds.', hierarchy=h + ["share_slopes"])
        self.add_argument("--no_joint_opt", action='store_true', help='Run alpha-CROWN bounds without joint optimization (only optimize alpha for the last layer bound).', hierarchy=h + ["no_joint_opt"])
        self.add_argument("--alpha_lr_decay", type=float, default=0.98, help='Learning rate decay factor during alpha-CROWN optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.', hierarchy=h + ["lr_decay"])
        self.add_argument("--full_conv_alpha", action='store_false', help='Use independent alpha for conv layers.', hierarchy=h + ["full_conv_alpha"])

        h = ["solver", "beta-crown"]
        self.add_argument("--batch_size", type=int, default=64, help='Batch size in beta-CROWN (number of parallel splits).', hierarchy=h + ["batch_size"])
        self.add_argument('--min_batch_size_ratio', type=float, default=0.1, help='The minimum batch size ratio in each iteration (splitting multiple layers if the number of domains is smaller than min_batch_size_ratio * batch_size).', hierarchy=h + ["min_batch_size_ratio"])
        self.add_argument("--lr_alpha", type=float, default=0.01, help='Learning rate for optimizing alpha during branch and bound.', hierarchy=h + ["lr_alpha"])
        self.add_argument("--lr_beta", type=float, default=0.05, help='Learning rate for optimizing beta during branch and bound.', hierarchy=h + ["lr_beta"])
        self.add_argument("--beta_lr_decay", type=float, default=0.98, help='Learning rate decay factor during beta-CROWN optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.', hierarchy=h + ["lr_decay"])
        self.add_argument("--optimizer", default="adam", help='Optimizer used for alpha and beta optimization.', hierarchy=h + ["optimizer"])
        self.add_argument("--iteration", type=int, default=50, help='Number of iteration for optimizing alpha and beta during branch and bound.', hierarchy=h + ["iteration"])
        self.add_argument('--no_beta', action='store_false', dest='beta', help='Disable/Enable beta split constraint (this option is for ablation study only and should not be used normally).', hierarchy=h + ["beta"])
        self.add_argument('--no_beta_warmup', action='store_false', dest='beta_warmup', help='Do not use beta warmup from branching history (this option is for ablation study only and should not be used normally).', hierarchy=h + ["beta_warmup"])
        self.add_argument('--enable_opt_interm_bounds', action='store_true', default=False, help='Enable optimizing intermediate bounds for beta-CROWN, only used when mip refine for now.', hierarchy=h + ["enable_opt_interm_bounds"])
        self.add_argument('--enable_all_node_split_LP', action='store_true', dest="all_node_split_LP", default=False, help='When all nodes are split during Bab but not verified, using LP to check.', hierarchy=h + ["all_node_split_LP"])

        h = ["solver", "forward"]
        self.add_argument('--forward_refine', action='store_true', help='Refine forward bound with CROWN for unstable neurons.', hierarchy=h + ["refine"])
        self.add_argument('--dynamic_forward', action='store_true', help='Use dynamic forward bound propagation where new input variables may be dynamically introduced for nonlinearities.', hierarchy=h + ["dynamic"])
        self.add_argument('--forward_max_dim', type=int, default=10000, help='Maximum input dimension for forward bounds in a batch.', hierarchy=h + ["max_dim"])

        ### preprocessor-hint: private-section-start
        h = ["solver", "intermediate_refinement"]
        self.add_argument('--opt_intermediate_beta', action='store_true', dest='opt_intermediate_beta', help='optimize constraint bias in compute bounds', hierarchy=h + ["enabled"], private=True)
        self.add_argument("--refinement_batch_size", type=int, default=10, help='batch size used for intermediate beta refinement.', hierarchy=h + ["batch_size"], private=True)
        self.add_argument('--opt_coeffs', action='store_true', dest='opt_coeffs', help='optimize coeffs in compute bounds', hierarchy=h + ["opt_coeffs"], private=True)
        self.add_argument('--opt_bias', action='store_true', dest='opt_bias', help='optimize constraint bias in compute bounds', hierarchy=h + ["opt_bias"], private=True)
        self.add_argument("--lr_intermediate_beta", type=float, default=0.05, help='learning rate for intermediate layer beta for refinement', hierarchy=h + ["lr"], private=True)
        self.add_argument('--intermediate_refinement_layers', nargs='+', type=int, default=[-1], help='layers to be refined, separated by commas. -1 means preactivation before last relu.', hierarchy=h + ["layers"], private=True)
        self.add_argument('--max_refinement_domains', type=int, default=1000, help='the max length of domains that we can use the opt_intermediate_beta', hierarchy=h + ["max_domains"], private=True)
        self.add_argument('--solver_pkg', type=str, default='gurobi', help="The LP solver package that we will use.", choices=["gurobi", "scip"], hierarchy=h + ["solver_pkg"])
        ### preprocessor-hint: private-section-end

        h = ["solver", "multi_class"]
        self.add_argument('--multi_class_method', type=str, help='Method for handling multi-class verification, which could be "loop", "allclass_domain", "splitable_domain".',
                          default='allclass_domain', choices=['loop', 'allclass_domain'], hierarchy=h + ["multi_class_method"])
        self.add_argument('--label_batch_size', type=int, help='Maximum target labels to handle in alpha-CROWN. Cannot be too large due to GPU memory limit.',
                          default=32, hierarchy=h + ["label_batch_size"])
        self.add_argument('--no_skip_with_refined_bound', action='store_false', dest='skip_with_refined_bound', hierarchy=h + ['skip_with_refined_bound'],
                          help='By default we skip the second alpha-CROWN execution if all slopes are already initialized. Setting this to avoid this feature.')

        h = ["solver", "mip"]
        self.add_argument('--mip_multi_proc', type=int, default=None,
                help='Number of multi-processes for mip solver. Each process computes a mip bound for an intermediate neuron. Default (None) is to auto detect the number of CPU cores (note that each process may use multiple threads, see the next option).', hierarchy=h + ["parallel_solvers"])
        self.add_argument('--mip_threads', type=int, default=1,
                help='Number of threads for echo mip solver process (default is to use 1 thread for each solver process).', hierarchy=h + ["solver_threads"])
        self.add_argument('--mip_perneuron_refine_timeout', type=float, default=15, help='MIP timeout threshold for improving each intermediate layer bound (in seconds).', hierarchy=h + ["refine_neuron_timeout"])
        self.add_argument('--mip_refine_timeout', type=float, default=0.8, help='Percentage (x100%) of time used for improving all intermediate layer bounds using mip. Default to be 0.8*timeout.', hierarchy=h + ["refine_neuron_time_percentage"])
        self.add_argument('--no_mip_early_stop', action='store_false', dest='mip_early_stop', help='Not early stop when finding a positive lower bound or a adversarial example during MIP.', hierarchy=h + ["early_stop"], private=True)
        self.add_argument('--disable_adv_warmup', action='store_false', dest='adv_warmup', help='Disable using PGD adv as MIP refinement warmup starts.', hierarchy=h + ["adv_warmup"], private=True)

        h = ["bab"]
        self.add_argument("--initial_max_domains", type=int, default=1, help='Number of domains we can add to domain list at the same time before bab.', hierarchy=h + ["initial_max_domains"])
        self.add_argument("--max_domains", type=int, default=200000, help='Max number of subproblems in branch and bound.', hierarchy=h + ["max_domains"])
        self.add_argument("--decision_thresh", type=float, default=0, help='Decision threshold of lower bounds. When lower bounds are greater than this value, verification is successful. Set to 0 for robustness verification.', hierarchy=h + ["decision_thresh"])
        # FIXME timeout should not be under "bab"
        self.add_argument("--timeout", type=float, default=360, help='Timeout (in second) for verifying one image/property.', hierarchy=h + ["timeout"])
        self.add_argument("--timeout_scale", type=float, default=1, help='Scale the timeout for development purpose.', hierarchy=h + ["timeout_scale"])
        self.add_argument("--override_timeout", type=float, default=None, help='Override timeout.', hierarchy=h + ["override_timeout"])
        self.add_argument("--get_upper_bound", action='store_true', help='Update global upper bound during BaB (has extra overhead, typically the upper bound is not used).', hierarchy=h + ["get_upper_bound"])
        self.add_argument("--DFS_percent", type=float, default=0., help='Percent of domains for depth first search (not used).', hierarchy=h + ["dfs_percent"])
        self.add_argument("--disable_pruning_in_iteration", action='store_false', dest='pruning_in_iteration', help='Disable verified domain pruning within iteration.', hierarchy=h + ["pruning_in_iteration"])
        self.add_argument("--pruning_in_iteration_ratio", type=float, default=0.2, help='When ratio of positive domains >= this ratio, prunning in iteration optimization is open.', hierarchy=h + ["pruning_in_iteration_ratio"])
        self.add_argument('--sort_targets', action='store_true', help='Sort targets before BaB.', hierarchy=h + ["sort_targets"])
        self.add_argument("--disable_batched_domain_list", action='store_false', dest='batched_domain_list', help='Disable batched domain list. Batched domain list is faster but picks domain to split in an unsorted way.', hierarchy=h + ['batched_domain_list'])
        self.add_argument("--optimized_intermediate_layers", type=str, default="", help='A list of layer names that will be optimized during branch and bound, separated by comma.', hierarchy=h + ['optimized_intermediate_layers'])
        self.add_argument("--no_interm_transfer", action='store_false', dest='interm_transfer', help='Skip the intermediate bound transfer to save transfer-to-CPU time. Require intermediate bound does not change. Caution: cannot be used with cplex cut or intermediate beta refinement.', hierarchy=h + ['interm_transfer'])

        ### preprocessor-hint: private-section-start
        # FIXME: cut should not be under bab. We don't have to use bab with cuts. It should be under "solver" category.
        h = ["bab", "cut"]
        self.add_argument('--enable_cut', action='store_true', dest='enable_cut', help='add initial cuts before BaB', hierarchy=h + ["enabled"], private=True)
        self.add_argument('--enable_bab_cut', action='store_true', dest='enable_bab_cut', help='enable cut constraints optimization during BaB', hierarchy=h + ["bab_cut"], private=True)
        self.add_argument('--enable_lp_cut', action='store_true', dest='enable_lp_cut', help='enable lp with cut constraints to debug', hierarchy=h + ["lp_cut"], private=True)
        self.add_argument('--cut_method', help='cutting plane method', hierarchy=h + ["method"], private=True)
        self.add_argument("--cut_lr_decay", type=float, default=1.0, help='Learning rate decay for optimizing cut betas.', hierarchy=h + ["lr_decay"])
        self.add_argument("--cut_iteration", type=int, default=100, help='Iterations for optimizing cut betas.', hierarchy=h + ["iteration"])
        self.add_argument("--cut_bab_iteration", type=int, default=-1, help='Iterations for optimizing cut betas during branch and bound. Set to -1 to use the same number of iterations without cuts.', hierarchy=h + ["bab_iteration"])
        self.add_argument("--cut_early_stop_patience", type=int, default=-1, help='Early stop patience for optimizing cuts. Set to -1 to use the same value when cuts are not used.', hierarchy=h + ["early_stop_patience"])
        self.add_argument("--cut_lr_beta", type=float, default=0.02, help='Learning rate for optimizing cut betas.', hierarchy=h + ["lr_beta"])
        self.add_argument("--number_cuts", type=int, default=50, help='Maximum number of cuts that we want to add.', hierarchy=h + ["number_cuts"])
        self.add_argument("--topk_cuts_in_filter", type=int, default=100, help='Only keep top K constraints when filtering cuts.', hierarchy=h + ["topk_cuts_in_filter"])
        self.add_argument("--batch_size_primal", type=int, default=100, help='Batch size when calculate primals, should be negative correlated to number of unstable neurons.', hierarchy=h + ["batch_size_primal"])
        self.add_argument("--add_implied_cuts", action='store_true', help='Add implied bound cuts.', hierarchy=h + ["add_implied_cuts"])
        self.add_argument("--add_input_cuts", action='store_true', help='Add input cuts.', hierarchy=h + ["add_input_cuts"])
        self.add_argument("--tmp_cuts", default=None, help='Automatically generated cuts if add_implied_cuts is True; otherwise one can feed manual cuts using this argument.', hierarchy=h + ["_tmp_cuts"])
        self.add_argument("--eran_cuts", default=None, help='Directory containing cuts exported from ERAN.', hierarchy=h + ["_eran_cuts"])
        self.add_argument("--cut_max_num", type=int, default=int(1e9), help='Maximum number of cuts.', hierarchy=h + ["max_num"])
        self.add_argument("--fixed_cuts", action='store_true', help='Use cuts generated from optimizable cuts but do not optimize them', private=True, hierarchy=h + ["fixed_cuts"])
        self.add_argument("--enable_patches_cut", action='store_true', help='if we need to optimize cuts bounds for conv patches layer', private=True, hierarchy=h + ["patches_cut"])
        self.add_argument("--cplex_cuts", action='store_true', help='Build and save mip mps models, let cplex find cuts, and use found cuts to improve lbs', private=True, hierarchy=h + ["cplex_cuts"])
        self.add_argument("--cplex_cuts_wait", type=float, default=0, help='Wait a bit after cplex warmup in seconds, so that we tend to get some cuts at early stage of bab', private=True, hierarchy=h + ["cplex_cuts_wait"])
        self.add_argument("--no_cplex_cuts_revpickup", action='store_false', help='Disable the inverse order domain pickup when cplex is enabled', private=True, hierarchy=h + ["cplex_cuts_revpickup"])
        self.add_argument("--no_cut_reference_bounds", action='store_false', help='Disable using reference bounds when cuts are used', private=True, hierarchy=h + ["cut_reference_bounds"])
        self.add_argument("--fix_cut_intermediate_bounds", action='store_true', help='Fix intermediate bounds when cuts are used', private=True, hierarchy=h + ["fix_intermediate_bounds"])
        self.add_argument("--lr_cuts", type=float, default=0.01, help='Learning rate for optimizing cuts', private=True, hierarchy=h + ["lr"])
        ### preprocessor-hint: private-section-end

        h = ["bab", "branching"]
        self.add_argument("--branching_method", default="kfsb", choices=["babsr", "fsb", "kfsb", "sb", "kfsb-intercept-only", "sb-fast", "naive"], help='Branching heuristic. babsr is fast but less accurate; fsb is slow but most accurate; kfsb is usualy a balance; kfsb-intercept-only relies on intercept only and improves the runtime; sb-fast is fast smart branching which relies on the A matrix.', hierarchy=h + ["method"])
        self.add_argument("--branching_candidates", type=int, default=3, help='Number of candidates to consider when using fsb or kfsb. More leads to slower but better branching.', hierarchy=h + ["candidates"])
        self.add_argument("--branching_reduceop", choices=["min", "max", "mean", "auto"], default="min", help='Reduction operation to compute branching scores from two sides of a branch (min or max). max can work better on some models.', hierarchy=h + ["reduceop"])
        self.add_argument("--sb_coeff_thresh", default=1e-3, type=float, help='Clamp values of coefficient matrix (A matrix) for sb-fast branching heuristic.', hierarchy=h + ["sb_coeff_thresh"])

        h = ["bab", "branching", "input_split"]
        self.add_argument("--enable_input_split", action='store_true', help='Branch on input domain rather than unstable neurons.', hierarchy=h + ["enable"])
        self.add_argument('--enhanced_bound_prop_method', default="alpha-crown", choices=["alpha-crown", "crown", "forward+crown", "crown-ibp"], help='Specify a tighter bound propgation method if a problem cannot be verified after --input_split_enhanced_bound_patience.', hierarchy=h + ["enhanced_bound_prop_method"])
        self.add_argument('--enhanced_branching_method', default="naive", choices=["sb", "sb-fast", "naive"], help='Specify a branching method if a problem cannot be verified after --input_split_enhanced_bound_patience.', hierarchy=h + ["enhanced_branching_method"])
        self.add_argument("--input_split_enhanced_bound_patience", type=int, default=1e8, help='Time in seconds that will use an enhanced bound propagation method (e.g., alpha-CROWN) to bound input split sub domains.', hierarchy=h + ["enhanced_bound_patience"])
        self.add_argument("--input_split_attack_patience", type=int, default=1e8, help='Time in seconds that will start PGD attack to find adv examples during input split.', hierarchy=h + ["attack_patience"])
        self.add_argument("--input_split_adv_check", type=int, default=0, help='After the number of visited nodes, we will run adv_check in input split.', hierarchy=h + ["adv_check"])
        self.add_argument("--sort_domain_interval", type=int, default=-1, help='If unsorted domains are used, sort the domains every sort_domain_interval iterations.', hierarchy=h + ["sort_domain_interval"])

        ### preprocessor-hint: private-section-start
        h = ["bab", "attack"]
        self.add_argument("--beam_dive", action='store_true', help='using beam dive only for adv, no verification any more', hierarchy=h + ["enabled"], private=True)
        self.add_argument("--candidates_number", type=int, default=8, help='number of candidates we selected during beam dive', hierarchy=h + ["beam_candidates"], private=True)
        self.add_argument("--split_depth", type=int, default=7, help='max split depth of bab during beam dive', hierarchy=h + ["beam_depth"], private=True)
        self.add_argument('--max_dive_fix_ratio', type=float, default=0.8, help='maximum portion of fixed neurons during diving.', hierarchy=h + ['max_dive_fix_ratio'], private=True)
        self.add_argument('--min_local_free_ratio', type=float, default=0.2, help='minimum portion of free neurons during local search.', hierarchy=h + ['min_local_free_ratio'], private=True)
        self.add_argument('--mip_dive_timeout', type=float, default=30.0, help='Timeout when MIP is used in diving.', hierarchy=h + ["mip_timeout"], private=True)
        self.add_argument('--mip_dive_start', type=int, default=5, help='Which iteration to start MIP when using beam search.', hierarchy=h + ["mip_start_iteration"], private=True)
        # The options below will be removed.
        self.add_argument("--max_dive_domains", type=int, default=-1, help='maximum dive domains maintained for adv detection, -1 if no dive domains', hierarchy=h + ["max_dive_domains"], private=True)
        self.add_argument("--num_dive_constraints", type=int, default=50, help='number of extra constraints applied to each dive domain', hierarchy=h + ["num_dive_constraints"], private=True)
        self.add_argument("--dive_rate", type=float, default=0.2, help='at most how much portion of dive domains in a full batch', hierarchy=h + ["dive_rate"], private=True)
        self.add_argument("--adv_dive", action='store_true', help='using adversarial example to initial dive', hierarchy=h + ["adv_dive"], private=True)
        self.add_argument('--adv_pool_threshold', type=float, default=None, help='Minimum value of difference when adding to adv_pool, default auto select..', hierarchy=h + ["adv_pool_threshold"], private=True)
        self.add_argument("--refined_mip_attacker", action='store_true', help='if we want to use full alpha crown bounds to refined intermediate bounds for mip solver attack', hierarchy=h + ["refined_mip_attacker"], private=True)
        self.add_argument("--refined_batch_size", type=float, default=None, help='what is the batch size used for full alpha crown bounds to refined intermediate bounds for mip solver attack (to avoid OOM), default None to be the same as mip_multi_proc', hierarchy=h + ["refined_batch_size"], private=True)
        ### preprocessor-hint: private-section-end

        h = ["attack"]
        self.add_argument('--pgd_order', choices=["before", "after", "middle", "skip"], default="before",  help='Run PGD before/after incomplete verification, or skip it.', hierarchy=h + ["pgd_order"])
        self.add_argument('--pgd_steps', type=int, default=100, help="Steps of PGD attack.", hierarchy=h + ["pgd_steps"])
        self.add_argument('--pgd_restarts', type=int, default=30, help="Number of random PGD restarts.", hierarchy= h + ["pgd_restarts"])
        self.add_argument('--no_pgd_early_stop', action='store_false', dest='pgd_early_stop', help="Early stop PGD when an adversarial example is found.", hierarchy=h + ["pgd_early_stop"])
        self.add_argument('--pgd_lr_decay', type=float, default=0.99, help='Learning rate decay factor used in PGD attack.', hierarchy= h + ["pgd_lr_decay"])
        self.add_argument('--pgd_alpha', type=str, default="auto", help='Step size of PGD attack. Default (auto) is epsilon/4.', hierarchy=h + ["pgd_alpha"])
        self.add_argument('--pgd_loss_mode', type=str, choices=['hinge', 'sum'], help='Loss mode for choosing the best delta.', hierarchy=h + ["pgd_loss_mode"])
        ### preprocessor-hint: private-section-start
        self.add_argument('--attack_mode', type=str, default='PGD', help='Attack mode.', choices=['auto_attack', 'diversed_PGD', 'diversed_GAMA_PGD', 'PGD', 'boundary'], hierarchy=h + ["attack_mode"])
        ### preprocessor-hint: private-section-start
        self.add_argument('--attack_gama_lambda', type=float, default=10., help='Regularization parameter in GAMA attack.', hierarchy=h + ["gama_lambda"])
        self.add_argument('--attack_gama_decay', type=float, default=0.9, help='Decay of regularization parameter in GAMA attack.', hierarchy=h + ["gama_decay"])
        self.add_argument('--check_clean', action='store_true', help='Check clean prediction for attack.', hierarchy=h + ["check_clean"])

        h = ["attack", "input_split"]
        self.add_argument('--input_split_pgd_steps', type=int, default=100, help="Steps of PGD attack in input split.", hierarchy= h + ["pgd_steps"])
        self.add_argument('--input_split_pgd_restarts', type=int, default=30, help="Number of random PGD restarts in input split.", hierarchy= h + ["pgd_restarts"])
        self.add_argument('--input_split_pgd_alpha', type=str, default="auto", help="Step size (alpha) in input split.", hierarchy= h + ["pgd_alpha"])

        h = ["attack", "input_split_enhanced"]
        self.add_argument('--input_split_enhanced_pgd_steps', type=int, default=200, help="Steps of PGD attack in massive pgd attack in input split.", hierarchy= h + ["pgd_steps"])
        self.add_argument('--input_split_enhanced_pgd_restarts', type=int, default=5000000, help="Number of random PGD restarts in massive pgd attack in input split.", hierarchy= h + ["pgd_restarts"])
        self.add_argument('--input_split_enhanced_pgd_alpha', type=str, default="auto", help="Step size (alpha) in massive pgd attack in input split.", hierarchy= h + ["pgd_alpha"])

        h = ["attack", "input_split_check_adv"]
        self.add_argument('--input_split_check_adv_pgd_steps', type=int, default=5, help="Steps of PGD attack in 'check_adv' in input split.", hierarchy= h + ["pgd_steps"])
        self.add_argument('--input_split_check_adv_pgd_restarts', type=int, default=5, help="Number of random PGD restarts in 'check_adv' in input split.", hierarchy= h + ["pgd_restarts"])
        self.add_argument('--input_split_check_adv_pgd_alpha', type=str, default="auto", help="Step size (alpha) in 'check_adv' in input split.", hierarchy= h + ["pgd_alpha"])
        ### preprocessor-hint: private-section-end
        

    def add_argument(self, *args, **kwargs):
        """Add a single parameter to the parser. We will check the 'hierarchy' specified and then pass the remaining arguments to argparse."""
        if 'hierarchy' not in kwargs:
            raise ValueError("please specify the 'hierarchy' parameter when using this function.")
        hierarchy = kwargs.pop('hierarchy')
        help = kwargs.get('help', '')
        private_option = kwargs.pop('private', False)
        # Make sure valid help is given
        if not private_option:
            if len(help.strip()) < 10:
                raise ValueError(f'Help message must not be empty, and must be detailed enough. "{help}" is not good enough.')
            elif (not help[0].isupper()) or help[-1] != '.':
                raise ValueError(f'Help message must start with an upper case letter and end with a dot (.); your message "{help}" is invalid.')
        self.defaults_parser.add_argument(*args, **kwargs)
        # Build another parser without any defaults.
        if 'default' in kwargs:
            kwargs.pop('default')
        self.no_defaults_parser.add_argument(*args, **kwargs)
        # Determine the variable that will be used to save the argument by argparse.
        if 'dest' in kwargs:
            dest = kwargs['dest']
        else:
            dest = re.sub('^-*', '', args[-1]).replace('-', '_')
        # Also register this parameter to the hierarchy dictionary.
        self.config_file_hierarchies[dest] = hierarchy
        if hierarchy is not None and not private_option:
            self.help_messages[','.join(hierarchy)] = help

    def set_dict_by_hierarchy(self, args_dict, h, value, nonexist_ok=True):
        """Insert an argument into the dictionary of all parameters. The level in this dictionary is determined by list 'h'."""
        # Create all the levels if they do not exist.
        current_level = self.all_args
        assert len(h) != 0
        for config_name in h:
            if config_name not in current_level:
                if nonexist_ok:
                    current_level[config_name] = {}
                else:
                    raise ValueError(f"Config key {h} not found!")
            last_level = current_level
            current_level = current_level[config_name]
        # Add config value to leaf node.
        last_level[config_name] = value

    def construct_config_dict(self, args_dict, nonexist_ok=True):
        """Based on all arguments from argparse, construct the dictionary of all parameters in self.all_args."""
        for arg_name, arg_val in args_dict.items():
            h = self.config_file_hierarchies[arg_name]  # Get levels for this argument.
            if h is not None:
                assert len(h) != 0
                self.set_dict_by_hierarchy(self.all_args, h, arg_val, nonexist_ok=nonexist_ok)

    def update_config_dict(self, old_args_dict, new_args_dict, levels=None):
        """Recursively update the dictionary of all parameters based on the dict read from config file."""
        if levels is None:
            levels = []
        if isinstance(new_args_dict, dict):
            # Go to the next dict level.
            for k in new_args_dict:
                self.update_config_dict(old_args_dict, new_args_dict[k], levels=levels + [k])
        else:
            # Reached the leaf level. Set the corresponding key.
            self.set_dict_by_hierarchy(old_args_dict, levels, new_args_dict, nonexist_ok=False)

    def dump_config(self, args_dict, level=[], out_to_doc=False, show_help=False):
        """Generate a config file based on aargs_dict with help information."""
        ret_string = ''
        for key, val in args_dict.items():
            if isinstance(val, dict):
                ret = self.dump_config(val, level + [key], out_to_doc, show_help)
                if len(ret) > 0:
                    # Next level is not empty, print it.
                    ret_string += ' ' * (len(level) * 2) + f'{key}:\n' + ret
            else:
                if show_help:
                    h = self.help_messages[','.join(level + [key])]
                    if 'debug' in key or 'not use' in h or 'not be use' in h or 'debug' in h or len(h) == 0:
                        # Skip some debugging options.
                        continue
                    h = f'  # {h}'
                else:
                    h = ''
                yaml_line = yaml.safe_dump({key: val}, default_flow_style=None).strip().replace('{', '').replace('}', '')
                ret_string += ' ' * (len(level) * 2) + f'{yaml_line}{h}\n'
        if len(level) > 0:
            return ret_string
        else:
            # Top level, output to file.
            if out_to_doc:
                output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs',
                        os.path.splitext(os.path.basename(sys.argv[0]))[0] + '_all_params.yaml')
                with open(output_name, 'w') as f:
                    f.write(ret_string)
            return ret_string

    def parse_config(self):
        """
        Main function to parse parameter configurations. The commandline arguments have the highest priority;
        then the parameters specified in yaml config file. If a parameter does not exist in either commandline
        or the yaml config file, we use the defaults defined in add_common_options() defined above.
        """
        # Parse an empty commandline to get all default arguments.
        default_args = vars(self.defaults_parser.parse_args([]))
        # Create the dictionary of all parameters, all set to their default values.
        self.construct_config_dict(default_args)
        # Update documents.
        # self.dump_config(self.all_args, out_to_doc=True, show_help=True)
        # These are arguments specified in command line.
        specified_args = vars(self.no_defaults_parser.parse_args())
        # Read the yaml config files.
        if 'config' in specified_args:
            with open(specified_args['config'], 'r') as config_file:
                loaded_args = yaml.safe_load(config_file)
                # Update the defaults with the parameters in the config file.
                self.update_config_dict(self.all_args, loaded_args)
        # Finally, override the parameters based on commandline arguments.
        self.construct_config_dict(specified_args, nonexist_ok=False)
        # For compatibility, we still return all the arguments from argparser.
        parsed_args = self.defaults_parser.parse_args()
        # FIXME remove this option as it has been deprecated
        if self["solver"]["multi_class"]["multi_class_method"] != "allclass_domain":
            raise RuntimeError('--multi_class_method is deprecated')
        # Print all configuration.
        print('Configurations:\n')
        print(self.dump_config(self.all_args))
        return parsed_args

    def keys(self):
        return self.all_args.keys()

    def items(self):
        return self.all_args.items()

    def __getitem__(self, key):
        """Read an item from the dictionary of parameters."""
        return self.all_args[key]

    def __setitem__(self, key, value):
        """Set an item from the dictionary of parameters."""
        self.all_args[key] = value


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("You must register a global parameter in arguments.py.")
    def __setitem__(self, key, value):
        if key not in self:
            raise RuntimeError("You must register a global parameter in arguments.py.")
        else:
            super().__setitem__(key, value)
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


# Global configuration variable
Config = ConfigHandler()
# Global variables
Globals = ReadOnlyDict({"starting_timestamp": int(time.time()), "example_idx": -1})

