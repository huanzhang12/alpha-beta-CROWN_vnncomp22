import time
import os
import numpy as np
import warnings
from collections import OrderedDict, deque
from contextlib import ExitStack

import torch
import torch.optim as optim
from torch.nn import Parameter

from .bound_op_map import bound_op_map
from .bound_ops import *
from .bounded_tensor import BoundedTensor, BoundedParameter
from .parse_graph import parse_module
from .perturbations import *
from .utils import *
from .patches import Patches
from .adam_element_lr import AdamElementLR
from .cuda_utils import double2float  # FIXME: (assigned to kaidi) use cuda based conversion functions.

### preprocessor-hint: private-section-start
# Debugging variables.
from .intermediate_refinement import Check_against_base_lp, Check_against_base_lp_layer
### preprocessor-hint: private-section-end


warnings.simplefilter("once")

class BoundedModule(nn.Module):
    """Bounded module with support for automatically computing bounds.

    Args:
        model (nn.Module): The original model to be wrapped by BoundedModule.

        global_input (tuple): A dummy input to the original model. The shape of
        the dummy input should be consistent with the actual input to the model
        except for the batch dimension.

        bound_opts (dict): Options for bounds. See
        `Bound Options <bound_opts.html>`_.

        device (str or torch.device): Device of the bounded module.
        If 'auto', the device will be automatically inferred from the device of
        parameters in the original model or the dummy input.

        custom_ops (dict): A dictionary of custom operators.
        The dictionary maps operator names to their corresponding bound classes
        (subclasses of `Bound`).

    """
    def __init__(self, model, global_input, bound_opts=None, auto_batch_dim=False, device='auto',
                 verbose=False, custom_ops={}):
        super(BoundedModule, self).__init__()
        if isinstance(model, BoundedModule):
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return

        self.global_input = global_input
        self.ori_training = model.training
        self.check_incompatible_nodes(model)

        if bound_opts is None:
            bound_opts = {}
        # Default options.
        default_bound_opts = {
            'ibp_relative': False,
            'conv_mode': 'patches',
            'sparse_intermediate_bounds': True,
            'sparse_conv_intermediate_bounds': True,
            'sparse_intermediate_bounds_with_ibp': True,
            'sparse_features_alpha': True,
            'sparse_spec_alpha': True,
            'minimum_sparsity': 0.9,
            'enable_opt_interm_bounds': False,
            'crown_batch_size': np.inf,
            'forward_refinement': False,
            'dynamic_forward': False,
            'forward_max_dim': int(1e9),
            'use_full_conv_alpha': True,  # Do not share alpha for conv layers.
            'use_full_conv_alpha_thresh': 512,  # Threshold for number of unstable neurons for each layer to disable use_full_conv_alpha.
        }
        default_bound_opts.update(bound_opts)
        self.bound_opts = default_bound_opts
        self.verbose = verbose
        self.custom_ops = custom_ops
        self.auto_batch_dim = auto_batch_dim
        if device == 'auto':
            try:
                self.device = next(model.parameters()).device
            except StopIteration:  # Model has no parameters. We use the device of input tensor.
                self.device = global_input.device
        else:
            self.device = device
        self.ibp_relative = self.bound_opts.get('ibp_relative', False)
        self.conv_mode = self.bound_opts.get('conv_mode', 'patches')

        if auto_batch_dim:
            logger.warning('Using automatic batch dimension inferring is not recommended.')
            self.init_batch_size = -1
        self.optimizable_activations = []
        self.relus = []  # save relu layers for convenience

        state_dict_copy = copy.deepcopy(model.state_dict())
        object.__setattr__(self, 'ori_state_dict', state_dict_copy)
        model.to(self.device)
        self.final_shape = model(*unpack_inputs(global_input, device=self.device)).shape
        self.bound_opts.update({'final_shape': self.final_shape})
        self._convert(model, global_input)
        self._mark_perturbed_nodes()

        # set the default values here
        optimize_bound_args = {'ob_iteration': 20, 'ob_beta': False, 'ob_alpha': True, 'ob_alpha_share_slopes': False,
                               'ob_opt_coeffs': False, 'ob_opt_bias': False,
                               'ob_optimizer': 'adam', 'ob_verbose': 0,
                               'ob_keep_best': True, 'ob_update_by_layer': True, 'ob_lr': 0.5,
                               'ob_lr_beta': 0.05, 'ob_lr_cut_beta': 5e-3, 'ob_init': True,
                               'ob_single_node_split': True, 'ob_lr_intermediate_beta': 0.1,
                               'ob_lr_coeffs': 0.01, 'ob_intermediate_beta': False, 'ob_intermediate_refinement_layers': [-1],
                               'ob_loss_reduction_func': reduction_sum, 'ob_stop_criterion_func': lambda x: False,
                               'ob_input_grad': False, 'ob_lr_decay': 0.98, 'ob_early_stop_patience': 10,
                               'ob_start_save_best': 2,
                               'ob_no_float64_last_iter': False, 'ob_no_amp': False,
                               'ob_pruning_in_iteration': False,
                               'ob_pruning_in_iteration_threshold': 0.2,
                               'ob_multi_spec_keep_func': lambda x: True} # always keep the domain
        # change by bound_opts
        optimize_bound_args.update(self.bound_opts.get('optimize_bound_args', {}))
        self.bound_opts.update({'optimize_bound_args': optimize_bound_args})

        self.next_split_hint = []  # Split hints, used in beta optimization.
        # Beta values for all intermediate bounds. Set to None (not used) by default.
        self.best_intermediate_betas = None
        # Initialization value for intermediate betas.
        self.init_intermediate_betas = None
        # whether using cut
        self.cut_used = False
        # a placeholder for cut timestamp, which would be a non-positive int
        self.cut_timestamp = -1

        # List of operators. When we are computing intermediate bounds for these
        # ops, we simply use IBP to propagate bounds from its input nodes,
        # instead of CROWN.
        self.ibp_intermediate = [BoundRelu, BoundNeg, BoundTranspose]

        # a placeholder to save the latest samplewise mask for pruning-in-iteration optimization
        self.last_update_preserve_mask = None

    """check whether the model has incompatible nodes that the conversion may be inaccurate"""
    def check_incompatible_nodes(self, model):
        node_types = [type(m) for m in list(model.modules())]

        if torch.nn.Dropout in node_types and torch.nn.BatchNorm1d in node_types and self.global_input.shape[0] == 1:
            print('We cannot support torch.nn.Dropout and torch.nn.BatchNorm1d at the same time!')
            print('Suggest to use another dummy input which has batch size larger than 1 and set model to train() mode.')
            return

        if not self.ori_training and torch.nn.Dropout in node_types:
            print('Dropout operation CANNOT be parsed during conversion when the model is in eval() mode!')
            print('Set model to train() mode!')
            self.ori_training = True

        if self.ori_training and torch.nn.BatchNorm1d in node_types:
            print('BatchNorm1d may raise error during conversion when the model is in train() mode!')
            print('Set model to eval() mode!')
            self.ori_training = False

    """Some operations are non-deterministic and deterministic mode will fail. So we temporary disable it."""
    def non_deter_wrapper(self, op, *args, **kwargs):
        if self.bound_opts.get('deterministic', False):
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)
        if self.bound_opts.get('deterministic', False):
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)

    def set_bound_opts(self, new_opts):
        for k, v in new_opts.items():
            # assert v is not dict, 'only support change optimize_bound_args'
            if type(v) == dict:
                self.bound_opts[k].update(v)
            else:
                self.bound_opts[k] = v

    @staticmethod
    def _get_A_norm(A):
        if not isinstance(A, (list, tuple)):
            A = (A, )
        norms = []
        for aa in A:
            if aa is not None:
                if isinstance(aa, Patches):
                    aa = aa.patches
                norms.append(aa.abs().sum().item())
            else:
                norms.append(None)
        return norms

    def __call__(self, *input, **kwargs):
        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            opt = "forward"
        for kwarg in [
            'disable_multi_gpu', 'no_replicas', 'get_property',
            'node_class', 'att_name']:
            if kwarg in kwargs:
                kwargs.pop(kwarg)
        if opt == "compute_bounds":
            return self.compute_bounds(**kwargs)
        else:
            return self.forward(*input, **kwargs)

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def load_state_dict(self, state_dict, strict=False):
        new_dict = OrderedDict()
        # translate name to ori_name
        for k, v in state_dict.items():
            if k in self.node_name_map:
                new_dict[self.node_name_map[k]] = v
        return super(BoundedModule, self).load_state_dict(new_dict, strict=strict)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                # translate name to ori_name
                if name in self.node_name_map:
                    name = self.node_name_map[name]
                yield name, v

    def train(self, mode=True):
        super().train(mode)
        for node in self._modules.values():
            node.train(mode=mode)

    def eval(self):
        super().eval()
        for node in self._modules.values():
            node.eval()

    def to(self, *args, **kwargs):
        # Moves and/or casts some attributes except pytorch will do by default.
        for node in self._modules.values():
            for attr in ['lower', 'upper', 'forward_value', 'd', 'lA',]:
                if hasattr(node, attr):
                    this_attr = getattr(node, attr)
                    if isinstance(this_attr, torch.Tensor):
                        # print(node, attr)
                        this_attr = this_attr.to(*args, **kwargs)
                        setattr(node, attr, this_attr)

            if hasattr(node, 'interval'):
                # construct new interval
                this_attr = getattr(node, 'interval')
                setattr(node, 'interval', (this_attr[0].to(*args, **kwargs), this_attr[1].to(*args, **kwargs)))

        return super().to(*args, **kwargs)

    def __getitem__(self, name):
        return self._modules[name]

    def final_node(self):
        return self[self.final_name]

    def get_forward_value(self, node):
        """ Recursively get `forward_value` for `node` and its parent nodes"""
        if getattr(node, 'forward_value', None) is not None:
            return node.forward_value
        inputs = [self.get_forward_value(inp) for inp in node.inputs]
        for inp in node.inputs:
            node.from_input = node.from_input or inp.from_input
        node.input_shape = inputs[0].shape if len(inputs) > 0 else None
        fv = node.forward(*inputs)
        if isinstance(fv, torch.Size) or isinstance(fv, tuple):
            fv = torch.tensor(fv, device=self.device)
        node.forward_value = fv
        node.output_shape = fv.shape
        # infer batch dimension
        if self.auto_batch_dim:
            if not hasattr(node, 'batch_dim'):
                inp_batch_dim = [inp.batch_dim for inp in node.inputs]
                try:
                    node.batch_dim = node.infer_batch_dim(self.init_batch_size, *inp_batch_dim)
                except:
                    raise Exception(
                        'Fail to infer the batch dimension of ({})[{}]: forward_value shape {}, input batch dimensions {}'.format(
                            node, node.name, node.forward_value.shape, inp_batch_dim))
        else:
            # In most cases, the batch dimension is just the first dimension
            # if the node depends on input. Otherwise if the node doesn't
            # depend on input, there is no batch dimension.
            node.batch_dim = 0 if node.from_input else -1
        # Unperturbed node but it is not a root node. Save forward_value to value.
        # (Can be used in forward bounds.)
        if not node.from_input and len(node.inputs) > 0:
            node.value = node.forward_value
        return fv

    def forward(self, *x, final_node_name=None, clear_forward_only=False):
        r"""Standard forward computation for the network.

        Args:
            x (tuple or None): Input to the model.

            final_node_name (str, optional): The name of the final node in the model. The value
            on the corresponding node will be returned.

            clear_forward_only (bool, default `False`): Whether only standard forward values stored
            on the nodes should be cleared. If `True`, only standard forward values stored on the
            nodes will be cleared. Otherwise, bound information on the nodes will also be cleared.

        Returns:
            output: The output of the model, or if `final_node_name` is not `None`, return the
            value on the corresponding node instead.
        """
        self._set_input(*x, clear_forward_only=clear_forward_only)
        if final_node_name:
            return self.get_forward_value(self[final_node_name])
        else:
            out = deque([self.get_forward_value(self[n]) for n in self.output_name])

            def _fill_template(template):
                if template is None:
                    return out.popleft()
                elif isinstance(template, list) or isinstance(template, tuple):
                    res = []
                    for t in template:
                        res.append(_fill_template(t))
                    return tuple(res) if isinstance(template, tuple) else res
                elif isinstance(template, dict):
                    res = {}
                    for key in template:
                        res[key] = _fill_template(template[key])
                    return res
                else:
                    raise NotImplementedError

            return _fill_template(self.output_template)

    """Mark the graph nodes and determine which nodes need perturbation."""
    def _mark_perturbed_nodes(self):
        degree_in = {}
        queue = deque()
        # Initially the queue contains all "root" nodes.
        for key in self._modules.keys():
            l = self[key]
            degree_in[l.name] = len(l.inputs)
            if degree_in[l.name] == 0:
                queue.append(l)  # in_degree ==0 -> root node

        while len(queue) > 0:
            l = queue.popleft()
            # Obtain all output node, and add the output nodes to the queue if all its input nodes have been visited.
            # the initial "perturbed" property is set in BoundInput or BoundParams object, depending on ptb.
            for name_next in l.output_name:
                node_next = self[name_next]
                if isinstance(l, BoundShape):
                    # Some nodes like Shape, even connected, do not really propagate bounds.
                    # TODO: make this a property of node?
                    pass
                else:
                    # The next node is perturbed if it is already perturbed, or this node is perturbed.
                    node_next.perturbed = node_next.perturbed or l.perturbed
                degree_in[name_next] -= 1
                if degree_in[name_next] == 0:  # all inputs of this node have been visited, now put it in queue.
                    queue.append(node_next)
        return

    def _clear_and_set_new(self, new_interval, clear_forward_only=False):
        for l in self._modules.values():
            if hasattr(l, 'linear'):
                if isinstance(l.linear, tuple):
                    for item in l.linear:
                        del (item)
                delattr(l, 'linear')

            # FIXME (04/26): We should add a clear() method to each Bound object, and clear desired objects there.
            if hasattr(l, 'patch_size'):
                l.patch_size = {}

            if clear_forward_only:
                if hasattr(l, 'forward_value'):
                    delattr(l, 'forward_value')
            else:
                for attr in ['lower', 'upper', 'interval', 'forward_value', 'd', 'lA', 'lower_d']:
                    if hasattr(l, attr):
                        delattr(l, attr)

            for attr in ['zero_backward_coeffs_l', 'zero_backward_coeffs_u', 'zero_lA_mtx', 'zero_uA_mtx']:
                setattr(l, attr, False)
            # Given an interval here to make IBP/CROWN start from this node
            if new_interval is not None and l.name in new_interval.keys():
                l.interval = tuple(new_interval[l.name][:2])
                l.lower = new_interval[l.name][0]
                l.upper = new_interval[l.name][1]
                if l.lower is not None:
                    l.lower = l.lower.detach().requires_grad_(False)
                if l.upper is not None:
                    l.upper = l.upper.detach().requires_grad_(False)
            # Mark all nodes as non-perturbed except for weights.
            if not hasattr(l, 'perturbation') or l.perturbation is None:
                l.perturbed = False

            # Clear operator-specific attributes
            l.clear()

    def _set_input(self, *x, new_interval=None, clear_forward_only=False):
        self._clear_and_set_new(new_interval=new_interval, clear_forward_only=clear_forward_only)
        inputs_unpacked = unpack_inputs(x)
        for name, index in zip(self.input_name, self.input_index):
            if index is None:
                continue
            node = self[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, (BoundedTensor, BoundedParameter)):
                node.perturbation = node.value.ptb
            else:
                node.perturbation = None
        # Mark all perturbed nodes.
        self._mark_perturbed_nodes()
        if self.auto_batch_dim and self.init_batch_size == -1:
            # Automatic batch dimension inferring: get the batch size from
            # the first dimension of the first input tensor.
            self.init_batch_size = inputs_unpacked[0].shape[0]

    def _get_node_input(self, nodesOP, nodesIn, node):
        ret = []
        ori_names = []
        for i in range(len(node.inputs)):
            found = False
            for op in nodesOP:
                if op.name == node.inputs[i]:
                    ret.append(op.bound_node)
                    break
            if len(ret) == i + 1:
                continue
            for io in nodesIn:
                if io.name == node.inputs[i]:
                    ret.append(io.bound_node)
                    ori_names.append(io.ori_name)
                    break
            if len(ret) <= i:
                raise ValueError('cannot find inputs of node: {}'.format(node.name))
        return ret, ori_names

    def _to(self, obj, dest, inplace=False):
        """ Move all tensors in the object to a specified dest (device or dtype).
        The inplace=True option is available for dict."""
        if obj is None:
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.to(dest)
        elif isinstance(obj, Patches):
            return obj.patches.to(dest)
        elif isinstance(obj, tuple):
            return tuple([self._to(item, dest) for item in obj])
        elif isinstance(obj, list):
            return list([self._to(item, dest) for item in obj])
        elif isinstance(obj, dict):
            if inplace:
                for k, v in obj.items():
                    obj[k] = self._to(v, dest, inplace=True)
                return obj
            else:
                return {k: self._to(v, dest) for k, v in obj.items()}
        else:
            raise NotImplementedError(type(obj))

    def _convert_nodes(self, model, global_input):
        r"""
        Returns:
            nodesOP (list): List of operator nodes
            nodesIn (list): List of input nodes
            nodesOut (list): List of output nodes
            template (object): Template to specify the output format
        """
        global_input_cpu = self._to(global_input, 'cpu')
        model.train() if self.ori_training else model.eval()
        model.to('cpu')
        nodesOP, nodesIn, nodesOut, template = parse_module(model, global_input_cpu)
        model.to(self.device)
        for i in range(0, len(nodesIn)):
            if nodesIn[i].param is not None:
                nodesIn[i] = nodesIn[i]._replace(param=nodesIn[i].param.to(self.device))
        global_input_unpacked = unpack_inputs(global_input)

        # Convert input nodes and parameters.
        for i, n in enumerate(nodesIn):
            if n.input_index is not None:
                nodesIn[i] = nodesIn[i]._replace(bound_node=BoundInput(
                    ori_name=nodesIn[i].ori_name,
                    value=global_input_unpacked[nodesIn[i].input_index],
                    perturbation=nodesIn[i].perturbation,
                    input_index=n.input_index))
            else:
                bound_class = BoundParams if isinstance(nodesIn[i].param, nn.Parameter) else BoundBuffers
                nodesIn[i] = nodesIn[i]._replace(bound_node=bound_class(
                    ori_name=nodesIn[i].ori_name,
                    value=nodesIn[i].param,
                    perturbation=nodesIn[i].perturbation))

        unsupported_ops = []

        # Convert other operation nodes.
        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs, ori_names = self._get_node_input(nodesOP, nodesIn, nodesOP[n])
            try:
                if nodesOP[n].op in self.custom_ops:
                    op = self.custom_ops[nodesOP[n].op]
                elif nodesOP[n].op in bound_op_map:
                    op = bound_op_map[nodesOP[n].op]
                elif nodesOP[n].op.startswith('aten::ATen'):
                    op = eval('BoundATen{}'.format(attr['operator'].capitalize()))
                elif nodesOP[n].op.startswith('onnx::'):
                    op = eval('Bound{}'.format(nodesOP[n].op[6:]))
                else:
                    raise KeyError
            except (NameError, KeyError):
                unsupported_ops.append(nodesOP[n])
                logger.error('The node has an unsupported operation: {}'.format(nodesOP[n]))
                continue

            attr['device'] = self.device

            # FIXME generalize
            if nodesOP[n].op == 'onnx::BatchNormalization' or getattr(op, 'TRAINING_FLAG', False):
                # BatchNormalization node needs model.training flag to set running mean and vars
                # set training=False to avoid wrongly updating running mean/vars during bound wrapper
                nodesOP[n] = nodesOP[n]._replace(
                    bound_node=op(
                        attr, inputs, nodesOP[n].output_index, self.bound_opts, False))
            else:
                nodesOP[n] = nodesOP[n]._replace(
                    bound_node=op(attr, inputs, nodesOP[n].output_index, self.bound_opts))

        if unsupported_ops:
            logger.error('Unsupported operations:')
            for n in unsupported_ops:
                logger.error(f'Name: {n.op}, Attr: {n.attr}')
            raise NotImplementedError('There are unsupported operations')

        for node in nodesIn + nodesOP:
            node.bound_node.input_name = node.inputs
            node.bound_node.name = node.name

        nodes_dict = {}
        for node in nodesOP + nodesIn:
            nodes_dict[node.name] = node.bound_node
        nodesOP = [n.bound_node for n in nodesOP]
        nodesIn = [n.bound_node for n in nodesIn]
        nodesOut = [nodes_dict[n] for n in nodesOut]

        return nodesOP, nodesIn, nodesOut, template

    def _build_graph(self, nodesOP, nodesIn, nodesOut, template):
        # We were assuming that the original model had only one output node.
        # When there are multiple output nodes, this seems to be the first output element.
        # In this case, we are assuming that we aim to compute the bounds for the first
        # output element by default.
        self.final_name = nodesOut[0].name
        self.input_name, self.input_index, self.root_name = [], [], []
        self.output_name = [n.name for n in nodesOut]
        self.output_template = template
        for node in nodesIn:
            self.add_input_node(node, index=node.input_index)
        self.add_nodes(nodesOP)
        if self.conv_mode == 'patches':
            self.root_name = [node.name for node in nodesIn]

    # Make sure the nodes already have `name` and `input_name`
    def add_nodes(self, nodes):
        nodes = [(node if isinstance(node, Bound) else node.bound_node) for node in nodes]
        for node in nodes:
            self._modules[node.name] = node
            node.output_name = []
            if not hasattr(node, 'input_name'):
                node.input_name = []
            if isinstance(node.input_name, str):
                node.input_name = [node.input_name]
            if len(node.inputs) == 0:
                self.root_name.append(node.name)
        for node in nodes:
            for l_pre in node.inputs:
                l_pre.output_name.append(node.name)
        for node in nodes:
            if isinstance(node, BoundOptimizableActivation):
                self.optimizable_activations.append(node)
            if isinstance(node, BoundRelu):
                self.relus.append(node)

    def add_input_node(self, node, index=None):
        self.add_nodes([node])
        self.input_name.append(node.name)
        # default value for input_index
        if index == 'auto':
            index = max([0] + [(i + 1) for i in self.input_index if i is not None])
        self.input_index.append(index)

    def rename_nodes(self, nodesOP, nodesIn, rename_dict):
        def rename(node):
            node.name = rename_dict[node.name]
            node.input_name = [
                rename_dict[name] for name in node.input_name]
            return node
        for i in range(len(nodesOP)):
            nodesOP[i] = rename(nodesOP[i])
        for i in range(len(nodesIn)):
            nodesIn[i] = rename(nodesIn[i])

    def _split_complex(self, nodesOP, nodesIn):
        finished = True
        for n in range(len(nodesOP)):
            if hasattr(nodesOP[n], 'complex') and \
                    nodesOP[n].complex:
                finished = False
                _nodesOP, _nodesIn, _nodesOut, _template = self._convert_nodes(
                    nodesOP[n].model, nodesOP[n].input)
                # assuming each supported complex operation only has one output
                assert len(_nodesOut) == 1

                name_base = nodesOP[n].name + '/split'
                rename_dict = {}
                for node in _nodesOP + _nodesIn:
                    rename_dict[node.name] = name_base + node.name
                num_inputs = len(nodesOP[n].inputs)
                for i in range(num_inputs):
                    rename_dict[_nodesIn[i].name] = nodesOP[n].input_name[i]
                rename_dict[_nodesOP[-1].name] = nodesOP[n].name

                self.rename_nodes(_nodesOP, _nodesIn, rename_dict)

                output_name = _nodesOP[-1].name
                # Any input node of some node within the complex node should be
                # replaced with the corresponding input node of the complex node.
                for node in _nodesOP:
                    for i in range(len(node.inputs)):
                        if node.input_name[i] in nodesOP[n].input_name:
                            index = nodesOP[n].input_name.index(node.input_name[i])
                            node.inputs[i] = nodesOP[n].inputs[index]
                # For any output node of this complex node, modify its input node
                for node in nodesOP:
                    if output_name in node.input_name:
                        index = node.input_name.index(output_name)
                        node.inputs[index] = _nodesOP[-1]

                nodesOP = nodesOP[:n] + _nodesOP + nodesOP[(n + 1):]
                nodesIn = nodesIn + _nodesIn[num_inputs:]

                break

        return nodesOP, nodesIn, finished

    """build a dict with {ori_name: name, name: ori_name}"""
    def _get_node_name_map(self):
        self.node_name_map = {}
        for node in self._modules.values():
            if isinstance(node, BoundInput) or isinstance(node, BoundParams):
                for p in list(node.named_parameters()):
                    if node.ori_name not in self.node_name_map:
                        self.node_name_map[node.ori_name] = node.name + '.' + p[0]
                        self.node_name_map[node.name + '.' + p[0]] = node.ori_name
                for p in list(node.named_buffers()):
                    if node.ori_name not in self.node_name_map:
                        self.node_name_map[node.ori_name] = node.name + '.' + p[0]
                        self.node_name_map[node.name + '.' + p[0]] = node.ori_name

    # convert a Pytorch model to a model with bounds
    def _convert(self, model, global_input):
        if self.verbose:
            logger.info('Converting the model...')

        if not isinstance(global_input, tuple):
            global_input = (global_input,)
        self.num_global_inputs = len(global_input)

        nodesOP, nodesIn, nodesOut, template = self._convert_nodes(model, global_input)
        global_input = self._to(global_input, self.device)

        while True:
            self._build_graph(nodesOP, nodesIn, nodesOut, template)
            self.forward(*global_input)  # running means/vars changed
            nodesOP, nodesIn, finished = self._split_complex(nodesOP, nodesIn)
            if finished:
                break

        self._get_node_name_map()

        # load self.ori_state_dict again to avoid the running means/vars changed during forward()
        self.load_state_dict(self.ori_state_dict)
        if self.ori_training:
            model.load_state_dict(self.ori_state_dict)
        delattr(self, 'ori_state_dict')

        # The final node used in the last time calling `compute_bounds`
        self.last_final_node = None
        self.used_nodes = []

        if self.verbose:
            logger.info('Model converted to support bounds')

    def init_slope(self, x, share_slopes=False, method='backward',
                   c=None, bound_lower=True, bound_upper=True, final_node_name=None,
                   new_interval=None, activation_opt_params=None, skip_bound_compute=False):
        for node in self.optimizable_activations:
            # initialize the parameters
            node.opt_init()

        if (not skip_bound_compute or new_interval is None or
                activation_opt_params is None or
                not all([relu.name in activation_opt_params for relu in self.relus])):
            skipped = False
            # if new interval is None, then CROWN interval is not present
            # in this case, we still need to redo a CROWN pass to initialize lower/upper
            with torch.no_grad():
                l, u = self.compute_bounds(
                    x=x, C=c, method=method, bound_lower=bound_lower,
                    bound_upper=bound_upper, final_node_name=final_node_name,
                    new_interval=new_interval)
        else:
            # we skip, but we still would like to figure out the "used", "perturbed", "backward_from" of each note in the graph
            skipped = True
            # this set the "perturbed" property
            self._set_input(*x, new_interval=new_interval)

            final = self.final_node() if final_node_name is None else self[final_node_name]
            self._set_used_nodes(final)

            self.backward_from = dict([(node, [final]) for node in self._modules])

        final_node_name = final_node_name or self.final_name

        init_intermediate_bounds = {}
        for node in self.optimizable_activations:
            if not node.used or not node.perturbed:
                continue
            start_nodes = []
            if method in ['forward', 'forward+backward']:
                start_nodes.append(('_forward', 1, None))
            if method in ['backward', 'forward+backward']:
                start_nodes += self.get_alpha_crown_start_nodes(
                    node, c=c, share_slopes=share_slopes, final_node_name=final_node_name)
            if skipped:
                node.restore_optimized_params(activation_opt_params[node.name])
            else:
                node.init_opt_parameters(start_nodes)
            init_intermediate_bounds[node.inputs[0].name] = ([node.inputs[0].lower.detach(), node.inputs[0].upper.detach()])

        print("Optimizable variables initialized.")
        if skip_bound_compute:
            return init_intermediate_bounds
        else:
            return l, u, init_intermediate_bounds

    def get_optimized_bounds(
            self, x=None, aux=None, C=None, IBP=False, forward=False, method='backward',
            bound_lower=True, bound_upper=False, reuse_ibp=False, return_A=False,
            final_node_name=None, average_A=False, new_interval=None, reference_bounds=None,
            aux_reference_bounds=None, needed_A_dict=None, cutter=None, decision_thresh=None,
            epsilon_over_decision_thresh=1e-4):
        # optimize CROWN lower bound by alpha and beta
        opts = self.bound_opts['optimize_bound_args']
        iteration = opts['ob_iteration']; beta = opts['ob_beta']; alpha = opts['ob_alpha']
        opt_coeffs = opts['ob_opt_coeffs']; opt_bias = opts['ob_opt_bias']
        verbose = opts['ob_verbose']; opt_choice = opts['ob_optimizer']
        single_node_split = opts['ob_single_node_split']
        keep_best = opts['ob_keep_best']; disable_update_intermediate_layer_bounds = opts['ob_update_by_layer'];
        # "update_by_layer" is False when intermediate layer bounds are optimized in beta-CROWN. (FIXME: change the name).
        init = opts['ob_init']; lr = opts['ob_lr']; lr_beta = opts['ob_lr_beta'];
        lr_cut_beta = opts['ob_lr_cut_beta']
        lr_intermediate_beta = opts['ob_lr_intermediate_beta']
        lr_decay = opts['ob_lr_decay']; lr_coeffs = opts['ob_lr_coeffs']
        loss_reduction_func = opts['ob_loss_reduction_func']
        stop_criterion_func = opts['ob_stop_criterion_func']
        input_grad = opts['ob_input_grad']
        no_float64_last_iter = opts['ob_no_float64_last_iter']
        no_amp = opts['ob_no_amp']
        early_stop_patience = opts['ob_early_stop_patience']
        intermediate_beta_enabled = opts['ob_intermediate_beta']
        start_save_best = opts['ob_start_save_best']
        multi_spec_keep_func = opts['ob_multi_spec_keep_func']

        enable_opt_interm_bounds = self.bound_opts.get('enable_opt_interm_bounds', False)
        sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False)

        assert bound_lower != bound_upper, 'we can only optimize lower OR upper bound at one time'
        assert alpha or beta, "nothing to optimize, use compute bound instead!"

        if C is not None:
            self.final_shape = C.size()[:2]
            self.bound_opts.update({'final_shape': self.final_shape})
        if init:
            # TODO: this should set up aux_reference_bounds.
            self.init_slope(x, share_slopes=opts['ob_alpha_share_slopes'],
                            method=method, c=C, final_node_name=final_node_name)

        # Optimizable activations that are actually used and perturbed
        optimizable_activations = [n for n in self.optimizable_activations if n.used and n.perturbed]
        # Relu node that are actually used
        relus = [n for n in self.relus  if n.used and n.perturbed]

        alphas, betas, parameters = [], [], []
        dense_coeffs_mask = []

        if alpha:
            for node in optimizable_activations:
                alphas.extend(list(node.alpha.values()))
                node.opt_start()
            # Alpha has shape (2, output_shape, batch_dim, node_shape)
            parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
            # best_alpha is a dictionary of dictionary. Each key is the alpha variable for one relu layer, and each value is a dictionary contains all relu layers after that layer as keys.
            best_alphas = OrderedDict()
            for m in optimizable_activations:
                best_alphas[m.name] = {}
                for alpha_m in m.alpha:
                    best_alphas[m.name][alpha_m] = m.alpha[alpha_m].detach().clone()
                    # We will directly replace the dictionary for each relu layer after optimization, so the saved alpha might not have require_grad=True.
                    m.alpha[alpha_m].requires_grad_()

        if beta:
            if len(relus) != len(optimizable_activations):
                # raise NotImplementedError("Beta-CROWN for tanh models is not supported yet")
                pass  # FIXME: just bypass node without beta optimization supported
                # raise NotImplementedError("Beta-CROWN for tanh models is not supported yet")

            if single_node_split:
                for node in relus:
                    if enable_opt_interm_bounds and node.sparse_beta is not None:
                        for key in node.sparse_beta.keys():
                            if node.sparse_beta[key] is not None: betas.append(node.sparse_beta[key])
                    else:
                        if node.sparse_beta is not None: betas.append(node.sparse_beta)
            else:
                betas = self.beta_params + self.single_beta_params
                if opt_coeffs:
                    coeffs = [dense_coeffs["dense"] for dense_coeffs in self.split_dense_coeffs_params] + self.coeffs_params
                    dense_coeffs_mask = [dense_coeffs["mask"] for dense_coeffs in self.split_dense_coeffs_params]
                    parameters.append({'params': coeffs, 'lr': lr_coeffs})
                    best_coeffs = [coeff.detach().clone() for coeff in coeffs]
                if opt_bias:
                    biases = self.bias_params
                    parameters.append({'params': biases, 'lr': lr_coeffs})
                    best_biases = [bias.detach().clone() for bias in biases]

            # Beta has shape (batch, max_splits_per_layer)
            parameters.append({'params': betas, 'lr': lr_beta, 'batch_dim': 0})

            if self.cut_used:
                # also need to optimize cut betas
                parameters.append({'params': self.cut_beta_params, 'lr': lr_cut_beta, 'batch_dim': 0})
                betas = betas + self.cut_beta_params

            if enable_opt_interm_bounds and betas:
                best_betas = OrderedDict()
                for m in optimizable_activations:
                    best_betas[m.name] = {}
                    for beta_m in m.sparse_beta:
                        best_betas[m.name][beta_m] = m.sparse_beta[beta_m].detach().clone()
                if self.cut_used:
                    best_betas["cut"] = []
                    for general_betas in self.cut_beta_params:
                        best_betas["cut"].append(general_betas.detach().clone())
            else:
                best_betas = [b.detach().clone() for b in betas]

            if self.cut_used and getattr(cutter, 'opt', False):
                parameters.append(cutter.get_parameters())

        start = time.time()

        if decision_thresh is not None and isinstance(decision_thresh, torch.Tensor):
            if decision_thresh.dim() == 1:
                # add the spec dim to be aligned with compute_bounds return
                decision_thresh = decision_thresh.unsqueeze(-1)

        ### preprocessor-hint: private-section-start
        if beta and intermediate_beta_enabled:
            # The list of layer numbers for refinement, can be positive or negative. -1 means refine the intermediate layer bound before last relu layer.
            intermediate_refinement_layers = opts['ob_intermediate_refinement_layers']
            if len(intermediate_refinement_layers) == 0:
                # Nothing to refine. Switch back to beta-CROWN.
                intermediate_beta_enabled = False
                self.best_intermediate_betas = None
                # Do not optimize intermediate layer bounds.
                disable_update_intermediate_layer_bounds = True
            else:
                # Change negative layer number to positive ones.
                intermediate_refinement_layers = [layer if layer > 0 else layer + len(relus) for layer in
                                                  intermediate_refinement_layers]
                # This is the first layer to refine; we do not need the specs for all layers before it.
                first_layer_to_refine = relus[min(intermediate_refinement_layers)].name
                # Change layer number to layer name.
                intermediate_refinement_layers = [relus[layer].name for layer in intermediate_refinement_layers]
                print(f'Layers for refinement: {intermediate_refinement_layers}; there are {len(relus) - len(intermediate_refinement_layers)} layers NOT being refined.')
                # We only need to set some intermediate layer bounds.
                partial_new_interval = new_interval.copy() if new_interval is not None else None  # Shallow copy.
                # beta_constraint_specs is a disctionary that saves the coefficients for beta for each relu layer.
                # all_intermediate_betas A list of all optimizable parameters for intermediate betas. Will be passed to the optimizer.
                # For each neuron in each layer, we have M intermediate_beta variables where M is the number of constraints.
                # We only need to collect some A matrices for the split constraints, so we keep a dictionary needed_A_dict for it.
                # These A matrices are the split constraints propagated back to earlier layers.
                beta_constraint_specs, all_intermediate_betas, needed_A_dict = self._init_intermediate_beta(x, opt_coeffs, intermediate_refinement_layers, first_layer_to_refine, partial_new_interval)
                # Add all intermediate layer beta to parameters.
                parameters.append({'params': all_intermediate_betas, 'lr': lr_intermediate_beta})
        ### preprocessor-hint: private-section-end

        if opt_choice == "adam-autolr":
            opt = AdamElementLR(parameters, lr=lr)
        elif opt_choice == "adam":
            opt = optim.Adam(parameters, lr=lr)
        elif opt_choice == 'sgd':
            opt = optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(opt_choice)

        # Create a weight vector to scale learning rate.
        loss_weight = torch.ones(size=(x[0].size(0),), device=x[0].device)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, lr_decay)

        if verbose > 0 and intermediate_beta_enabled:
            self.print_optimized_beta(relus, intermediate_beta_enabled=True)

        # best_intermediate_bounds is linked to aux_reference_bounds!
        best_intermediate_bounds = {}
        if sparse_intermediate_bounds and aux_reference_bounds is None and reference_bounds is not None:
            aux_reference_bounds = {}
            for name, (lb, ub) in reference_bounds.items():
                aux_reference_bounds[name] = [lb.detach().clone(), ub.detach().clone()]
        if aux_reference_bounds is None:
            aux_reference_bounds = {}

        with torch.no_grad():
            pruning_in_iteration = False
            # for computing the positive domain ratio
            original_size = x[0].shape[0]
            preserve_mask = None

        # record the overhead due to extra operations from pruning-in-iteration
        pruning_time = 0.

        need_grad = True
        patience = 0
        for i in range(iteration):
            if cutter:
                # cuts may be optimized by cutter
                self.cut_module = cutter.cut_module

            intermediate_constr = None
            ### preprocessor-hint: private-section-start
            if beta and intermediate_beta_enabled:
                intermediate_constr = self._get_intermediate_beta_specs(x, aux, opt_coeffs, beta_constraint_specs, needed_A_dict, new_interval)
                reference_bounds = new_interval  # If we still optimize all intermediate neurons, we can use new_interval as reference bounds.
            ### preprocessor-hint: private-section-end

            if not disable_update_intermediate_layer_bounds:
                reference_bounds = new_interval  # If we still optimize all intermediate neurons, we can use new_interval as reference bounds.

            if i == iteration - 1:
                # No grad update needed for the last iteration
                need_grad = False

                if self.device == 'cuda' and torch.get_default_dtype() == torch.float32 and not no_float64_last_iter:
                    # we are using float64 only in the last iteration to help improve floating point error.
                    self.to(torch.float64)
                    C = C.to(torch.float64)
                    x = self._to(x, torch.float64)
                    # best_intermediate_bounds is linked to aux_reference_bounds!
                    # we only need call .to() for one of them
                    self._to(aux_reference_bounds, torch.float64, inplace=True)
                    new_interval = self._to(new_interval, torch.float64)

            # with torch.cuda.amp.autocast() if not no_amp and i != iteration - 1 else ExitStack():
            if True:  # TODO temporally disable amp
                # we will use last update preserve mask in caller functions to recover lA, l, u, etc to full batch size
                self.last_update_preserve_mask = preserve_mask
                with torch.no_grad() if not need_grad else ExitStack():
                    # ret is lb, ub or lb, ub, A_dict (if return_A is set to true)
                    ret = self.compute_bounds(
                        x, aux, C, method=method, IBP=IBP, forward=forward,
                        bound_lower=bound_lower, bound_upper=bound_upper, reuse_ibp=reuse_ibp,
                        return_A=return_A, final_node_name=final_node_name, average_A=average_A,
                        # If we set neuron bounds individually, or if we are optimizing intermediate layer bounds using beta, we do not set new_interval.
                        # When intermediate betas are used, we must set new_interval to None because we want to recompute all intermediate layer bounds.
                        new_interval=partial_new_interval if beta and intermediate_beta_enabled else new_interval if disable_update_intermediate_layer_bounds else None,
                        # This is the currently tightest interval, which will be used to pass split constraints when intermediate betas are used.
                        reference_bounds=reference_bounds,
                        # This is the interval used for checking for unstable neurons.
                        aux_reference_bounds=aux_reference_bounds if sparse_intermediate_bounds else None,
                        # These are intermediate layer beta variables and their corresponding A matrices and biases.
                        intermediate_constr=intermediate_constr, needed_A_dict=needed_A_dict,
                        update_mask=preserve_mask)

                ret_l, ret_u = ret[0], ret[1]

                if beta and opt_bias and not single_node_split:
                    ret_l = ret_l + self.beta_bias()  # add bias for the bias term of split constraint.
                    ret = (ret_l, ret_u) + ret[2:]  # add the A_dict if it exists in ret.

                if self.cut_used:
                    # betas[-1]: (2(0 lower, 1 upper), spec, batch, num_constrs)
                    if ret_l is not None:
                        if i % cutter.log_interval == 0 and len(self.cut_beta_params) > 0:
                            print(f"{i}, lb beta sum: {self.cut_beta_params[-1][0].sum() / ret_l.size(0)}, worst: {ret_l.min()}")
                    if ret_u is not None:
                        if i % 10 == 0 and len(self.cut_beta_params) > 0:
                            print(f"{i}, ub beta sum: {self.cut_beta_params[-1][1].sum() / ret_l.size(0)}, worst: {ret_u.min()}")

                if i == 0:
                    best_ret_l = torch.zeros_like(ret[0], device=x[0].device, dtype=x[0].dtype) - 1e8
                    best_ret = []
                    for ri in ret[:2]:
                        if ri is not None:
                            best_ret.append(ri.detach().clone())
                        else:
                            best_ret.append(None)
                    for node in optimizable_activations:
                        best_intermediate_bounds[node.name] = [node.inputs[0].lower.detach().clone(), node.inputs[0].upper.detach().clone()]
                        if sparse_intermediate_bounds:
                            # Always using the best bounds so far as the reference bounds.
                            aux_reference_bounds[node.inputs[0].name] = best_intermediate_bounds[node.name]

                l = ret_l
                if ret_l is not None and ret_l.shape[1] != 1:  # Reduction over the spec dimension.
                    l = loss_reduction_func(ret_l)
                u = ret_u
                if ret_u is not None and ret_u.shape[1] != 1:
                    u = loss_reduction_func(ret_u)

                # full_l, full_ret_l and full_u, full_ret_u is used for update the best
                full_ret_l, full_ret_u = ret_l, ret_u
                full_l, full_u = l, u
                full_ret = ret

                # positive domains may already be filtered out, so we use all domains - negative domains to compute
                if decision_thresh is not None:
                    if isinstance(decision_thresh, torch.Tensor) and decision_thresh.numel() > 1 and preserve_mask is not None:
                        if decision_thresh.shape[-1] == 1:
                            # single spec with pruned domains
                            positive_domain_num = original_size - (ret_l.view(-1) <= decision_thresh[preserve_mask].view(-1)).sum()
                        else:
                            # mutliple spec with pruned domains
                            positive_domain_num = original_size - multi_spec_keep_func(ret_l <= decision_thresh[preserve_mask]).sum()
                    else:
                        if ret_l.shape[-1] == 1:
                            # single spec
                            positive_domain_num = original_size - (ret_l.view(-1) <= decision_thresh.view(-1)).sum()
                        else:
                            # multiple spec
                            positive_domain_num = original_size - multi_spec_keep_func(ret_l <= decision_thresh).sum()

                else:
                    positive_domain_num = -1
                positive_domain_ratio = float(positive_domain_num) / float(original_size)
                # threshold is 10% by default
                next_iter_pruning_in_iteration = opts['ob_pruning_in_iteration'] and decision_thresh is not None and \
                                                 (positive_domain_ratio > opts['ob_pruning_in_iteration_threshold'])

                if pruning_in_iteration:
                    stime = time.time()
                    with torch.no_grad():
                        if isinstance(decision_thresh, torch.Tensor) and decision_thresh.numel() > 1:
                            if decision_thresh.shape[-1] == 1:
                                now_preserve_mask = (ret_l <= decision_thresh[preserve_mask]).view(-1).nonzero().view(-1)
                            else:
                                now_preserve_mask = multi_spec_keep_func(ret_l <= decision_thresh[preserve_mask]).nonzero().view(-1)
                        else:
                            if decision_thresh.shape[-1] == 1:
                                now_preserve_mask = (ret_l <= decision_thresh).view(-1).nonzero().view(-1)
                            else:
                                now_preserve_mask = multi_spec_keep_func(ret_l <= decision_thresh).nonzero().view(-1)

                        # recover l and u to full batch size so that later we can directly update using
                        # the full batch size l and u
                        if l is not None:
                            if isinstance(decision_thresh, torch.Tensor) and decision_thresh.numel() > 1:
                                full_ret_l = decision_thresh.clone().to(ret_l.device).type(ret_l.dtype) + epsilon_over_decision_thresh
                            else:
                                full_ret_l = torch.full((original_size,) + tuple(ret_l.shape[1:]),
                                                        fill_value=decision_thresh + epsilon_over_decision_thresh,
                                                        device=ret_l.device, dtype=ret_l.dtype)
                            full_ret_l[preserve_mask] = ret_l
                            if full_ret_l.shape[1] > 1:
                                full_l = loss_reduction_func(full_ret_l)
                            else:
                                full_l = full_ret_l
                        else:
                            full_ret_l = None
                        if u is not None:
                            if isinstance(decision_thresh, torch.Tensor) and decision_thresh.numel() > 1:
                                full_ret_u = decision_thresh.clone().to(ret_u.device).type(ret_u.dtype) + epsilon_over_decision_thresh
                            else:
                                full_ret_u = torch.full((original_size,) + tuple(ret_u.shape[1:]),
                                                        fill_value=decision_thresh + epsilon_over_decision_thresh,
                                                        device=ret_u.device, dtype=ret_u.dtype)
                            full_ret_u[preserve_mask] = ret_u
                            if full_ret_u.shape[1] > 1:
                                full_u = loss_reduction_func(full_ret_u)
                            else:
                                full_u = full_ret_u
                        else:
                            full_ret_u = full_u = None
                        full_ret = (full_ret_l, full_ret_u) + ret[2:]

                        if return_A:
                            raise Exception('Pruning in iteration optimization does not support return A yet. '
                                            'Please fix or discard this optimization by setting --disable_pruning_in_iteration '
                                            'or bab: pruning_in_iteration: false')

                        if C is not None and C.shape[0] == x[0].shape[0]:
                            # means C is also batch specific
                            C = C[now_preserve_mask]

                        x = list(x)
                        pre_prune_size = x[0].shape[0]
                        x[0].data = x[0][now_preserve_mask].data
                        if hasattr(x[0], 'ptb'):
                            if x[0].ptb.x_L is not None:
                                x[0].ptb.x_L = x[0].ptb.x_L[now_preserve_mask]
                            if x[0].ptb.x_U is not None:
                                x[0].ptb.x_U = x[0].ptb.x_U[now_preserve_mask]
                        x = tuple(x)

                        if beta and intermediate_beta_enabled:
                            # prune the partial_new_interval
                            interval_to_prune = partial_new_interval
                        elif disable_update_intermediate_layer_bounds:
                            interval_to_prune = new_interval
                        else:
                            interval_to_prune = None
                        if interval_to_prune is not None:
                            for k, v in interval_to_prune.items():
                                interm_interval_l, interm_interval_r = v[0], v[1]
                                if interm_interval_l.shape[0] == pre_prune_size:
                                    # the first dim is batch size and matches preserve mask
                                    interm_interval_l = interm_interval_l[now_preserve_mask]
                                if interm_interval_r.shape[0] == pre_prune_size:
                                    # the first dim is batch size and matches preserve mask
                                    interm_interval_r = interm_interval_r[now_preserve_mask]
                                interval_to_prune[k] = [interm_interval_l, interm_interval_r]

                        if aux_reference_bounds is not None:
                            for k in aux_reference_bounds:
                                aux_ref_l, aux_ref_r = aux_reference_bounds[k]
                                if aux_ref_l.shape[0] == pre_prune_size:
                                    # the first dim is batch size and matches the preserve mask
                                    aux_ref_l = aux_ref_l[now_preserve_mask]
                                if aux_ref_r.shape[0] == pre_prune_size:
                                    # the first dim is batch size and matches the preserve mask
                                    aux_ref_r = aux_ref_r[now_preserve_mask]
                                aux_reference_bounds[k] = [aux_ref_l, aux_ref_r]

                        # update the global mask here for possible next iteration
                        preserve_mask_next = preserve_mask[now_preserve_mask]

                        # print(f'DEBUG    # preserved domain = {torch.sum(preserve_mask_next).item()}')
                    ttime = time.time()
                    pruning_time += ttime - stime
                else:
                    pass

                ### preprocessor-hint: private-section-start
                if beta and intermediate_beta_enabled and Check_against_base_lp:
                    # NOTE stop_criterion here is not valid
                    full_total_loss = total_loss = loss_ = loss = self._modules[Check_against_base_lp_layer].upper[0,55]
                    stop_criterion = torch.zeros_like(full_ret_l, dtype=torch.bool)
                else:
                    ### preprocessor-hint: private-section-end
                    ### preprocessor-hint: private-replacement \t\t\tif True:
                    loss_ = l if bound_lower else -u
                    full_loss_ = full_l if bound_lower else -full_u
                    stop_criterion = stop_criterion_func(full_ret_l) if bound_lower else stop_criterion_func(-full_ret_u)
                    if type(stop_criterion) != bool and stop_criterion.numel() > 1 and pruning_in_iteration:
                        stop_criterion = stop_criterion[preserve_mask]
                    total_loss = -1 * loss_
                    full_total_loss = -1 * full_loss_
                    if type(stop_criterion) == bool:
                        loss = total_loss.sum() * (not stop_criterion)
                    else:
                        loss = (total_loss * stop_criterion.logical_not()).sum()

            stop_criterion_final = isinstance(stop_criterion, torch.Tensor) and stop_criterion.all()

            if i == iteration - 1 and not no_amp:
                best_ret = list(best_ret)
                if best_ret[0] is not None:
                    best_ret[0] = best_ret[0].to(torch.get_default_dtype())
                if best_ret[1] is not None:
                    best_ret[1] = best_ret[1].to(torch.get_default_dtype())

            if i == iteration - 1 and self.device == 'cuda' and torch.get_default_dtype() == torch.float32 and not no_float64_last_iter:
                # print('Switch back to default precision from {}!!!'.format(ret[0].dtype))
                total_loss = total_loss.to(torch.get_default_dtype())
                full_total_loss = full_total_loss.to(torch.get_default_dtype())
                # switch back to default precision
                self.to(torch.get_default_dtype())
                x[0].to(torch.get_default_dtype())
                full_ret = list(full_ret)
                if isinstance(ret[0], torch.Tensor):
                    # round down lower bound
                    full_ret[0] = double2float(full_ret[0], "down")
                if isinstance(ret[1], torch.Tensor):
                    # round up upper bound
                    full_ret[1] = double2float(full_ret[1], "up")
                for _k, _v in best_intermediate_bounds.items():
                    _v[0] = double2float(_v[0], "down")
                    _v[1] = double2float(_v[1], "up")
                    best_intermediate_bounds[_k] = _v
                if return_A:
                    full_ret[2] = self._to(full_ret[2], torch.get_default_dtype())

            with torch.no_grad():
                # for lb and ub, we update them in every iteration since updating them is cheap
                need_update = False
                if keep_best:
                    idx_mask = (full_ret_l > best_ret_l).any(dim=1).view(-1)
                    if idx_mask.any():
                        need_update = True
                        # we only pick up the results improved in a batch
                        idx = idx_mask.nonzero(as_tuple=True)[0]
                        if not no_amp:
                            total_loss = total_loss.to(best_ret_l)
                            # TODO convert A to fp32
                        best_ret_l[idx] = torch.maximum(full_ret_l[idx], best_ret_l[idx])
                        if full_ret[0] is not None:
                            best_ret[0][idx] = torch.maximum(full_ret[0][idx], best_ret[0][idx])
                        if full_ret[1] is not None:
                            best_ret[1][idx] = torch.minimum(full_ret[1][idx], best_ret[1][idx])
                        if return_A:
                            # TODO A should also be updated by idx
                            best_ret = (best_ret[0], best_ret[1], full_ret[2])

                # Save variables if this is the best iteration.
                # To save computational cost, we only check keep_best at the first (in case divergence) and second half iterations
                # or before early stop by either stop_criterion or early_stop_patience reached
                # if i < 1 or i > iteration / 2 or stop_criterion_final or patience == early_stop_patience:
                if i < 1 or i > (iteration / int(start_save_best)) or stop_criterion_final or patience == early_stop_patience:
                    if need_update:
                        patience = 0

                        # for update propose, we condition the idx to update only on domains preserved
                        if pruning_in_iteration:
                            # local sparse index of preserved samples where idx = true
                            local_idx = idx_mask[preserve_mask].nonzero().view(-1)
                            # idx is global sparse index of preserved samples where idx = true
                            new_idx = torch.zeros_like(idx_mask, dtype=torch.bool, device=idx.device)
                            new_idx[preserve_mask] = idx_mask[preserve_mask]
                            idx = new_idx.nonzero().view(-1)

                        # FIXME skip these when intermediate bounds are fixed
                        for ii, node in enumerate(optimizable_activations):
                            if not pruning_in_iteration:
                                # Update best intermediate layer bounds only when they are optimized. If they are already fixed in new_interval, then do nothing.
                                if new_interval is None or node.inputs[0].name not in new_interval or not disable_update_intermediate_layer_bounds:
                                    best_intermediate_bounds[node.name][0][idx] = torch.max(
                                        best_intermediate_bounds[node.name][0][idx], node.inputs[0].lower[idx])
                                    best_intermediate_bounds[node.name][1][idx] = torch.min(
                                        best_intermediate_bounds[node.name][1][idx], node.inputs[0].upper[idx])
                            else:
                                # Update best intermediate layer bounds only when they are optimized. If they are already fixed in new_interval, then do nothing.
                                if new_interval is None or node.inputs[0].name not in new_interval or not disable_update_intermediate_layer_bounds:
                                    best_intermediate_bounds[node.name][0][idx] = torch.max(
                                        best_intermediate_bounds[node.name][0][idx], node.inputs[0].lower[local_idx])
                                    best_intermediate_bounds[node.name][1][idx] = torch.min(
                                        best_intermediate_bounds[node.name][1][idx], node.inputs[0].upper[local_idx])

                            if alpha:
                                # Each alpha has shape (2, output_shape, batch, *shape) for ReLU.
                                # For other activation function this can be different.
                                for alpha_m in node.alpha:
                                    if node.alpha_batch_dim == 2:
                                        best_alphas[node.name][alpha_m][:,:,idx] = node.alpha[alpha_m][:,:,idx]
                                    elif node.alpha_batch_dim == 3:
                                        best_alphas[node.name][alpha_m][:,:,:,idx] = node.alpha[alpha_m][:,:,:,idx]
                                    else:
                                        raise ValueError(f"alpha_batch_dim={node.alpha_batch_dim} must be set to 2 or 3 in BoundOptimizableActivation")
                            # if beta and single_node_split:
                            # best_betas[ii][idx] = betas[ii][idx].detach().clone()

                        if beta and single_node_split:
                            if enable_opt_interm_bounds and betas:
                                for ii, node in enumerate(optimizable_activations):
                                    for key in node.sparse_beta.keys():
                                        best_betas[node.name][key] = node.sparse_beta[key].detach().clone()
                                if self.cut_used:
                                    for gbidx, general_betas in enumerate(self.cut_beta_params):
                                        best_betas["cut"][gbidx] = general_betas.detach().clone()
                            else:
                                if self.cut_used:
                                    regular_beta_length = len(betas) - len(self.cut_beta_params)
                                    for beta_idx in range(regular_beta_length):
                                        # regular beta crown betas
                                        best_betas[beta_idx][idx] = betas[beta_idx][idx]
                                    for cut_beta_idx in range(len(self.cut_beta_params)):
                                        # general cut beta crown general_betas
                                        best_betas[regular_beta_length + cut_beta_idx][:, :, idx, :] = \
                                            betas[regular_beta_length + cut_beta_idx][:, :, idx, :]
                                else:
                                    for beta_idx in range(len(betas)):
                                        # regular beta crown betas
                                        best_betas[beta_idx][idx] = betas[beta_idx][idx]
                                    # print([(bb[idx].sum().item(), b[idx].sum().item()) for bb, b in zip(best_betas, betas)])

                        if not single_node_split and beta:
                            for ii, b in enumerate(betas):
                                best_betas[ii][idx] = b[idx]

                            if opt_coeffs:
                                best_coeffs = [co.detach().clone() for co in coeffs]  # TODO: idx-wise
                            if opt_bias:
                                best_biases = [bias.detach().clone() for bias in biases]  # TODO: idx-wise

                        ### preprocessor-hint: private-section-start
                        if beta and intermediate_beta_enabled:
                            self.save_best_intermediate_betas(relus, idx)
                        ### preprocessor-hint: private-section-end
                    else:
                        patience += 1

            if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
                print(f"****** iter [{i}]",
                      f"loss: {loss.item()}, lr: {opt.param_groups[0]['lr']}")

            if stop_criterion_final:
                print(f"\nall verified at {i}th iter")
                break

            if patience > early_stop_patience:
                print(f'Early stop at {i}th iter due to {early_stop_patience} iterations no improvement!')
                break

            current_lr = [param_group['lr'] for param_group in opt.param_groups]

            opt.zero_grad(set_to_none=True)

            if input_grad and x[0].ptb.x_L.grad is not None:
                # for input guided split
                x[0].ptb.x_L.grad = x[0].ptb.x_U.grad = None

            if verbose > 0:
                print(f"*** iter [{i}]\n", f"loss: {loss.item()}", total_loss.squeeze().detach().cpu().numpy(), "lr: ", current_lr)
                if beta:
                    self.print_optimized_beta(relus, intermediate_beta_enabled)
                    if opt_coeffs:
                        for co in coeffs:
                            print(f'coeff sum: {co.abs().sum():.5g}')
                if beta and i == 0 and verbose > 0:
                    breakpoint()

            if i != iteration - 1:
                # we do not need to update parameters in the last step since the best result already obtained
                loss.backward()
                # All intermediate variables are not needed at this point.
                self._clear_and_set_new(None)
                if opt_choice == "adam-autolr":
                    opt.step(lr_scale=[loss_weight, loss_weight])
                else:
                    opt.step()

            if beta:
                # Clipping to >=0.
                for b in betas:
                    b.data = (b >= 0) * b.data
                ### preprocessor-hint: private-section-start
                if intermediate_beta_enabled:
                    for b in all_intermediate_betas:
                        b.data = torch.clamp(b.data, min=0)
                ### preprocessor-hint: private-section-end
                for dmi in range(len(dense_coeffs_mask)):
                    # apply dense mask to the dense split coeffs matrix
                    coeffs[dmi].data = dense_coeffs_mask[dmi].float() * coeffs[dmi].data

            # Possibly update cuts if they are parameterized and optimzied
            if self.cut_used:
                cutter.update_cuts()

            if alpha:
                for m in optimizable_activations:
                    m.clip_alpha_()

            if beta and opt_choice == "adam-autolr" and i > iteration * 0.2:
                # If loss has become worse for some element, reset those to current best.
                # FIXME Unlikely to work. `worse_idx` is not defined.
                self.beta_reset_worse_idx(
                    betas, best_alphas, best_betas, relus, alpha=alpha, beta=beta,
                    single_node_split=single_node_split,
                    enable_opt_interm_bound=enable_opt_interm_bounds)

            scheduler.step()

            if pruning_in_iteration:
                preserve_mask = preserve_mask_next
            if not pruning_in_iteration and next_iter_pruning_in_iteration:
                # init preserve_mask etc
                preserve_mask = torch.arange(0, x[0].shape[0], device=x[0].device, dtype=torch.long)
                pruning_in_iteration = True

        if pruning_in_iteration:
            # overwrite pruned cells in best_ret by threshold + eps
            if return_A:
                fin_l, fin_u, fin_A = best_ret
            else:
                fin_l, fin_u = best_ret; fin_A = None
            if fin_l is not None:
                new_fin_l = full_ret_l
                new_fin_l[preserve_mask] = fin_l[preserve_mask]
                fin_l = new_fin_l
            if fin_u is not None:
                new_fin_u = full_ret_u
                new_fin_u[preserve_mask] = fin_u[preserve_mask]
                fin_u = new_fin_u
            if return_A:
                best_ret = (fin_l, fin_u, fin_A)
            else:
                best_ret = (fin_l, fin_u)

        # if beta and intermediate_beta_enabled and verbose > 0:
        if verbose > 0:
            breakpoint()

        if keep_best:
            def update_best(dest, src):
                for item_dest, item_src in zip(dest, src):
                    if enable_opt_interm_bounds:
                        for key in item_dest.keys():
                            item_dest[key].data = item_src[key].data
                    else:
                        item_dest.data = item_src.data
            # Set all variables to their saved best values.
            with torch.no_grad():
                for idx, node in enumerate(optimizable_activations):
                    if alpha:
                        # Assigns a new dictionary.
                        node.alpha = best_alphas[node.name]
                    # Update best intermediate layer bounds only when they are optimized. If they are already fixed in new_interval, then do nothing.
                    node.inputs[0].lower.data = best_intermediate_bounds[node.name][0].data
                    node.inputs[0].upper.data = best_intermediate_bounds[node.name][1].data
                    if beta:
                        if single_node_split and hasattr(node, 'sparse_beta') and node.sparse_beta is not None:
                            if enable_opt_interm_bounds:
                                for key in node.sparse_beta.keys():
                                    node.sparse_beta[key].copy_(best_betas[node.name][key])
                            else:
                                node.sparse_beta.copy_(best_betas[idx])
                        else:
                            update_best(betas, best_betas)
                            if opt_coeffs:
                                update_best(coeffs, best_coeffs)
                            if opt_bias:
                                update_best(biases, best_biases)
                if self.cut_used:
                    regular_beta_length = len(betas) - len(self.cut_beta_params)
                    for ii in range(len(self.cut_beta_params)):
                        self.cut_beta_params[ii].data = best_betas[regular_beta_length + ii].data


        if new_interval is not None and not disable_update_intermediate_layer_bounds:
            for l in self._modules.values():
                if l.name in new_interval.keys() and hasattr(l, "lower"):
                    # l.interval = tuple(new_interval[l.name][:2])
                    l.lower = torch.max(l.lower, new_interval[l.name][0])
                    l.upper = torch.min(l.upper, new_interval[l.name][1])
                    infeasible_neurons = l.lower > l.upper
                    if infeasible_neurons.any():
                        print('infeasible!!!!!!!!!!!!!!', infeasible_neurons.sum().item(), infeasible_neurons.nonzero()[:, 0])

        if self.cut_used and beta:
            print("first 10 best general betas:", best_betas[-1].view(2,-1)[0][:10], "sum:", best_betas[-1][0].sum().item())
        print("best_l after optimization:", best_ret_l.sum().item(), "with beta sum per layer:", [p.sum().item() for p in betas])
        # np.save('solve_slope.npy', np.array(record))
        print('alpha/beta optimization time:', time.time() - start)

        for node in optimizable_activations:
            node.opt_end()

        # update pruning ratio
        if opts['ob_pruning_in_iteration'] and decision_thresh is not None:
            stime = time.time()
            with torch.no_grad():
                if full_l.numel() > 0:
                    if isinstance(decision_thresh, torch.Tensor):
                        if decision_thresh.shape[-1] == 1:
                            neg_domain_num = torch.sum(full_ret_l.view(-1) <= decision_thresh.view(-1)).item()
                        else:
                            neg_domain_num = torch.sum(multi_spec_keep_func(full_ret_l <= decision_thresh)).item()
                    else:
                        if full_l.shape[-1] == 1:
                            neg_domain_num = torch.sum(full_ret_l.view(-1) <= decision_thresh).item()
                        else:
                            neg_domain_num = torch.sum(multi_spec_keep_func(full_ret_l <= decision_thresh)).item()
                    now_pruning_ratio = 1.0 - float(neg_domain_num) / float(full_l.shape[0])
                    print('pruning_in_iteration open status:', pruning_in_iteration)
                    print('ratio of positive domain =', full_l.shape[0] - neg_domain_num, '/',
                          full_l.numel(), '=', now_pruning_ratio)
            pruning_time += time.time() - stime
            print('pruning-in-iteration extra time:', pruning_time)

        return best_ret

    def check_prior_bounds(self, node):
        if node.prior_checked or not (node.used and node.perturbed):
            return

        for n in node.inputs:
            self.check_prior_bounds(n)

        if getattr(node, 'nonlinear', False):
            for n in node.inputs:
                self.compute_intermediate_bounds(n, prior_checked=True)

        for i in getattr(node, 'requires_input_bounds', []):
            self.compute_intermediate_bounds(node.inputs[i], prior_checked=True)

        node.prior_checked = True

    def compute_intermediate_bounds(self, node, prior_checked=False):
        if getattr(node, 'lower', None) is not None:
            return

        logger.debug(f'Getting the bounds of {node}')

        if not prior_checked:
            self.check_prior_bounds(node)

        if not node.perturbed:
            fv = self.get_forward_value(node)
            node.interval = node.lower, node.upper = fv, fv
            return

        # FIXME check that weight perturbation is not affected
        #      (from_input=True should be set for weights)
        if not node.from_input and hasattr(node, 'forward_value'):
            node.lower = node.upper = self.get_forward_value(node)
            return

        reference_bounds = self.reference_bounds

        if self.use_forward:
            node.lower, node.upper = self.forward_general(node=node, concretize=True)
        else:
            #FIXME need clean up

            # assign concretized bound for ReLU layer to save computational cost
            # FIXME: Put ReLU after reshape will cause problem!
            if self.check_IBP_intermediate(node):
                # Intermediate bounds for some operators are directly
                # computed from their input nodes by IBP (such as BoundRelu, BoundNeg)
                logger.debug(f'IBP propagation for intermediate bounds on {node}')
            elif isinstance(node, BoundReshape) and \
                    hasattr(node.inputs[0], 'lower') and \
                    hasattr(node.inputs[1], 'value'):
                # TODO merge this with `check_IBP_intermediate`
                # Node for input value.
                val_input = node.inputs[0]
                # Node for input parameter (e.g., shape, permute)
                arg_input = node.inputs[1]
                node.lower = node.forward(val_input.lower, arg_input.value)
                node.upper = node.forward(val_input.upper, arg_input.value)
                node.interval = (node.lower, node.upper)
            else:
                # For the first linear layer, IBP can give the same tightness as CROWN.
                if self.check_IBP_first_linear(node):
                    return

                sparse_intermediate_bounds_with_ibp = self.bound_opts.get('sparse_intermediate_bounds_with_ibp', True)
                # Sparse intermediate bounds can be enabled if aux_reference_bounds are given. (this is enabled for ReLU only, and not for other activations.)
                sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False) and isinstance(self[node.output_name[0]], BoundRelu)

                ref_intermediate_lb, ref_intermediate_ub = None, None
                if sparse_intermediate_bounds:
                    if node.name not in self.aux_reference_bounds:
                        # If aux_reference_bounds are not available, we can use IBP to compute these bounds.
                        if sparse_intermediate_bounds_with_ibp:
                            with torch.no_grad():
                                # Get IBP bounds for this layer; we set delete_bounds_after_use=True which does not save extra intermediate bound tensors.
                                ref_intermediate_lb, ref_intermediate_ub = self.IBP_general(
                                    node=node, delete_bounds_after_use=True)
                        else:
                            sparse_intermediate_bounds = False
                    else:
                        ref_intermediate_lb, ref_intermediate_ub = self.aux_reference_bounds[node.name]
                ### preprocessor-hint: private-section-start
                if Check_against_base_lp:
                    sparse_intermediate_bounds = False
                ### preprocessor-hint: private-section-end

                newC, reduced_dim, unstable_idx, unstable_size = self.get_sparse_C(
                    node, sparse_intermediate_bounds, ref_intermediate_lb, ref_intermediate_ub)

                if unstable_idx is None or unstable_size > 0:
                    if self.return_A:
                        node.lower, node.upper, _ = self.backward_general(
                            C=newC, node=node, unstable_idx=unstable_idx,
                            unstable_size=unstable_size)
                    else:
                        # Compute backward bounds only when there are unstable neurons, or when we don't know which neurons are unstable.
                        node.lower, node.upper = self.backward_general(
                            C=newC, node=node, unstable_idx=unstable_idx,
                            unstable_size=unstable_size)

                if reduced_dim:
                    self.restore_sparse_bounds(
                        node, unstable_idx, unstable_size,
                        ref_intermediate_lb, ref_intermediate_ub)

                # node.lower and node.upper (intermediate bounds) are computed in the above function.
                # If we have bound references, we set them here to always obtain a better set of bounds.
                if node.name in reference_bounds:
                    # Initially, the reference bound and the computed bound can be exactly the same when intermediate layer beta is 0. This will prevent gradients flow. So we need a small guard here.
                    ### preprocessor-hint: private-section-start
                    if Check_against_base_lp:
                        if node.name != Check_against_base_lp_layer:
                            # For LP checking, fix all other intermediate layer bounds.
                            node.lower = reference_bounds[node.name][0]
                            node.upper = reference_bounds[node.name][1]
                    elif self.intermediate_constr is not None:
                        ### preprocessor-hint: private-section-end
                        ### preprocessor-hint: private-replacement \t\t\t\t\t\tif intermediate_constr is not None:
                        # Intermediate layer beta is used.
                        # Note that we cannot just take the reference bounds if they are better - this makes alphas have zero gradients.
                        node.lower = torch.max((0.9 * reference_bounds[node.name][0] + 0.1 * node.lower), node.lower)
                        node.upper = torch.min((0.9 * reference_bounds[node.name][1] + 0.1 * node.upper), node.upper)
                        # Additionally, if the reference bounds say a neuron is stable, we always keep it. (FIXME: this is for ReLU only).
                        lower_stable = reference_bounds[node.name][0] >= 0.
                        node.lower[lower_stable] = reference_bounds[node.name][0][lower_stable]
                        upper_stable = reference_bounds[node.name][1] <= 0.
                        node.upper[upper_stable] = reference_bounds[node.name][1][upper_stable]
                    else:
                        # Set the intermediate layer bounds using reference bounds, always choosing the tighter one.
                        node.lower = torch.max(reference_bounds[node.name][0], node.lower).detach() - node.lower.detach() + node.lower
                        node.upper = node.upper - (node.upper.detach() - torch.min(reference_bounds[node.name][1], node.upper).detach())
                    # Otherwise, we only use reference bounds to check which neurons are unstable.

                node.interval = (node.lower, node.upper)  # FIXME (12/28): we should be consistent, and only use node.interval, do not use node.lower or node.upper!

    def compute_bounds(self, x=None, aux=None, C=None, method='backward', IBP=False, forward=False,
                       bound_lower=True, bound_upper=True, reuse_ibp=False, reuse_alpha=False,
                       return_A=False, needed_A_dict=None, final_node_name=None, average_A=False, new_interval=None,
                       reference_bounds=None, intermediate_constr=None, alpha_idx=None,
                       aux_reference_bounds=None, need_A_only=False,
                       cutter=None, decision_thresh=None,
                       update_mask=None):
        r"""Main function for computing bounds.

        Args:
            x (tuple or None): Input to the model. If it is None, the input from the last
            `forward` or `compute_bounds` call is reused. Otherwise: the number of elements in the tuple should be
            equal to the number of input nodes in the model, and each element in the tuple
            corresponds to the value for each input node respectively. It should look similar
            as the `global_input` argument when used for creating a `BoundedModule`.

            aux (object, optional): Auxliary information that can be passed to `Perturbation`
            classes for initializing and concretizing bounds, e.g., additional information
            for supporting synonym word subsitution perturbaiton.

            C (Tensor): The specification matrix that can map the output of the model with an
            additional linear layer. This is usually used for maping the logits output of the
            model to classification margins.

            method (str): The main method for bound computation. Choices:
                * `IBP`: purely use Interval Bound Propagation (IBP) bounds.
                * `CROWN-IBP`: use IBP to compute intermediate bounds, but use CROWN (backward mode LiRPA) to compute the bounds of the final node.
                * `CROWN`: purely use CROWN to compute bounds for intermediate nodes and the final node.
                * `Forward`: purely use forward mode LiRPA to compute the bounds.
                * `Forward+Backward`: use forward mode LiRPA to compute bounds for intermediate nodes, but further use CROWN to compute bounds for the final node.
                * `CROWN-Optimized` or `alpha-CROWN`: use CROWN, and also optimize the linear relaxation parameters for activations.
                * `forward-optimized`: use forward bounds with optimized linear relaxation.

            IBP (bool, optional): If `True`, use IBP to compute the bounds of intermediate nodes.
            It can be automatically set according to `method`.

            forward (bool, optional): If `True`, use the forward mode bound propagation to compute the bounds
            of intermediate nodes. It can be automatically set according to `method`.

            bound_lower (bool, default `True`): If `True`, the lower bounds of the output needs to be computed.

            bound_upper (bool, default `True`): If `True`, the upper bounds of the output needs to be computed.

            reuse_ibp (bool, optional): If `True` and `method` is None, reuse the previously saved IBP bounds.

            reuse_alpha (bool, optional): If `True`, reuse previously saved alpha values when they are not being optimized.

            decision_thresh: If it is not None, it should be a float, and from CROWN-optimized mode, we will use this decision_thresh
                to dynamically optimize those domains that <= the threshold

            update_mask: None or  bool tensor([batch_size])
                If set to a tensor, only update the alpha and beta of selected element (with element=1)

        Returns:
            bound (tuple): a tuple of computed lower bound and upper bound respectively.
        """
        logger.debug(f'Compute bounds with {method}')
        
        if needed_A_dict is None: needed_A_dict = {}
        if not bound_lower and not bound_upper:
            raise ValueError('At least one of bound_lower and bound_upper must be True')

        # Several shortcuts.
        method = method.lower() if method is not None else method
        if method == 'ibp':
            # Pure IBP bounds.
            method = None
            IBP = True
        elif method in ['ibp+backward', 'ibp+crown', 'crown-ibp']:
            method = 'backward'
            IBP = True
        elif method == 'crown':
            method = 'backward'
        elif method == 'forward':
            forward = True
        elif method == 'forward+backward' or method == 'forward+crown':
            method = 'backward'
            forward = True
        elif method in ['crown-optimized', 'alpha-crown', 'forward-optimized']:
            # The lower and upper bound need two separate rounds of optimization.
            if method == 'forward-optimized':
                method = 'forward'
            else:
                method = 'backward'
            if bound_lower:
                ret1 = self.get_optimized_bounds(
                    x=x, C=C, method=method, new_interval=new_interval,
                    reference_bounds=reference_bounds, bound_lower=bound_lower,
                    bound_upper=False, return_A=return_A,
                    aux_reference_bounds=aux_reference_bounds,
                    needed_A_dict=needed_A_dict, final_node_name=final_node_name,
                    cutter=cutter, decision_thresh=decision_thresh)
            if bound_upper:
                ret2 = self.get_optimized_bounds(
                    x=x, C=C, method=method, new_interval=new_interval,
                    reference_bounds=reference_bounds, bound_lower=False,
                    bound_upper=bound_upper, return_A=return_A,
                    aux_reference_bounds=aux_reference_bounds,
                    needed_A_dict=needed_A_dict, final_node_name=final_node_name,
                    cutter=cutter, decision_thresh=decision_thresh)
            if bound_lower and bound_upper:
                if return_A:
                    # Needs to merge the A dictionary.
                    lA_dict = ret1[2]
                    uA_dict = ret2[2]
                    merged_A = {}
                    for node_name in lA_dict:
                        merged_A[node_name] = {
                            "lA": lA_dict[node_name]["lA"],
                            "uA": uA_dict[node_name]["uA"],
                            "lbias": lA_dict[node_name]["lbias"],
                            "ubias": uA_dict[node_name]["ubias"],
                        }
                    return ret1[0], ret2[1], merged_A
                else:
                    return ret1[0], ret2[1]
            elif bound_lower:
                return ret1  # ret1[1] is None.
            elif bound_upper:
                return ret2  # ret2[0] is None.

        if reference_bounds is None:
            reference_bounds = {}
        if aux_reference_bounds is None:
            aux_reference_bounds = {}

        # If y in self.backward_node_pairs[x], then node y is visited when
        # doing backward bound propagation starting from node x.
        self.backward_from = dict([(node, []) for node in self._modules])

        if not bound_lower and not bound_upper:
            raise ValueError('At least one of bound_lower and bound_upper in compute_bounds should be True')
        A_dict = {} if return_A else None

        if x is not None:
            self._set_input(*x, new_interval=new_interval)

        if IBP and method is None and reuse_ibp:
            # directly return the previously saved ibp bounds
            return self.ibp_lower, self.ibp_upper
        root = [self[name] for name in self.root_name]
        batch_size = root[0].value.shape[0]
        dim_in = 0

        for i in range(len(root)):
            value = root[i].forward()
            if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
                root[i].linear, root[i].center, root[i].aux = \
                    root[i].perturbation.init(value, aux=aux, forward=forward)
                # This input/parameter has perturbation. Create an interval object.
                if self.ibp_relative:
                    root[i].interval = Interval(
                        None, None, root[i].linear.nominal, root[i].linear.lower_offset, root[i].linear.upper_offset)
                else:
                    root[i].interval = Interval(
                        root[i].linear.lower, root[i].linear.upper, ptb=root[i].perturbation)
                if forward:
                    root[i].dim = root[i].linear.lw.shape[1]
                    dim_in += root[i].dim
            else:
                if self.ibp_relative:
                    root[i].interval = Interval(
                        None, None, value, torch.zeros_like(value), torch.zeros_like(value))
                else:
                    # This inpute/parameter does not has perturbation.
                    # Use plain tuple defaulting to Linf perturbation.
                    root[i].interval = (value, value)
                    root[i].forward_value = root[i].forward_value = root[i].value = root[i].lower = root[i].upper = value

            if self.ibp_relative:
                root[i].lower, root[i].upper = root[i].interval.lower, root[i].interval.upper
            else:
                root[i].lower, root[i].upper = root[i].interval

        if forward:
            self.init_forward(root, dim_in)

        final = self.final_node() if final_node_name is None else self[final_node_name]
        logger.debug(f'Final node {final.__class__.__name__}({final.name})')

        if IBP:
            res = self.IBP_general(node=final, C=C)
            if self.ibp_relative:
                self.ibp_lower, self.ibp_upper = res.lower, res.upper
            else:
                self.ibp_lower, self.ibp_upper = res

        if method is None:
            return self.ibp_lower, self.ibp_upper

        if C is None:
            # C is an identity matrix by default
            if final.output_shape is None:
                raise ValueError('C is not provided while node {} has no default shape'.format(final.shape))
            dim_output = int(prod(final.output_shape[1:]))
            # TODO: use an eyeC object here.
            C = torch.eye(dim_output, device=self.device).expand(batch_size, dim_output, dim_output)

        # Reuse previously saved alpha values, even if they are not optimized now
        if reuse_alpha:
            for node in self.optimizable_activations:
                node.opt_reuse()
        else:
            for node in self.optimizable_activations:
                node.opt_no_reuse()

        # Inject update mask inside the activations
        if update_mask is None:
            for node in self.optimizable_activations:
                node.clean_alpha_beta_update_mask()
        else:
            for node in self.optimizable_activations:
                node.set_alpha_beta_update_mask(update_mask)

        for n in self._modules.values():
            n.prior_checked = False # Check whether all prior intermediate bounds already exist
            # check whether weights are perturbed and set nonlinear for the BoundMatMul operation
            if isinstance(n, (BoundLinear, BoundConv, BoundBatchNormalization)):
                n.nonlinear = False
                for node in n.inputs[1:]:
                    if hasattr(node, 'perturbation'):
                        if node.perturbation is not None:
                            n.nonlinear = True
            if isinstance(i, BoundRelu):
                for node in i.inputs:
                    if isinstance(node, BoundConv):
                        node.relu_followed = True # whether this Conv is followed by a ReLU

        # BFS to find out whether each node is used given the current final node
        self._set_used_nodes(final)

        # FIXME clean
        self.use_forward = forward
        self.root = root
        self.batch_size = batch_size
        self.dim_in = dim_in
        self.return_A = return_A
        self.A_dict = A_dict
        self.needed_A_dict = needed_A_dict
        self.intermediate_constr = intermediate_constr
        self.reference_bounds = reference_bounds
        self.aux_reference_bounds = aux_reference_bounds
        self.final_node_name = final.name

        self.check_prior_bounds(final)

        if method == 'backward':
            # This is for the final output bound. No need to pass in intermediate layer beta constraints.
            ret = self.backward_general(
                C=C, node=final, bound_lower=bound_lower, bound_upper=bound_upper,
                average_A=average_A, need_A_only=need_A_only, unstable_idx=alpha_idx, update_mask=update_mask)
            # FIXME when C is specified, lower and upper should not be saved to 
            # final.lower and final.upper, because they are not the bounds for the node.
            final.lower, final.upper = ret[0], ret[1]
            return ret
        elif method == 'forward':
            return self.forward_general(C=C, node=final, concretize=True)
        else:
            raise NotImplementedError

    def _set_used_nodes(self, final):
        if final.name != self.last_final_node:
            self.last_final_node = final.name
            self.used_nodes = []
            for i in self._modules.values():
                i.used = False
            final.used = True
            queue = deque([final])
            while len(queue) > 0:
                n = queue.popleft()
                self.used_nodes.append(n)
                for n_pre in n.inputs:
                    if not n_pre.used:
                        n_pre.used = True
                        queue.append(n_pre)

    from .interval_bound import IBP_general, _IBP_loss_fusion, check_IBP_intermediate, check_IBP_first_linear
    from .forward_bound import forward_general, forward_general_dynamic, init_forward
    from .backward_bound import (
        backward_general, get_sparse_C, check_optimized_variable_sparsity, restore_sparse_bounds,
        get_alpha_crown_start_nodes, get_unstable_locations,
        batched_backward)
    from .beta_crown import (
        beta_bias, save_best_intermediate_betas,
        beta_reset_worse_idx, print_optimized_beta)

    """Add perturbation to an intermediate node and it is treated as an independent
    node in bound computation."""

    def add_intermediate_perturbation(self, node, perturbation):
        node.perturbation = perturbation
        node.perturbed = True
        # NOTE This change is currently inreversible
        if not node.name in self.root_name:
            self.root_name.append(node.name)

    ### preprocessor-hint: private-section-start
    from .intermediate_refinement import _init_intermediate_beta
    from .intermediate_refinement import _get_intermediate_beta_specs
    from .intermediate_refinement import _get_intermediate_beta_bounds
    ### preprocessor-hint: private-section-end

    from .solver_module import build_solver_module, _build_solver_input, _build_solver_general
