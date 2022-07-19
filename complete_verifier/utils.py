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

import sys
import os
import gzip
import collections
from functools import partial
from ast import literal_eval
import importlib
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import onnx2pytorch
import onnx
import onnxruntime as ort
import arguments
import warnings
from attack_pgd import attack_pgd, build_conditions, auto_attack, attack_with_general_specs

# Import all model architectures.
from model_defs import *
from read_vnnlib import read_vnnlib
from onnx_opt import compress_onnx

# FIXME move to attack_pgd.py
def attack(model_ori, x, data_lb, data_ub, vnnlib, verified_status, verified_success, crown_filtered_labels=None):
    # assert len(vnnlib) == 1, vnnlib_path
    initialization = 'uniform'
    GAMA_loss = False
    if 'auto_attack' not in arguments.Config["attack"]["attack_mode"]:
        if "diversed" in arguments.Config["attack"]["attack_mode"]:
            initialization = "osi"
        if "GAMA" in arguments.Config["attack"]["attack_mode"]:
            GAMA_loss = True

        # In this file, we only consider batch_size == 1
        assert x.shape[0] == 1

        list_target_label_arrays, data_min_repeat, data_max_repeat = process_vnn_lib_attack(vnnlib, x)

        # data_min/max_repeat: [batch_size, spec_num, *input_shape]
        # list_target_label_arrays: [batch_size, spec_num, C_mat, rhs_mat]

        if (crown_filtered_labels is not None):
            assert len(list_target_label_arrays) == 1 # only support batch_size=1 cases
            list_target_label_arrays_new = [[]]
            for i in range(len(list_target_label_arrays[0])):
                C_mat, rhs = list_target_label_arrays[0][i]
                label = np.where(C_mat[0] == -1)
                if (crown_filtered_labels[label[0][0]] == True): continue
                list_target_label_arrays_new[0].append(list_target_label_arrays[0][i])
            list_target_label_arrays = list_target_label_arrays_new
            print(f"Remain {len(list_target_label_arrays[0])} labels need to be attacked.")

        attack_ret, attack_images, attack_margin = attack_with_general_specs(model_ori, x, data_min_repeat[:,:len(list_target_label_arrays[0]),...],
                                                                             data_max_repeat[:,:len(list_target_label_arrays[0]),...], list_target_label_arrays, initialization=initialization, GAMA_loss=GAMA_loss)

    else:
        raise NotImplementedError('auto attack does not support general cases')
        attack_ret, attack_images, attack_margin = auto_attack(model_ori, x, data_min=data_min, data_max=data_max, vnnlib=vnnlib)


    if attack_ret:
        # Attack success.
        attack_output = model_ori(attack_images.view(-1, *x.shape[1:]))
        if arguments.Config["general"]["save_adv_example"]:
            save_cex(attack_images, attack_output, x, vnnlib, arguments.Config["attack"]["cex_path"], data_max_repeat, data_min_repeat)
        verified_status = "unsafe-pgd"
        verified_success = True
        return verified_status, verified_success, attack_images

    return verified_status, verified_success, None


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w


def reshape_bounds(lower_bounds, upper_bounds, y, global_lb=None):
    with torch.no_grad():
        last_lower_bounds = torch.zeros(size=(1, lower_bounds[-1].size(1)+1), dtype=lower_bounds[-1].dtype, device=lower_bounds[-1].device)
        last_upper_bounds = torch.zeros(size=(1, upper_bounds[-1].size(1)+1), dtype=upper_bounds[-1].dtype, device=upper_bounds[-1].device)
        last_lower_bounds[:, :y] = lower_bounds[-1][:, :y]
        last_lower_bounds[:, y+1:] = lower_bounds[-1][:, y:]
        last_upper_bounds[:, :y] = upper_bounds[-1][:, :y]
        last_upper_bounds[:, y+1:] = upper_bounds[-1][:, y:]
        lower_bounds[-1] = last_lower_bounds
        upper_bounds[-1] = last_upper_bounds
        if global_lb is not None:
            last_global_lb = torch.zeros(size=(1, global_lb.size(1)+1), dtype=global_lb.dtype, device=global_lb.device)
            last_global_lb[:, :y] = global_lb[:, :y]
            last_global_lb[:, y+1:] = global_lb[:, y:]
            global_lb = last_global_lb
    return lower_bounds, upper_bounds, global_lb


def convert_mlp_model(model, dummy_input):
    model.eval()
    feature_maps = {}

    def get_feature_map(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()

        return hook

    def conv_to_dense(conv, inputs):
        b, n, w, h = inputs.shape
        kernel = conv.weight
        bias = conv.bias
        I = torch.eye(n * w * h).view(n * w * h, n, w, h)
        W = F.conv2d(I, kernel, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        # input_flat = inputs.view(b, -1)
        b1, n1, w1, h1 = W.shape
        # out = torch.matmul(input_flat, W.view(b1, -1)).view(b, n1, w1, h1)
        new_bias = bias.view(1, n1, 1, 1).repeat(1, 1, w1, h1)

        dense_w = W.view(b1, -1).transpose(1, 0)
        dense_bias = new_bias.view(-1)

        new_m = nn.Linear(in_features=dense_w.shape[1], out_features=dense_w.shape[0], bias=m.bias is not None)
        new_m.weight.data.copy_(dense_w)
        new_m.bias.data.copy_(dense_bias)

        return new_m

    new_modules = []
    modules = list(model.named_modules())[1:]
    for mi, (name, m) in enumerate(modules):

        if mi+1 < len(modules) and isinstance(modules[mi+1][-1], nn.Conv2d):
            m.register_forward_hook(get_feature_map(name))
            model(dummy_input)
            pre_conv_input = feature_maps[name]
        elif mi == 0 and isinstance(m, nn.Conv2d):
            pre_conv_input = dummy_input

        if isinstance(m, nn.Linear):
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.data)
            new_m.bias.data.copy_(m.bias)
            new_modules.append(new_m)
        elif isinstance(m, nn.ReLU):
            new_modules.append(nn.ReLU())
        elif isinstance(m, nn.Flatten):
            pass
            # will flatten at the first layer
            # new_modules.append(nn.Flatten())
        elif isinstance(m, nn.Conv2d):
            new_modules.append(conv_to_dense(m, pre_conv_input))
        else:
            print(m, 'not support in convert_mlp_model')
            raise NotImplementedError

    #  add flatten at the beginning
    new_modules.insert(0, nn.Flatten())
    seq_model = nn.Sequential(*new_modules)

    return seq_model

def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively (https://stackoverflow.com/a/3233356)."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_pgd_acc(model, X, labels, eps, data_min, data_max, batch_size):
    start = arguments.Config["data"]["start"]
    total = arguments.Config["data"]["end"]
    clean_correct = 0
    robust_correct = 0
    model = model.to(device=arguments.Config["general"]["device"])
    X = X.to(device=arguments.Config["general"]["device"])
    labels = labels.to(device=arguments.Config["general"]["device"])
    if isinstance(data_min, torch.Tensor):
        data_min = data_min.to(device=arguments.Config["general"]["device"])
    if isinstance(data_max, torch.Tensor):
        data_max = data_max.to(device=arguments.Config["general"]["device"])
    if isinstance(eps, torch.Tensor):
        eps = eps.to(device=arguments.Config["general"]["device"])
    if arguments.Config["attack"]["pgd_alpha"] == 'auto':
        alpha = eps.mean() / 4 if isinstance(eps, torch.Tensor) else eps / 4
    else:
        alpha = float(arguments.Config["attack"]["pgd_alpha"])
    while start < total:
        end = min(start + batch_size, total)
        batch_X = X[start:end]
        batch_labels = labels[start:end]
        if arguments.Config["specification"]["type"] == "lp":
            # Linf norm only so far.
            data_ub = torch.min(batch_X + eps, data_max)
            data_lb = torch.max(batch_X - eps, data_min)
        else:
            # Per-example, per-element lower and upper bounds.
            data_ub = data_max[start:end]
            data_lb = data_min[start:end]
        clean_output = model(batch_X)

        best_deltas, last_deltas = attack_pgd(model, X=batch_X, y=batch_labels, epsilon=float("inf"), alpha=alpha,
                num_classes=arguments.Config["data"]["num_outputs"],
                attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"],
                upper_limit=data_ub, lower_limit=data_lb, multi_targeted=True, lr_decay=arguments.Config["attack"]["pgd_lr_decay"],
                target=None, early_stop=arguments.Config["attack"]["pgd_early_stop"])
        attack_images = torch.max(torch.min(batch_X + best_deltas, data_ub), data_lb)
        attack_output = model(attack_images)
        clean_labels = clean_output.argmax(1)
        attack_labels = attack_output.argmax(1)
        batch_clean_correct = (clean_labels == batch_labels).sum().item()
        batch_robust_correct = (attack_labels == batch_labels).sum().item()
        if start == 0:
            print("Clean prediction for first a few examples:")
            print(clean_output[:10].detach().cpu().numpy())
            print("PGD prediction for first a few examples:")
            print(attack_output[:10].detach().cpu().numpy())
        print(f'batch start {start}, batch size {end - start}, clean correct {batch_clean_correct}, robust correct {batch_robust_correct}')
        clean_correct += batch_clean_correct
        robust_correct += batch_robust_correct
        start += batch_size
        del clean_output, best_deltas, last_deltas, attack_images, attack_output
    print(f'data start {arguments.Config["data"]["start"]} end {total}, clean correct {clean_correct}, robust correct {robust_correct}')
    return clean_correct, robust_correct


def get_test_acc(model, input_shape=None, X=None, labels=None, is_channel_last=False, batch_size=256):
    device = arguments.Config["general"]["device"]
    if X is None and labels is None:
        # Load MNIST or CIFAR, used for quickly debugging.
        database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        mean = torch.tensor(arguments.Config["data"]["mean"])
        std = torch.tensor(arguments.Config["data"]["std"])
        normalize = transforms.Normalize(mean=mean, std=std)
        if input_shape == (3, 32, 32):
            testset = torchvision.datasets.CIFAR10(root=database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        elif input_shape == (1, 28, 28):
            testset = torchvision.datasets.MNIST(root=database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        else:
            raise RuntimeError("Unable to determine dataset for test accuracy.")
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        testloader = [(X, labels)]
    total = 0
    correct = 0
    if device != 'cpu':
        model = model.to(device)
    print_first_batch = True
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device != 'cpu':
                images = images.to(device)
                labels = labels.to(device)
            if is_channel_last:
                images = images.permute(0,2,3,1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if print_first_batch:
                print_first_batch = False
                for i in range(min(outputs.size(0), 10)):
                    print(f"Image {i} norm {images[i].abs().sum().item()} label {labels[i].item()} correct {labels[i].item() == outputs[i].argmax().item()}\nprediction {outputs[i].cpu().numpy()}")
    print(f'correct {correct} of {total}')



def unzip_and_optimize_onnx(path, onnx_optimization_flags='none'):
    if (onnx_optimization_flags == 'none'):
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f"Onnx optimization with flag {onnx_optimization_flags}")
        npath = path + ".optimized"
        if (os.path.exists(npath)):
            print(f"Found existed optimized onnx model at {npath}")
            return onnx.load(npath)
        else:
            print(f"Generate optimized onnx model to {npath}")
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)



def inference_onnx(path, *inputs):
    # print(inputs)
    sess = ort.InferenceSession(unzip_and_optimize_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res


#TODO Use `load_model_onnx_new` to replace it.
@torch.no_grad()
def load_model_onnx(path, input_shape, compute_test_acc=False, force_convert=False, force_return=False,
                    quirks=None):
    onnx_optimization_flags = arguments.Config["model"]["onnx_optimization_flags"]
    if arguments.Config["model"]["cache_onnx_conversion"]:
        path_cache = f'{path}.cache'
        if os.path.exists(path_cache):
            print(f'Loading converted model from {path_cache}')
            return torch.load(path_cache)
    quirks = {} if quirks is None else quirks
    if arguments.Config["model"]["onnx_quirks"]:
        try:
            config_quirks = literal_eval(arguments.Config["model"]["onnx_quirks"])
        except ValueError as e:
            print(f'ERROR: onnx_quirks {arguments.Config["model"]["onnx_quirks"]} cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)
    print(f'Loading onnx {path} wih quirks {quirks}')

    is_channel_last = False

    # pip install onnx2pytorch
    onnx_model = unzip_and_optimize_onnx(path, onnx_optimization_flags)

    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    # FIXME do not use this
    input_shape = tuple(input_shape)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=quirks)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    dummy = torch.randn([1, *input_shape])
    # pytorch_model = torch.nn.Sequential(*list(pytorch_model.modules())[1:])

    conversion_check_result = True
    try:
        # check conversion correctness
        # FIXME dtype of dummy may not match the onnx model, which can cause runtime error
        conversion_check_result = np.allclose(pytorch_model(dummy).numpy(), inference_onnx(path, dummy.numpy())[0], 1e-4, 1e-5)
    except:
        warnings.warn(f'Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; traceback.print_exc()
        print('*****************************************\n')
    if not conversion_check_result:
        print('\n**************************')
        print('Model might not be converted correctly. Please check onnx conversion carefully.')
        print('**************************\n')

    #FIXME don't use pytorch_model.modules()
    #Generally no need to convert to a sequential model

    if force_return:
        if arguments.Config["model"]["cache_onnx_conversion"]:
            torch.save((pytorch_model, is_channel_last), path_cache)
        return pytorch_model, is_channel_last

    if force_convert:
        new_modules = []
        modules = list(pytorch_model.modules())[1:]
        for mi, m in enumerate(modules):
            if isinstance(m, torch.nn.Linear):
                new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
                new_m.weight.data.copy_(m.weight.data)
                new_m.bias.data.copy_(m.bias)
                new_modules.append(new_m)
            elif isinstance(m, torch.nn.ReLU):
                new_modules.append(torch.nn.ReLU())
            elif isinstance(m, onnx2pytorch.operations.flatten.Flatten):
                new_modules.append(torch.nn.Flatten())
            elif isinstance(m, onnx2pytorch.operations.shape.Shape):
                print('convert x.view(x.size(0), -1) to Flatten() layer')
                if isinstance(modules[mi+2], onnx2pytorch.operations.gather.Gather) and isinstance(modules[mi+3], onnx2pytorch.operations.unsqueeze.Unsqueeze) and \
                        isinstance(modules[mi+1], onnx2pytorch.operations.constant.Constant) and isinstance(modules[mi+4], onnx2pytorch.operations.reshape.Reshape):
                    new_modules.append(torch.nn.Flatten())
                    modules.pop(mi)
                    modules.pop(mi)
                    modules.pop(mi)
                    modules.pop(mi)
            else:
                print(m)
                raise NotImplementedError

        seq_model = nn.Sequential(*new_modules)

        if arguments.Config["model"]["cache_onnx_conversion"]:
            torch.save((seq_model, is_channel_last), path_cache)
        return seq_model, is_channel_last

    # Check model input shape.
    if onnx_shape != input_shape:
        # Change channel location.
        onnx_shape_ = onnx_shape[2:] + onnx_shape[:2]
        if onnx_shape_ == input_shape:
            is_channel_last = True
        else:
            print(f"Unexpected input shape in onnx: {onnx_shape}, given {input_shape}")

    # Fixup converted ONNX model. For ResNet we directly return; for other models, we convert them to a Sequential model.
    # We also need to handle NCHW and NHWC formats here.
    conv_c, conv_h, conv_w = input_shape
    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    need_permute = False
    for mi, m in enumerate(modules):
        if isinstance(m, onnx2pytorch.operations.add.Add):
            # ResNet model. No need to convert to sequential.
            return pytorch_model, is_channel_last
        if isinstance(m, torch.nn.Conv2d):
            # Infer the output size of conv.
            conv_h, conv_w = conv_output_shape((conv_h, conv_w), m.kernel_size, m.stride, m.padding)
            conv_c = m.weight.size(0)
        if isinstance(m, onnx2pytorch.operations.reshape.Reshape):
            # Replace reshape with flatten.
            new_modules.append(nn.Flatten())
            # May need to permute the next linear layer if the model was in NHWC format.
            need_permute = True and is_channel_last
        elif isinstance(m, torch.nn.Linear) and need_permute:
            # The original model is in NHWC format and we now have NCHW format, so the dense layer's weight must be adjusted.
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.view(m.weight.size(0), conv_h, conv_w, conv_c).permute(0, 3, 1, 2).contiguous().view(m.weight.size(0), -1))
            new_m.bias.data.copy_(m.bias)
            need_permute = False
            new_modules.append(new_m)
        elif isinstance(m, torch.nn.ReLU) and mi == (len(modules)-1):
            # not add relu if last layer is relu
            pass
        else:
            new_modules.append(m)

    seq_model = nn.Sequential(*new_modules)
    # TODO hard code temporally for cifar2020
    if 'convBigRELU__PGD' in path:
        bn = nn.BatchNorm2d(3, eps=0.0).eval()
        bn.running_mean.data = seq_model[0].constant.flatten().detach()
        bn.running_var.data = seq_model[2].constant.flatten().detach().pow(2)
        seq_model = nn.Sequential(bn, *seq_model[4:])  # remove sub and div

    if compute_test_acc:
        get_test_acc(seq_model, input_shape)

    if arguments.Config["model"]["cache_onnx_conversion"]:
        torch.save((seq_model, is_channel_last), path_cache)

    return seq_model, is_channel_last


def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """
    if 'onnx' not in arguments.Config['model']['name']:
        # You can customize this function to load your own model based on model name.
        model_ori = eval(arguments.Config['model']['name'])()
        model_ori.eval()
        print(model_ori)

    if not weights_loaded:
        return model_ori

    if arguments.Config["model"]["path"] is not None:
        if 'onnx' in arguments.Config["model"]["path"]:
            input_shape = None
            if "MNIST" in arguments.Config["data"]["dataset"]:
                input_shape = (1,28,28)
            elif "CIFAR" in arguments.Config["data"]["dataset"]:
                input_shape = (1,32,32)
            else:
                raise NotImplementedError()
            model_ori, is_channel_last = load_model_onnx(arguments.Config["model"]["path"], input_shape=input_shape)
        else:
            sd = torch.load(arguments.Config["model"]["path"], map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            model_ori.load_state_dict(sd)
    else:
        print("Warning: pretrained model path is not given!")

    return model_ori


########################################
# Preprocess and load the datasets
########################################
def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Proprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
    upper_limit, lower_limit = 1., 0.
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs


def load_cifar_sample_data(normalized=True, MODEL="a_mix"):
    """
    Load sampled cifar data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    if normalized:
        X = preprocess_cifar(X)
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    if normalized:
        print("Sampled data loaded. Data already preprocessed!")
    else:
        print("Sampled data loaded. Data not preprocessed yet!")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_mnist_sample_data(MODEL="mnist_a_adv"):
    """
    Load sampled mnist data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_dataset():
    """
    Load regular datasets such as MNIST and CIFAR.
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    normalize = transforms.Normalize(mean=arguments.Config["data"]["mean"], std=arguments.Config["data"]["std"])
    if arguments.Config["data"]["dataset"] == 'MNIST':
        loader = datasets.MNIST
    elif arguments.Config["data"]["dataset"] == 'CIFAR':
        loader = datasets.CIFAR10
    elif arguments.Config["data"]["dataset"] == 'CIFAR100':
        loader = datasets.CIFAR100
    else:
        raise ValueError("Dataset {} not supported.".format(arguments.Config["data"]["dataset"]))
    test_data = loader(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor(arguments.Config["data"]["mean"])
    test_data.std = torch.tensor(arguments.Config["data"]["std"])
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    return test_data, data_max, data_min


def load_sampled_dataset():
    """
    Load sampled data and define the robustness region
    """
    if arguments.Config["data"]["dataset"] == "CIFAR_SAMPLE":
        X, labels, runnerup = load_cifar_sample_data(normalized=True, MODEL=arguments.Config['model']['name'])
        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)
        eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)
    elif arguments.Config["data"]["dataset"] == "MNIST_SAMPLE":
        X, labels, runnerup = load_mnist_sample_data(MODEL=arguments.Config['model']['name'])
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)
        eps_temp = 0.3
        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
    return X, labels, data_max, data_min, eps_temp, runnerup


def load_sdp_dataset(eps_temp=None):
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sdp')
    if arguments.Config["data"]["dataset"] == "CIFAR_SDP":
        X = np.load(os.path.join(database_path, "cifar/X_sdp.npy"))
        X = preprocess_cifar(X)
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "cifar/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)

        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    elif arguments.Config["data"]["dataset"] == "MNIST_SDP":
        X = np.load(os.path.join(database_path, "mnist/X_sdp.npy"))
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "mnist/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = torch.tensor(0.3)

        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)

        print("############################")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    else:
        exit("sdp dataset not supported!")

    return X, y, data_max, data_min, eps_temp, runnerup


def load_generic_dataset(eps_temp=None):
    """Load MNIST/CIFAR test set with normalization."""
    print("Trying generic MNIST/CIFAR data loader.")
    test_data, data_max, data_min = load_dataset()
    if eps_temp is None:
        raise ValueError('You must specify an epsilon')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    runnerup = None
    # Rescale epsilon.
    eps_temp = torch.reshape(eps_temp / torch.tensor(arguments.Config["data"]["std"], dtype=torch.get_default_dtype()), (1, -1, 1, 1))

    return X, labels, data_max, data_min, eps_temp, runnerup


def load_eran_dataset(eps_temp=None):
    """
    Load sampled data and define the robustness region
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/eran')

    if arguments.Config["data"]["dataset"] == "CIFAR_ERAN":
        X = np.load(os.path.join(database_path, "cifar_eran/X_eran.npy"))
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, -1, 1, 1).astype(np.float32)
        std = np.array([0.2023, 0.1994, 0.201]).reshape(1, -1, 1, 1).astype(np.float32)
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "cifar_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 2. / 255.

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_ERAN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))
        mean = 0.1307
        std = 0.3081
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_ERAN_UN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_MADRY_UN":
        X = np.load(os.path.join(database_path, "mnist_madry/X.npy")).reshape(-1, 1, 28, 28)
        labels = np.load(os.path.join(database_path, "mnist_madry/y.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    else:
        raise(f'Unsupported dataset {arguments.Config["data"]["dataset"]}')

    return X, labels, data_max, data_min, eps_temp, runnerup


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        spec = importlib.util.spec_from_file_location("customized", def_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(def_file)
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def load_verification_dataset(eps_before_normalization):
    if arguments.Config["data"]["dataset"].startswith("Customized("):
        # FIXME (01/10/22): fully document customized data loader.
        # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
        # X is the data matrix in (batch, ...).
        # labels are the groud truth labels, a tensor of integers.
        # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
        # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
        # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
        # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
        # Target label is the targeted attack label; can be set to None.
        data_config = eval(arguments.Config["data"]["dataset"])(eps=eps_before_normalization)
        if len(data_config) == 5:
            X, labels, data_max, data_min, eps_new = data_config
            runnerup, target_label = None, None
        elif len(data_config) == 6:
            X, labels, data_max, data_min, eps_new, runnerup = data_config
            target_label = None
        elif len(data_config) == 7:
            X, labels, data_max, data_min, eps_new, runnerup, target_label = data_config
        else:
            print("Data config types not correct!")
            exit()
        assert X.size(0) == labels.size(0), "batch size of X and labels should be the same!"
        assert (data_max - data_min).min()>=0, "data_max should always larger or equal to data_min!"
        return X, labels, runnerup, data_max, data_min, eps_new, target_label
    target_label = None
    # Add your customized dataset here.
    if arguments.Config["data"]["pkl_path"] is not None:
        # FIXME (01/10/22): "pkl_path" should not exist in public code!
        # for oval20 base, wide, deep or other datasets saved in .pkl file, we load the pkl file here.
        assert arguments.Config["specification"]["epsilon"] is None, 'will use epsilon saved in .pkl file'
        gt_results = pd.read_pickle(arguments.Config["data"]["pkl_path"])
        test_data, data_max, data_min = load_dataset()
        X, labels = zip(*test_data)
        X = torch.stack(X, dim=0)
        labels = torch.tensor(labels)
        runnerup = None
        idx = gt_results["Idx"].to_list()
        X, labels = X[idx], labels[idx]
        target_label = gt_results['prop'].to_list()
        eps_new = gt_results['Eps'].to_list()
        print('Overwrite epsilon that saved in .pkl file, they should be after normalized!')
        eps_new = [torch.reshape(torch.tensor(i, dtype=torch.get_default_dtype()), (1, -1, 1, 1)) for i in eps_new]
        data_config = (X, labels, data_max, data_min, eps_new, runnerup, target_label)
    # Some special model loaders.
    elif "ERAN" in arguments.Config["data"]["dataset"] or "MADRY" in arguments.Config["data"]["dataset"]:
        data_config = load_eran_dataset(eps_temp=eps_before_normalization)
    elif "SDP" in arguments.Config["data"]["dataset"]:
        data_config = load_sdp_dataset(eps_temp=eps_before_normalization)
    elif "SAMPLE" in arguments.Config["data"]["dataset"]:
        # Sampled datapoints (a small subset of MNIST/CIFAR), only for reproducing some paper results.
        data_config = load_sampled_dataset()
    elif "CIFAR" in arguments.Config["data"]["dataset"] or "MNIST" in arguments.Config["data"]["dataset"]:
        # general MNIST and CIFAR dataset with mean/std defined in config file.
        data_config = load_generic_dataset(eps_temp=eps_before_normalization)
    else:
        exit("Dataset not supported in this file! Please customize load_verification_dataset() function in utils.py.")

    if len(data_config) == 5:
        (X, labels, data_max, data_min, eps_new) = data_config
        runnerup = None
    elif len(data_config) == 6:
        (X, labels, data_max, data_min, eps_new, runnerup) = data_config
    elif len(data_config) == 7:
        (X, labels, data_max, data_min, eps_new, runnerup, target_label) = data_config

    if arguments.Config["specification"]["norm"] != np.inf:
        assert arguments.Config["data"]["std"].count(arguments.Config["data"]["std"][0]) == len(
            arguments.Config["data"]["std"]), print('For non-Linf norm, we only support 1d eps.')
        arguments.Config["data"]["std"] = arguments.Config["data"]["std"][0]
        eps_new = eps_new[0, 0, 0, 0]  # only support eps as a scalar for non-Linf norm

    # FIXME (01/10/22): we should have a common interface for dataloader.
    return X, labels, runnerup, data_max, data_min, eps_new, target_label


def convert_test_model(model_ori):
    # NOTE: It looks like `in_features` and `out_features` are in the wrong order
    # after converting the onnx model to pytorch model.
    # Swap them below.
    modules = []
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            layer = nn.Linear(m.in_features, m.out_features)  # Fix a bug in onnx converter for test models.
            layer.weight.data = m.weight.data.to(torch.float)
            layer.bias.data = m.bias.data.to(torch.float) if m.bias is not None else torch.zeros_like(layer.bias.data)
            modules.append(layer)
            # pdb.set_trace()
        else:
            modules.append(m)
    model_ori = nn.Sequential(*modules)

    return model_ori


def convert_nn4sys_model(model_ori):
    model_ori = nn.Sequential(*list(model_ori._modules.values()))
    # Split the model into v1 and v2 models to resolve numerical issues
    modules_v1 = []
    modules_v2 = []
    stage = 1
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            if m.weight.abs().max() > 1e9:
                stage = 2 if len(modules_v2) == 0 else 3
                continue
        else:
            continue
        if stage == 1:
            modules_v1 += [m, nn.ReLU(inplace=True)]
        elif stage == 2:
            dim = modules_v1[-2].out_features - 1
            lin = nn.Linear(m.in_features - dim, m.out_features - dim)
            lin.weight.data = m.weight[:lin.out_features, :lin.in_features]
            lin.weight = lin.weight.to(dtype=torch.float64)
            lin.bias.data = m.bias[:lin.out_features]
            lin.bias = lin.bias.to(dtype=torch.float64)
            modules_v2 += [lin, nn.ReLU(inplace=True)]
    x = torch.tensor([[119740.8]], dtype=torch.float64)
    modules_v1 = modules_v1[:-1]
    model_v1 = nn.Sequential(*modules_v1)
    y = model_v1(x)
    dim = y.size(-1) - 1
    modules_v2 = modules_v2[:-1]
    linear_ident = nn.Linear(1, dim, bias=False)
    linear_ident.weight.data = torch.ones_like(linear_ident.weight, dtype=torch.float64)
    modules_v2.insert(0, linear_ident)
    model_v2 = nn.Sequential(*modules_v2)
    y[:, :-2] *= (y[:, 1:-1] <= 0).int()
    select = (y[:, :-1] > 0).int()
    y2 = model_v2(x)
    y2 = y2[:] * select
    res = y2.sum(dim=-1, keepdim=True)
    res_ref = model_ori(x)
    print(res.item(), res_ref.item())

    model_ori = (model_v1, model_v2, model_ori)   
    return model_ori


class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean)/self.std)


def save_cex(adv_example, adv_output, x, vnnlib, res_path, data_max, data_min):
    list_target_label_arrays, _, _ = process_vnn_lib_attack(vnnlib, x)
    C_mat, rhs_mat, cond_mat, same_number_const = build_conditions(x, list_target_label_arrays)

    # [num_example, num_restarts, num_spec, output_dim] -> 
    # [num_example, num_restarts, num_or_spec, num_and_spec, output_dim]
    C_mat = C_mat.view(C_mat.shape[0], 1, len(cond_mat[0]), -1, C_mat.shape[-1])
    rhs_mat = rhs_mat.view(rhs_mat.shape[0], len(cond_mat[0]), -1)
    adv_example = adv_example[:,0:1]
    adv_output = adv_output[0:1]
    # adv_example and adv_output are duplicate num_or_spec times due to the duplicated data_max and data_min
    #[num_example, num_or_spec, num_and_spec]

    attack_margin = torch.matmul(C_mat, adv_output.unsqueeze(1).unsqueeze(-1)).squeeze(-1) - rhs_mat
    data_max = data_max.view(data_max.shape[0], 1, len(cond_mat[0]), -1, *x.shape[1:])
    data_min = data_min.view(data_min.shape[0], 1, len(cond_mat[0]), -1, *x.shape[1:])
    
    violated = (attack_margin < 0).all(-1)
    # [num_example, 1, num_or_spec]
    max_valid = (adv_example <= data_max).view(*data_max.shape[:4], -1)
    min_valid = (adv_example >= data_min).view(*data_max.shape[:4], -1)
    # [num_example, 1, num_or_spec, num_and_spec, -1]

    max_valid = max_valid.all(-1).all(-1)
    min_valid = min_valid.all(-1).all(-1)
    # [num_example, 1, num_or_spec]


    violate_index = (violated & max_valid & min_valid).nonzero()
    # [num_examples, num_restarts, num_or_spec]

    x = adv_example.view(-1)
    with open(res_path, 'w+') as f:
        # f.write("; Counterexample with prediction: {}\n".format(attack_label))
        # f.write("\n")

        input_dim = np.prod(adv_example[0].shape)
        # for i in range(input_dim):
        #     f.write("(declare-const X_{} Real)\n".format(i))
        #
        # for i in range(adv_output.shape[1]):
        #     f.write("declare-const Y_{} Real)\n".format(i))

        # f.write("; Input assignment:\n")
        f.write("(")
        for i in range(input_dim):
            f.write("(X_{}  {})\n".format(i, x[i].item()))
        
        # f.write("\n")
        # f.write("; Output obtained:\n")
        for i in range(adv_output.shape[1]):
            if i == 0:
                f.write("(Y_{} {})".format(i, adv_output[0,i]))
            else:
                f.write("\n(Y_{} {})".format(i, adv_output[0,i]))
        f.write(")")
        f.flush()

        '''
        ## generate the specifications from C matrix and rhs_mat
        violated_C = C_mat[violate_index[:,0], violate_index[:,1], violate_index[:,2]] # [num_vio, num_and_spec, output_dim]
        rhs_mat = rhs_mat[violate_index[:,0], violate_index[:,2]] # [num_vio_or, num_and_spec]
        f.write("; Violated output constraints:\n")
        f.write("(assert (or\n")

        for or_index, _or in enumerate(violated_C):
            f.write('(and ')
            for and_index, spec in enumerate(_or):
                # f.write('(<= ')
                y_list = []
                for index, factor in enumerate(spec):
                    if factor == 1:
                        y_list.append((1, index))
                        break
                for index, factor in enumerate(spec):
                    if factor == -1:
                        y_list.append((-1, index))
                        break
                if rhs_mat[or_index, and_index] != 0:
                    y_list.append((0, rhs_mat[or_index, and_index].item()))

                if y_list[0][0] == 1:
                    f.write('(<= ')
                else:
                    f.write('(>= ')
                
                for yy in y_list:
                    if yy[0] != 0:
                        f.write("Y_{} ".format(yy[1]))
                    else:
                        f.write(str(yy[1] * y_list[0][0]))
                
                f.write(')')
                
            f.write(')\n')
        f.write("))")
        '''

def process_vnn_lib_attack(vnnlib, x):
    list_target_label_arrays = [[]]
    data_min_repeat = []
    data_max_repeat = []

    for vnn in vnnlib:
        data_range = torch.Tensor(vnn[0])
        spec_num = len(vnn[1])


        data_max_ = data_range[:,1].view(-1, *x.shape[1:]).to(x.device).expand(spec_num, *x.shape[1:]).unsqueeze(0)
        data_min_ = data_range[:,0].view(-1, *x.shape[1:]).to(x.device).expand(spec_num, *x.shape[1:]).unsqueeze(0)

        data_max_repeat.append(data_max_)
        data_min_repeat.append(data_min_)


        list_target_label_arrays[0].extend(list(vnn[1]))

    data_min_repeat = torch.cat(data_min_repeat, dim=1)
    data_max_repeat = torch.cat(data_max_repeat, dim=1)

    return list_target_label_arrays, data_min_repeat, data_max_repeat


def convert_carvana_model_vnnlib(model, vnnlib, c_mode='naive'):
    gt = np.array(vnnlib[0][0][-31*47:])  # binary mask ground truth
    new_x = vnnlib[0][0][:3*31*47]  # real input image
    assert np.all((gt[:, 0] == gt[:, 1]))   # make sure the mask ground truth has no perturbation
    assert len(np.unique(gt)) == 2  # make sure the mask ground truth is binary

    new_c = []
    if c_mode == 'one_by_one':
        # generate a spec which try to verify all properties one by one (similar with standard cifar-10 verified accuracy)
        for idx, gt_i in enumerate(gt):
            this_c = np.zeros((1, 2914))  # 2914 = 2*31*47, output size
            if gt_i[0] == 0:  # dim 0 > dim 1
                this_c[0, idx] = 1  # ground truth idx
                this_c[0, idx + 1457] = -1  # target idx
            else:  # dim 0 < dim 1
                this_c[0, idx] = -1  # target idx
                this_c[0, idx + 1457] = 1  # ground truth idx
            new_c.append((this_c, np.array([0])))
        model = nn.Sequential(model, nn.Flatten())

    elif c_mode == 'together':
        # generate a spec which try to verify all properties together
        this_c = np.zeros((1457, 2914))  # 2914 = 2*31*47, output size
        for idx, gt_i in enumerate(gt):
            if gt_i[0] == 0:  # dim 0 > dim 1
                this_c[idx, idx] = 1  # ground truth idx
                this_c[idx, idx + 1457] = -1  # target idx
            else:  # dim 0 < dim 1
                this_c[idx, idx] = -1  # target idx
                this_c[idx, idx + 1457] = 1  # ground truth idx
        new_c.append((this_c, np.zeros([1457])))
        model = nn.Sequential(model, nn.Flatten())

    elif c_mode == 'naive':
        # original spec, count correct classified pixels
        new_c.append(vnnlib[0][1][0])
        model = Step_carvana(model, gt[:, 0])
    else:
        raise NotImplementedError

    new_vnnlib = [(new_x, new_c)]
    return model, new_vnnlib


def parse_model_shape_vnnlib(file_root, onnx_path, vnnlib_path):
    is_channel_last = False

    # FIXME clean up
    if arguments.Config["data"]["dataset"] in 'NN4SYS':
        shape = (-1, 1)
        from convert_nn4sys_model import convert_and_save_nn4sys, get_path
        path = get_path(os.path.join(file_root, onnx_path))
        if not os.path.exists(path):
            convert_and_save_nn4sys(os.path.join(file_root, onnx_path))
        # load pre-converted model
        model_ori = torch.load(path)
        print(f'Loaded from {path}')
        vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path), regression=True)
        return model_ori, is_channel_last, shape, vnnlib

    if arguments.Config["data"]["dataset"] == 'Carvana':
        shape = (-1, 3, 31, 47)
        path = os.path.join(file_root, onnx_path[:-5] + '_split.onnx')
        if not os.path.exists(path):
            from split_onnx import split_carvana
            print('Split carvana model from:', os.path.join(file_root, onnx_path))
            split_carvana(os.path.join(file_root, onnx_path))
        else:
            print(f'Loaded split model from {path}')
        model_ori, is_channel_last = load_model_onnx(path, input_shape=shape[1:], force_return=True)
        vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))
        model_ori, vnnlib = convert_carvana_model_vnnlib(model_ori, vnnlib, c_mode='naive')

        return model_ori, is_channel_last, shape, vnnlib

    if arguments.Config["data"]["dataset"] == 'MNIST':
        shape = (-1, 1, 28, 28)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:])
    elif arguments.Config["data"]["dataset"] == 'CIFAR':
        shape = (-1, 3, 32, 32)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:])
    elif arguments.Config["data"]["dataset"] == 'CIFAR100':
        shape = (-1, 3, 32, 32)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:])
    elif arguments.Config["data"]["dataset"] == 'TinyImageNet':
        shape = (-1, 3, 56, 56)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:])
    elif arguments.Config["data"]["dataset"] == 'ImageNet':
        shape = (-1, 3, 224, 224)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'ACASXU':
        shape = (-1, 5)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(5,), force_return=True)
        model_ori = torch.nn.Sequential(*list(model_ori.modules())[2:])  # remove sub(0) layer in ACASXU
    elif arguments.Config["data"]["dataset"] in 'TEST':
        shape = (-1, 1)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(1,), force_return=True)
        model_ori = convert_test_model(model_ori)
    elif arguments.Config["data"]["dataset"] == 'NN4SYS_2022':
        from convert_nn4sys_model import parse_nn4sys
        shape, model_ori = parse_nn4sys(file_root, onnx_path)
    elif arguments.Config["data"]["dataset"] == 'Cifar_biasfield':
        shape = (-1, 16)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'Collins-rul-cnn':
        if 'window_20' in onnx_path:
            shape = (-1, 1, 20, 20)
        elif 'window_40' in onnx_path:
            shape = (-1, 1, 40, 20)
        else:
            raise NotImplementedError
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'TLLVerifyBench':
        shape = (-1, 2)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'Dubinsrejoin':
        shape = (-1, 8)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'Cartpole':
        shape = (-1, 4)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'Lunarlander':
        shape = (-1, 8)
        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    elif arguments.Config["data"]["dataset"] == 'Reach_probability':
        if 'gcas' in onnx_path:
            shape = (-1, 14)
        elif 'robot' in onnx_path:
            shape = (-1, 5)
        elif 'vdp' in onnx_path:
            shape = (-1, 3)
        else:
            raise NotImplementedError

        model_ori, is_channel_last = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=shape[1:], force_return=True)
    else:
        raise NotImplementedError

    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))

    return model_ori, is_channel_last, shape, vnnlib
