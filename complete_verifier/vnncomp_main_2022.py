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
import argparse
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument("CATEGORY", type=str)
parser.add_argument("ONNX_FILE", type=str, default=None, help='ONNX_FILE')
parser.add_argument("VNNLIB_FILE", type=str, default=None, help='VNNLIB_FILE')
parser.add_argument("RESULTS_FILE", type=str, default=None, help='RESULTS_FILE')
parser.add_argument("TIMEOUT", type=float, default=180, help='timeout for one property')

args = parser.parse_args()

python_path = sys.executable
library_path = os.path.dirname(os.path.realpath(__file__))

cmd = f"{python_path} {library_path}/bab_verification_general.py --config {library_path}/"

if args.CATEGORY == "carvana_unet_2022":
    if "unet_simp" in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/carvana-unet-simp.yaml"
    else:
        cmd += "exp_configs/vnncomp22/carvana-unet-upsample.yaml"

elif args.CATEGORY == "cifar100_tinyimagenet_resnet":
    if 'CIFAR100_resnet_small' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_small_2022.yaml"
    elif 'CIFAR100_resnet_medium' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_med_2022.yaml"
    elif 'CIFAR100_resnet_large' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_large_2022.yaml"
    elif 'CIFAR100_resnet_super' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_super_2022.yaml"
    else:
        cmd += "exp_configs/vnncomp22/tinyimagenet_2022.yaml"

elif args.CATEGORY == "cifar_biasfield":
    cmd += "exp_configs/vnncomp22/cifar_biasfield.yaml"

elif args.CATEGORY == "collins_rul_cnn":
    cmd += "exp_configs/vnncomp22/collins-rul-cnn.yaml"

elif args.CATEGORY == "mnist_fc":
    if '256x2' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/mnistfc_small.yaml"
    else:
        cmd += "exp_configs/vnncomp22/mnistfc.yaml"

elif args.CATEGORY == "nn4sys":
    if 'lindex' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/nn4sys_2022_lindex.yaml"
    elif '128d' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/nn4sys_2022_128d.yaml"
    else:
        cmd += "exp_configs/vnncomp22/nn4sys_2022_2048d.yaml"

elif args.CATEGORY == "oval21":
    cmd += "exp_configs/vnncomp22/oval22.yaml"

elif args.CATEGORY == "reach_prob_density":
    if 'gcas' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/reach_probability_gcas.yaml"
    else:
        cmd += "exp_configs/vnncomp22/reach_probability.yaml"

elif args.CATEGORY == "rl_benchmarks":
    if 'cartpole' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cartpole.yaml"
    elif 'lunarlander' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/lunarlander.yaml"
    else:
        cmd += "exp_configs/vnncomp22/dubins-rejoin.yaml"

elif args.CATEGORY == "sri_resnet_a":
    cmd += "exp_configs/vnncomp22/resnet_A.yaml"

elif args.CATEGORY == "sri_resnet_b":
    cmd += "exp_configs/vnncomp22/resnet_B.yaml"

elif args.CATEGORY == "tllverifybench":
    cmd += "exp_configs/vnncomp22/tllVerifyBench.yaml"

elif args.CATEGORY == "vggnet16_2022":
    cmd += "exp_configs/vnncomp22/vggnet16.yaml"

elif args.CATEGORY == "acasxu":
    cmd += "exp_configs/vnncomp22/acasxu.yaml"

elif args.CATEGORY == "cifar2020":
    cmd += "exp_configs/vnncomp22/cifar2020_2_255.yaml"

elif args.CATEGORY == "test":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --enable_input_split --dataset TEST --pgd_order skip"

else:
    exit("CATEGORY {} not supported yet".format(args.CATEGORY))

if 'test_prop' in args.VNNLIB_FILE:  # Handle mismatched category name for the test instance, to allow correct measurement of overhead.
        cmd = f"{python_path} {library_path}/bab_verification_general.py --config exp_configs/vnncomp22/acasxu.yaml"
elif 'test_nano' in args.VNNLIB_FILE or 'test_tiny' in args.VNNLIB_FILE or 'test_small' in args.VNNLIB_FILE:
        cmd = f"{python_path} {library_path}/bab_verification_general.py --enable_input_split --dataset TEST --pgd_order skip"

cmd += " --onnx_path " + str(args.ONNX_FILE)
cmd += " --vnnlib_path " + str(args.VNNLIB_FILE)
cmd += " --results_file " + str(args.RESULTS_FILE)
cmd += " --timeout " + str(args.TIMEOUT)

print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

os.system(cmd)
