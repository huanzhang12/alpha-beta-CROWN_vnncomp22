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
import arguments
import os
import torch
from utils import load_model_onnx, convert_nn4sys_model

def get_path(onnx_file):
    return f'{os.path.basename(onnx_file)}.pt'

def convert_and_save_nn4sys(onnx_file):
    model_ori = load_model_onnx(onnx_file, input_shape=(1,1))
    model_ori = convert_nn4sys_model(model_ori)
    name = get_path(onnx_file)
    torch.save(model_ori, name)
    print(f'Converted model saved to {name}')

def parse_nn4sys(file_root, onnx_path):
    if 'lindex' in onnx_path:
        shape = (-1, 1)
        model_ori, _ = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(shape[1:]), force_return=True)
    elif 'dual' in onnx_path:
        shape = (-1, 22, 14)
        model_ori, _ = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(shape[1:]), force_return=True)
    else:
        shape = (-1, 11, 14)
        model_ori, _ = load_model_onnx(os.path.join(file_root, onnx_path), input_shape=(shape[1:]), force_return=True)
    return shape, model_ori


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_file', type=str)
    args = parser.parse_args()
    convert_and_save_nn4sys(args.onnx_file)
