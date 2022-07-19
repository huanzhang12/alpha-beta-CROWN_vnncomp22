import torch
import time
import torch.nn as nn
from auto_LiRPA import BoundedTensor
from beta_CROWN_solver import LiRPAConvNet
import numpy as np
from auto_LiRPA.perturbations import PerturbationLpNorm
class VDP(nn.Module):
    def __init__(self):
        super(VDP, self).__init__()
        self.fc1 = torch.nn.Linear(3, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 3)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)

def precompile_jit_kernels():
    if torch.cuda.is_available():
        print(f'Pre-compile jit kernels on a toy network...')
        start_time = time.time()
        model_ori = VDP()
        data = torch.randn(1, 3).cuda()
        num_outputs = 3
        y = 0
        labels = torch.tensor([y]).long()
        c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), num_outputs - 1, num_outputs))
        model = LiRPAConvNet(model_ori, y, None, in_size=data.shape, c=c)

        data_lb = torch.tensor([[-2.5, -2.5, 0.]], device="cuda").cuda()
        data_ub = torch.tensor([[2.5, 2.5, 5.0]], device="cuda").cuda()
        ptb = PerturbationLpNorm(norm=np.inf, eps=0.02, x_L = data_lb, x_U = data_ub)
        x = BoundedTensor(data, ptb).cuda()

        lb, ub, aux = model.net.init_slope((x, ), share_slopes=True, c=c, bound_upper=False)
        model.net.set_bound_opts({'optimize_bound_args': {'ob_iteration': 2, 'ob_no_float64_last_iter': True, 'ob_no_amp': True}})
        ret = model.net.compute_bounds(x=(x,), method='CROWN-Optimized', C=c, bound_upper=False)

        del data, c, data_lb, data_ub, ptb, x, lb, ub, aux
        del model_ori, model, labels, ret, I

        torch.cuda.empty_cache()
        print(f'JIT kernels compiled in {time.time() - start_time:.4f}s.')
