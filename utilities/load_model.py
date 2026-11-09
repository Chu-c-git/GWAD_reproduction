import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class CudaGraphForward:
    """
    Callable that runs `module(x)` by replaying a captured CUDA graph.

    The attack queries the target model / Delta-Net one example at a time, so each
    eager forward is bound by Python + kernel-launch overhead while the GPU idles.
    A CUDA graph replays the very same kernels (same cuDNN/cuBLAS algorithms, same
    weights) with a single launch, so outputs are bit-identical to eager.

    One graph is captured lazily per (shape, dtype, device, grad mode). Anything not
    eligible (CPU tensors, module in training mode, autograd enabled, graphs disabled
    via GWAD_CUDA_GRAPHS=0) falls back to the plain eager forward.
    Assumes the module's parameters are not modified/moved after capture.
    """

    def __init__(self, module):
        self.module = module
        self.graphs = {}
        self.enabled = os.environ.get('GWAD_CUDA_GRAPHS', '1') != '0' and torch.cuda.is_available()

    def __call__(self, x):
        if (not self.enabled or not isinstance(x, torch.Tensor) or not x.is_cuda
                or self.module.training or torch.is_grad_enabled()):
            return self.module(x)
        inference = torch.is_inference_mode_enabled()
        key = (tuple(x.shape), x.dtype, x.device, inference)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(x, inference)
            if not isinstance(entry[2], torch.Tensor):
                # e.g. DummyNet returns a python scalar: nothing to graph
                self.enabled = False
                return self.module(x)
            self.graphs[key] = entry
        graph, static_in, static_out = entry
        static_in.copy_(x)
        graph.replay()
        return static_out.clone()

    def _capture(self, x, inference):
        ctx = torch.inference_mode if inference else torch.no_grad
        with ctx():
            static_in = x.clone()
            # warm up on a side stream (lazy cuDNN/cuBLAS init must not be captured)
            s = torch.cuda.Stream(device=x.device)
            s.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.module(static_in)
            torch.cuda.current_stream(x.device).wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self.module(static_in)
        return graph, static_in, static_out


def graphed_forward(module):
    """Shared CudaGraphForward for `module` (cached on the module itself)."""
    fwd = module.__dict__.get('_cuda_graph_forward')
    if fwd is None:
        fwd = CudaGraphForward(module)
        module.__dict__['_cuda_graph_forward'] = fwd
    return fwd


class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

    def forward(self, x):
        return 0


def get_models(data_type, model_name):
    if data_type == 'cifar10':
        if model_name == 'resnet18':
            from net.cifar10.resnet import resnet18
            model = resnet18()
            model.load_state_dict(torch.load('model/cifar10/resnet18.pt', map_location='cpu'))
            model.name = 'resnet-18'
            model.loss = 'cross entropy'
        elif model_name == 'resnet34':
            from net.cifar10.resnet import resnet34
            model = resnet34()
            model.load_state_dict(torch.load('model/cifar10/resnet34.pt', map_location='cpu'))
            model.name = 'resnet-34'
            model.loss = 'cross entropy'
        elif model_name == 'resnet50':
            from net.cifar10.resnet import resnet50
            model = resnet50()
            model.load_state_dict(torch.load('model/cifar10/resnet50.pt', map_location='cpu'))
            model.name = 'resnet-50'
            model.loss = 'cross entropy'
        elif model_name == 'mobilenet':
            from net.cifar10.mobilenetv2 import MobileNetV2
            model = MobileNetV2()
            model.load_state_dict(torch.load('model/cifar10/mobilenet_v2.pt', map_location='cpu'))
            model.name = 'mobilenet-v2'
            model.loss = 'cross entropy'
        else:
            raise Exception("Invalid model name")
    else:
        model = DummyNet()
        model.name = 'dummy'
    return model
