import python_wgpu_torch_playground as plg
import numpy
import torch


def test_module_resolves():
    assert plg.numpy == numpy
    assert plg.torch == torch
