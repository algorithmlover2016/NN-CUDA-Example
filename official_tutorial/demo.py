import logging
import torch
import pdb
pdb.set_trace()

torch.ops.load_library("build/libwarp_perspective.so")

def compute(x, y, z):
    return x.matmul(y) + torch.relu(z)

def compute_my_ops(x, y, z):
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + torch.relu(z)

@torch.jit.script
def compute_script(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    return x.matmul(y) + z

print(compute_script.graph)

@torch.jit.script
def compute_script_my_ops(x, y):
    if bool(x[0][0] == 42):
        z = 5
    else:
        z = 10
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + z

def main():
    inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(4, 5)]
    trace = torch.jit.trace(compute, inputs)
    print(trace.graph)

    inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(8, 5)]
    trace = torch.jit.trace(compute_my_ops, inputs)
    print(trace.graph)

    inputs = [torch.randn(4, 8), torch.randn(8, 5)]
    trace = torch.jit.trace(compute_script, inputs)
    print(trace.graph)

    inputs = [torch.randn(4, 8), torch.randn(8, 5)]
    trace = torch.jit.trace(compute_script_my_ops, inputs)
    print(trace.graph)
if __name__ == "__main__":
    main()
