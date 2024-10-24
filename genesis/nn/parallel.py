import genesis
import torch
import torch.distributed as dist

class DistributedDataParallel(genesis.nn.Module):
    def __init__(self, model, device_ids):
        self.model = model
        self.device_ids = device_ids
        self.model.cuda(device_ids[0])
        self.world_size = dist.get_world_size()

        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(grad):
            dist.all_reduce(grad.data.data, op=dist.ReduceOp.SUM)
            grad.data.data /= self.world_size
        return hook

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

