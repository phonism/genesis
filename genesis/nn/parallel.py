import genesis
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
            grad.data.data /= self.world_size
            dist.all_reduce(grad.data.data, op=dist.ReduceOp.SUM)
            retry_count = 0
            max_retries = 5
            while retry_count <= max_retries:
                try:
                    dist.all_reduce(grad.data.data, op=dist.ReduceOp.SUM)
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"all_reduce failed (attempt {retry_count}): {e}") 
                    if retry_count > max_retries:
                        print("Max retries reached. Proceeding without successful all_reduce.")
                        break
                    else:
                        print("Retrying all_reduce operation...")
        return hook

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def state_dict(self, prefix=""):
        return self.model.state_dict(prefix=prefix)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def parameters(self):
        return self.model.parameters()
    
    def num_parameters(self):
        return self.model.num_parameters()
