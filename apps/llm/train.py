import sys
sys.path.append("../../")
import os
import json
import genesis
from genesis import nn, init
from genesis.utils import profile
import time
import genesis.nn.functional as F
import random
import torch
import numpy as np
from genesis.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import ModelArgs, Transformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

set_seed(42)

multi_gpu = True

batch_size = 1
vocab_size = 151936
block_size = 2048

config = ModelArgs()
model = Transformer(config)
model.setup_caches(batch_size, config.block_size)

if multi_gpu:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(model, [local_rank])
else:
    local_rank = 0
    model.cuda()


optimizer = genesis.optim.AdamW(model.parameters(), lr=0.0001)
print("num_parameters:", model.num_parameters())

def get_batch(bs):
    x = []
    y = []
    cnt = 0
    with open("../../../datasets/train_data") as f:
        for line in f:
            cnt += 1
            if multi_gpu:
                if cnt % dist.get_world_size() != local_rank:
                    continue
            js = json.loads(line)
            x.append(js["input_ids"][:block_size])
            y.append(js["input_ids"][1:(block_size + 1)])
            if len(x) == batch_size:
                if multi_gpu:
                    xx = genesis.Tensor(np.array(x), device=genesis.device(local_rank))
                    yy = genesis.Tensor(np.array(y), device=genesis.device(local_rank))
                else:
                    xx = genesis.Tensor(np.array(x), device=genesis.device("cuda"))
                    yy = genesis.Tensor(np.array(y), device=genesis.device("cuda"))
                x = []
                y = []
                yield xx, yy

start_time = time.time()
total_cnt = 0
batch_loss = 0
batch_cnt = 0

accumulation_steps = 8
for idx, data in enumerate(get_batch(batch_size)):
    s_time = time.time()
    x = data[0]
    y = data[1]
    logits = model(x)

    B, T, C = 1, config.block_size, config.vocab_size
    logits = F.reshape(logits, (B * T, C))
    targets = F.reshape(y, (B*T,))
    loss = nn.SoftmaxLoss()(logits, targets)
    loss = loss / accumulation_steps
    s_time = time.time()
    loss.backward()
    s_time = time.time()

    batch_loss += loss.detach().numpy()
    batch_cnt += 1
    if (total_cnt + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        if local_rank == 0:
            print("step:", total_cnt, " loss:", batch_loss, " time:", time.time() - start_time)
        batch_loss = 0
        batch_cnt = 0
        start_time = time.time()
        if (total_cnt + 1) % (accumulation_steps * 100) == 0 and local_rank == 0:
            genesis.save( model.state_dict(), "../../../checkpoints/model.bin")
            print("save done!")
    total_cnt += 1

if multi_gpu:
    dist.destroy_process_group()
