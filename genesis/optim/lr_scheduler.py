import math

class LambdaLR:
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.lr_lambdas = lr_lambda
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lrs = self.optimizer.lr

    def state_dict(self):
        return {
                "last_epoch": self.last_epoch,
                "base_lrs": self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]
        self.base_lrs = state_dict["base_lrs"]

    def get_last_lr(self):
        return self._last_lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr_mult = self.lr_lambdas(self.last_epoch) 
        self.optimizer.lr = self.base_lrs * lr_mult
        self._last_lr = self.optimizer.lr

def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) \
                / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)
