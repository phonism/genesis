import genesis
import os

class autocast:
    def __enter__(self):
        genesis.enable_autocast = True
        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        genesis.enable_autocast = False
        genesis.upgrade = False
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return False

class GradScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self._scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0 
        self.is_inf = None
    
    def scale(self, loss):
        return loss * self._scale 
    
    def unscale_(self, optimizer):
        for param in optimizer.params:
            if param.grad is not None:
                param.grad.data /= self._scale

    def found_inf(self, optimizer=None):
        if self.is_inf is not None:
            return self.is_inf
        found_inf = False 
        for param in optimizer.params:
            if param.grad is not None:
                # Use genesis operations on the Tensor directly
                if genesis.isinf(param.grad).any() or genesis.isnan(param.grad).any():
                    found_inf = True
                    break 
        self.is_inf = found_inf
        return self.is_inf
    
    def step(self, optimizer):
        if not self.found_inf(optimizer):
            self.unscale_(optimizer)
            optimizer.step()
        
        optimizer.zero_grad()
    
    def update(self):
        if self.found_inf:
            self._scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker % self.growth_interval == 0:
                self._scale *= self.growth_factor 
        self.is_inf = None
