import time
import atexit
from functools import wraps
import torch
import torch.nn as nn

start_time = time.time()
profile_data = {}

def profile(target):
    if isinstance(target, type):
        for attr_name, attr_value in target.__dict__.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                setattr(target, attr_name, profile_method(attr_value))
        return target
    elif callable(target):
        return profile_method(target)
    else:
        raise TypeError("Profile can only be applied to functions or classes")

def profile_method(func):
    func_name = f"{func.__module__}.{func.__qualname__}"
    profile_data[func_name] = {"calls": 0, "total_time": 0.0}

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        profile_data[func_name]["calls"] += 1
        profile_data[func_name]["total_time"] += end_time - start_time

        return result

    return wrapper

def print_profile_data():
    all_time = time.time() - start_time
    if profile_data != {}:
        print(f"Program cost {all_time:.4f} seconds!")
        for func_name, data in profile_data.items():
            print(f"{func_name}: {data['calls']} calls, {data['total_time']:.4f} total seconds")

atexit.register(print_profile_data)
