import torch

import os
import json
import logging
import pynvml
import random
import numpy as np
from collections import Counter
from configparser import RawConfigParser


def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        cuda_devices = range(deviceCount)

    assert max(cuda_devices) < deviceCount
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))

def init_logging(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    logging.basicConfig(filename=file_path,level=logging.INFO)


def init_seed(seed=1024):
    random.seed(seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
    # torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)


def load_conf(path, verbose=True):
    cf = RawConfigParser()
    cf.read(path)

    for title in cf:
        if verbose:
            print(f"\n[{title}]")
        for key in cf[title]:
            value = cf[title][key].encode(
                'utf-8') if type(cf[title][key]) is not str else cf[title][key]
            if verbose:
                print(f"{key}={cf[title][key]}")
    return cf


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
