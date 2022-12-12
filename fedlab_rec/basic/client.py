from abc import abstractclassmethod, abstractmethod
from ..utils.serialization import SerializationTool
from random import randint
from typing import List

import torch

from copy import deepcopy
from ..utils.functional import get_best_gpu



class Client:
    """Base class. Simulate multiple clients in sequence in a single process.
    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 num_clients: int,
                 cuda: bool,
                 device: str = None,
                 personal: bool = False) -> None:
        
        self.num_clients = num_clients
        self.cuda = cuda

        if cuda:
            # dynamic gpu acquire.
            if device is None:
                self.device = get_best_gpu()
            else:
                self.device = device
            self.model = deepcopy(model).cuda(self.device)
        else:
            self.model = deepcopy(model).cpu()
    
        if personal:
            self.parameters = [
                SerializationTool.serialize_model(self.model) for _ in range(num_clients)
            ] 
        else:
            self.parameters = SerializationTool.serialize_model(self.model)
        

    def setup_dataset(self):
        """Override this function to set up local dataset for clients"""
        raise NotImplementedError()

    def setup_trainer(self):
        """"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def uplink_package(self):
        """Return a tensor list for uploading to server.
            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def local_process(self, id_list: list, payload: List[torch.Tensor]):
        """Define the local main process."""
        # Args:
        #     id_list (list): The list consists of client ids.
        #     payload (List[torch.Tensor]): The information that server broadcasts to clients.
        raise NotImplementedError()

    def train(self):
        """Override this method to define the algorithm of training your model. This function should manipulate :attr:`self._model`"""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()

    def validate(self):
        """Validate quality of local model."""
        raise NotImplementedError()