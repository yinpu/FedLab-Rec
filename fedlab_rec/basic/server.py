from abc import abstractmethod
from typing import List
from copy import deepcopy
from ..utils.functional import get_best_gpu

import torch



class ServerHandler:
    """An abstract class representing handler of parameter server.
    Please make sure that your self-defined server handler class subclasses this class
    Example:
        Read source code of :class:`SyncServerHandler` and :class:`AsyncServerHandler`.
        
    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool): Use GPUs or not.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 cuda: bool,
                 device: str = None) -> None:
        
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

    @property
    @abstractmethod
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def if_stop(self) -> bool:
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return False

    @abstractmethod
    def setup_optim(self):
        """Override this function to load your optimization hyperparameters."""
        raise NotImplementedError()

    @abstractmethod
    def global_update(self, buffer):
        raise NotImplementedError()

    @abstractmethod
    def load(self, payload):
        """Override this function to define how to update global model (aggregation or optimization)."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self):
        """Override this function to define the evaluation of global model."""
        raise NotImplementedError()