from abc import abstractmethod, ABC


class FedDataset(ABC):
    @abstractmethod
    def get_dataloader(self, idx, mode='train'):
        raise NotImplementedError()





    

