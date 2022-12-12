import sys
sys.path.append("../..")
from fedlab_rec.basic.client import Client
from fedlab_rec.utils.serialization import SerializationTool
from fedlab_rec.trainer.ranker_trainer import CTRTrainer
from fedlab_rec.utils.dataset import RecDataset

class NCFClient(Client):
    def __init__(self, model, num_clients, cuda, device, 
                 trainer: CTRTrainer, dataset:RecDataset, include_names, personal = True):
        super().__init__(model, num_clients, cuda, device, personal)
        self.trainer = trainer
        self.dataset = dataset
        self.include_names = include_names
        
    def local_process(self, payload, id_list):
        self.cache = []
        for id in id_list:
            SerializationTool.deserialize_model(self.model, self.parameters[id])
            SerializationTool.deserialize_model(self.model, payload)
            self.trainer.setup(self.model)
            self.trainer.fit(self.dataset.get_dataloader(id, mode="train"),
                             self.dataset.get_dataloader(id, mode='valid'))
            self.trainer.evaluate(self.dataset.get_dataloader(id, mode="test"))
            self.cache.append(SerializationTool.serialize_model(self.model, self.include_names))
    
    @property
    def uplink_package(self):
        return self.cache
            
            
    