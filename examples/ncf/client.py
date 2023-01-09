import sys
sys.path.append("../..")
from fedlab_rec.basic.client import Client
from fedlab_rec.utils.serialization import SerializationTool
from fedlab_rec.trainer.ranker_trainer import CTRTrainer

class NCFClient(Client):
    def __init__(self, model, num_clients, cuda, device, 
                 trainer, dataset, include_names, personal = True):
        super().__init__(model, num_clients, cuda, device, personal)
        self.trainer = trainer
        self.dataset = dataset
        self.include_names = include_names
    
    def load_model(self, payload, client_id):
        SerializationTool.deserialize_model(self.model, self.parameters[client_id])
        SerializationTool.deserialize_model(self.model, payload)
        
    def local_eval(self, payload, id_list):
        ndcg_all = 0.0
        hr_all = 0.0
        for id in id_list:
            self.load_model(payload, id)
            self.trainer.setup(self.model)
            ndcgs, hrs = self.trainer.evaluate(self.dataset.get_dataloader(id, mode="test"))
            ndcg_all += ndcgs[0]
            hr_all += hrs[0]
        print(f"hr@10: {hr_all/len(id_list)}, ndcg@10: {ndcg_all/len(id_list)}")
        
    def local_train(self, payload, id_list):
        self.cache = []
        for id in id_list:
            self.load_model(payload, id)
            self.trainer.setup(self.model)
            self.trainer.fit(self.dataset.get_dataloader(id, mode="train"))
            self.parameters[id] = SerializationTool.serialize_model(self.model)
            self.cache.append(SerializationTool.serialize_model(self.model, self.include_names))
        
    @property
    def uplink_package(self):
        return self.cache
            
            
    