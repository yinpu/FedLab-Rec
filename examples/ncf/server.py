import sys
sys.path.append("../..")
from fedlab_rec.basic.server import ServerHandler
from fedlab_rec.utils.serialization import SerializationTool
import random
from copy import deepcopy

class NCFServer(ServerHandler):
    def __init__(self, model, cuda, device, num_clients, global_round, include_names, sample_ratio):
        super().__init__(model, cuda, device)
        self.num_clients = num_clients
        self.global_round = global_round
        self.include_names = include_names
        self.sample_ratio = sample_ratio
        self.round = 0
        
    @property
    def if_stop(self):
        return self.round >= self.global_round

    @property
    def num_clients_per_round(self):
        return max(1, int(self.sample_ratio * self.num_clients))
    
    def sample_clients(self):
        """Return a list of client rank indices selected randomly. The client ID is from ``0`` to
        ``self.num_clients -1``."""
        selection = random.sample(range(self.num_clients),
                                  self.num_clients_per_round)
        return sorted(selection)
    
    @property
    def downlink_package(self):
        return SerializationTool.serialize_model(self.model, self.include_names)
    
    def load(self, payloads):
        weight = 1/len(payloads)
        for index, payload in enumerate(payloads):
            payload =  {k:weight*deepcopy(v) for k, v in payload}
            if index == 0:
                SerializationTool.deserialize_model(self.model, payload, mode="copy")
            else:
                SerializationTool.deserialize_model(self.model, payload, mode="add")
    
    
    
    

