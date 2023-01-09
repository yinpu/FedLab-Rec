import sys
sys.path.append("../..")
from client import NCFClient
from server import NCFServer
from fedlab_rec.trainer.ranker_trainer import CTRTrainer
from fedlab_rec.data.ml_100k import ML100kDataSet
from fedlab_rec.utils.functional import load_conf
from fedlab_rec.model.ncf import NCF
from fedlab_rec.utils.metric import topk_metrics
import pandas as pd
import argparse

class NCFPipeline:
    def __init__(self, client: NCFClient, server: NCFServer, num_clients):
        self.client = client
        self.server = server
        self.num_clients = num_clients
        self.epoch_flag = [False]*self.num_clients
        
    def __call__(self):
        while self.server.if_stop is False:
            # server side
            sampled_clients = self.server.sample_clients()
            for id in sampled_clients:
                self.epoch_flag[id] = True
            broadcast = self.server.downlink_package

            # client side
            self.client.local_train(broadcast, sampled_clients)
            uploads = self.client.uplink_package

            # server side
            self.server.load(uploads)
            
            if all(self.epoch_flag):
                broadcast = self.server.downlink_package
                self.client.local_eval(broadcast, list(range(self.num_clients)))
                self.epoch_flag = [False]*self.num_clients

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, help='', default="run.conf")
    args = parser.parse_args()
    if args.conf:
        config = load_conf(args.conf)['Default']
    else:
        raise BaseException("please provider a conf file")
    dataset = ML100kDataSet(root_dir='../../data/ml-100k',
                            batch_size=int(config['batch_size']))
    local_trainer = CTRTrainer(optimizer_params=eval(config['optimizer_params']),
                               n_epoch=int(config["epochs"]),
                               evaluate_fn=topk_metrics,
                               device="cuda:0")
    model = NCF(user_num=dataset.users_num, item_num=dataset.items_num)
    client = NCFClient(model=model,
                       num_clients=dataset.users_num,
                       cuda=True,
                       device="cuda:0",
                       trainer=local_trainer,
                       dataset=dataset,
                       include_names=['mlp_item_embeddings', 'gmf_item_embeddings', 'mlp', 'output_logits'])
    server = NCFServer(model=model,
                       cuda=True,
                       device="cuda:0",
                       num_clients=dataset.users_num,
                       global_round=int(config['com_round']),
                       sample_clients_num=int(config['sample_client']),
                       include_names=['mlp_item_embeddings', 'gmf_item_embeddings', 'mlp', 'output_logits'])
    pipeline = NCFPipeline(client, server, dataset.users_num)
    pipeline()
    
            
        

