import sys
sys.path.append("../..")
from client import NCFClient
from server import NCFServer
from fedlab_rec.trainer.ranker_trainer import CTRTrainer
from fedlab_rec.utils.dataset import RecDataset
from fedlab_rec.utils.functional import load_conf
from fedlab_rec.model.ncf import NCF
import pandas as pd
import argparse

class NCFPipeline:
    def __init__(self, client: NCFClient, server: NCFServer):
        self.client = client
        self.server = server
        
    def __call__(self):
        while self.server.if_stop is False:
            # server side
            sampled_clients = self.server.sample_clients()
            broadcast = self.server.downlink_package

            # client side
            self.client.local_process(broadcast, sampled_clients)
            uploads = self.client.uplink_package

            # server side
            self.server.load(uploads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, help='', default="run.conf")
    args = parser.parse_args()
    if args.conf:
        config = load_conf(args.conf)['Default']
    else:
        raise BaseException("please provider a conf file")
    df = pd.read_feather("../../data/ml-1m/ratings.feather")
    dataset = RecDataset(df=df,
                         num_clients=int(config['total_client']),
                         batch_size=int(config['batch_size']))
    local_trainer = CTRTrainer(optimizer_params=eval(config['optimizer_params']),
                               n_epoch=int(config["epochs"]),
                               device="cuda:0")
    model = NCF(user_num=df['user_id'].max()+1, item_num=df['item_id'].max()+1)
    client = NCFClient(model=model,
                       num_clients=int(config['total_client']),
                       cuda=True,
                       device="cuda:0",
                       trainer=local_trainer,
                       dataset=dataset,
                       include_names=['mlp_item_embeddings', 'gmf_item_embeddings'])
    server = NCFServer(model=model,
                       cuda=True,
                       device="cuda:0",
                       num_clients=int(config['total_client']),
                       global_round=int(config['com_round']),
                       sample_ratio=float(config['sample_ratio']),
                       include_names=['mlp_item_embeddings', 'gmf_item_embeddings'])
    pipeline = NCFPipeline(client, server)
    pipeline()
    
            
        

