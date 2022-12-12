import os
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from ..utils.earlystop import EarlyStopper


class CTRTrainer(object):
    """A general trainer for single task learning.
    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_metric=None,
        earlystop_mode="max",
        earlystop_patience=10,
        evaluate_fn=None,
        device="cpu"
    ):
        self.optimizer_params = optimizer_params
        if optimizer_params is None:
            self.optimizer_params = {"lr": 1e-3}
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.evaluate_fn = evaluate_fn
        if evaluate_fn is None:
            self.evaluate_fn = roc_auc_score  #default evaluate function
        self.n_epoch = n_epoch
        self.device = device
        self.earlystop_metric = earlystop_metric
        self.earlystop_patience = earlystop_patience
        self.earlystop_mode = earlystop_mode
    
    def setup(self, model):
        self.model = model
        self.optimizer = self.optimizer_fn(self.model.parameters(), **self.optimizer_params)
        self.scheduler = None
        if self.scheduler_fn is not None:
            self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)
        self.early_stopper = None
        if self.earlystop_metric is not None:
            self.early_stopper = EarlyStopper(patience=self.earlystop_patience, mode=self.earlystop_mode)

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, data_dict in enumerate(tk0):
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}  #tensor to GPU
            loss = self.model.cal_loss(data_dict)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            if val_dataloader:
                auc = self.evaluate(val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper is not None and \
                    self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        #torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  #save best auc model

    def evaluate(self, data_loader):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, data_dict in enumerate(tk0):
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                y_pred, y = self.model(data_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def predict(self, data_loader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, data_dict in enumerate(tk0):
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                y_pred, y = self.model(data_dict)
                predicts.extend(y_pred.tolist())
        return predicts