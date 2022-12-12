import copy

class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience, mode="max"):
        self.patience = patience
        self.trial_counter = 0
        self.best_value = None
        self.best_weights = None
        self.mode = mode

    def stop_training(self, value, model):
        """whether to stop training.
        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """
        if (self.best_value is None) or\
            (self.mode == "max" and value > self.best_value) or\
            (self.mode == "min" and value < self.best_value):
            self.best_value = value
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True