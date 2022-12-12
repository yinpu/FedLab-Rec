import torch


class SerializationTool(object):
    @staticmethod
    def jug_need_param(name, include_names):
        if include_names is None:
            return True
        return any([include_name in name for include_name in include_names])
    
    
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module, include_names=None):
        grad_dict = {}
        for name, param in model.named_parameters():
            if SerializationTool.jug_need_param(name, include_names):
                grad_dict[name] = param.grad.data.cpu()
        return grad_dict

    @staticmethod
    def serialize_model(model: torch.nn.Module, include_names=None):
        grad_dict = {}
        for name, param in model.named_parameters():
            if SerializationTool.jug_need_param(name, include_names):
                grad_dict[name] = param.data.cpu()
        return grad_dict

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters_dict,
                          mode="copy"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.
        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
        """
        for name, parameter in model.named_parameters():
            if name not in serialized_parameters_dict:
                continue
            if mode == "copy":
                parameter.data.copy_(serialized_parameters_dict[name])
            elif mode == "add":
                parameter.data.add_(serialized_parameters_dict[name])
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))