import torch.nn as nn


class EMA:
    def __init__(self, ema_decay):
        super().__init__()
        self.ema_decay = ema_decay
        self.backup = {}
        self.shadow = {}

    def register(self, current_model: nn.Module):
        self.shadow = {}
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def reset_device(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name].to(param.data.device)

    def update(self, current_model: nn.Module, with_decay=True):
        for name, param in current_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if with_decay:
                    new_average = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
                else:
                    new_average = param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, current_model: nn.Module):
        if self.backup:
            raise RuntimeError("EMA parameters are already installed")
        parameters = [(name, param) for name, param in current_model.named_parameters() if param.requires_grad]
        missing = [name for name, _ in parameters if name not in self.shadow]
        if missing:
            raise RuntimeError(f"EMA is missing parameters: {missing}")
        for name, param in parameters:
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self, current_model: nn.Module):
        for name, param in current_model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
