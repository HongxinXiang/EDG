from collections import OrderedDict

from torch import nn
import torch
import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_classifier(arch, in_features, num_tasks):
    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, in_features // 2)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(in_features // 2, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))


class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()
        self.network = get_classifier(arch="arch1", in_features=in_features, num_tasks=out_features)

        self.apply(weights_init)

    def forward(self, x):
        logit = self.network(x)
        return logit


def load_checkpoint(pretrained_pth, EDPredictor,
                    optimizer=None, lr_scheduler=None, logger=None):
    log = logger.info if logger is not None else print
    flag = False
    resume_desc = None
    if os.path.isfile(pretrained_pth):
        pretrained_model = torch.load(pretrained_pth)
        resume_desc = pretrained_model["desc"]
        model_list = [("EDPredictor", EDPredictor)]
        if optimizer is not None:
            model_list.append(("optimizer", optimizer, "optimizer"))
        if lr_scheduler is not None:
            model_list.append(("lr_scheduler", lr_scheduler, "lr_scheduler"))
        for model_key, model in model_list:
            try:
                model.load_state_dict(pretrained_model[model_key])
            except:
                ckp_keys = list(pretrained_model[model_key])
                cur_keys = list(model.state_dict())
                model_sd = model.state_dict()
                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = pretrained_model[model_key][ckp_key]
                model.load_state_dict(model_sd)
            log("[resume info] resume {} completed.".format(model_key))
        flag = True
    else:
        log("===> No checkpoint found at '{}'".format(pretrained_pth))

    return flag, resume_desc
