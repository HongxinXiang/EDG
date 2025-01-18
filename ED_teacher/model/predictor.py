import os
from collections import OrderedDict

import timm
import torch
from torch import nn

from ED_teacher.model.model_utils import weights_init, get_classifier


def get_timm_model_names():
    model_list = timm.list_models()
    return model_list


def get_predictor(arch, in_features, num_tasks, inner_dim=None, dropout=0.2, activation_fn=None):
    if inner_dim is None:
        inner_dim = in_features // 2
    if activation_fn is None:
        activation_fn = "gelu"

    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, inner_dim)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(inner_dim, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))
    elif arch == "none":
        return nn.Identity()


class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()
        self.network = get_classifier(arch="arch1", in_features=in_features, num_tasks=out_features)

        self.apply(weights_init)

    def forward(self, x):
        logit = self.network(x)
        return logit


def save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict, desc, epoch, save_path, name_pre, name_post='_best'):
    state = {
        'epoch': epoch,
        'desc': desc
    }

    if model_dict is not None:
        for key in model_dict.keys():
            model = model_dict[key]
            state[key] = {k: v.cpu() for k, v in model.state_dict().items()}
    if optimizer_dict is not None:
        for key in optimizer_dict.keys():
            optimizer = optimizer_dict[key]
            state[key] = optimizer.state_dict()
    if lr_scheduler_dict is not None:
        for key in lr_scheduler_dict.keys():
            lr_scheduler = lr_scheduler_dict[key]
            state[key] = lr_scheduler.state_dict()

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))


class Backboneredictor(torch.nn.Module):
    def __init__(self, model_name, head_arch, num_tasks, pretrained=False, head_arch_params=None, **kwargs):
        super(FramePredictor, self).__init__()

        assert model_name in get_timm_model_names()

        self.model_name = model_name
        self.head_arch = head_arch
        self.num_tasks = num_tasks
        self.pretrained = pretrained
        if head_arch_params is None:
            head_arch_params = {"inner_dim": None, "dropout": 0.2, "activation_fn": None}
        self.head_arch_params = head_arch_params

        # create base model
        self.model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        # some attributes of base model
        self.classifier_name = self.model.default_cfg["classifier"]
        self.in_features = self.get_in_features()
        # self-defined head for prediction
        self_defined_head = self.create_self_defined_head()
        self.set_self_defined_head(self_defined_head)

    def forward(self, x):
        return self.model(x)

    def get_in_features(self):
        if type(self.classifier_name) == str:
            if "." not in self.classifier_name and isinstance(getattr(self.model, self.classifier_name), torch.nn.modules.linear.Identity):
                in_features = self.model.num_features
            else:
                classifier = self.model
                for item in self.classifier_name.split("."):
                    classifier = getattr(classifier, item)
                in_features = classifier.in_features
        elif type(self.classifier_name) == tuple or type(self.classifier_name) == list:
            in_features = []
            for item_name in self.classifier_name:
                classifier = self.model
                for item in item_name.split("."):
                    classifier = getattr(classifier, item)
                in_features.append(classifier.in_features)
        else:
            raise Exception("{} is undefined.".format(self.classifier_name))
        return in_features

    def create_self_defined_head(self):
        if type(self.in_features) == list or type(self.in_features) == tuple:
            assert len(self.classifier_name) == len(self.in_features)
            head_predictor = []
            for item_in_features in self.in_features:
                single_predictor = get_predictor(arch=self.head_arch, in_features=item_in_features,
                                                 num_tasks=self.num_tasks,
                                                 inner_dim=self.head_arch_params["inner_dim"],
                                                 dropout=self.head_arch_params["dropout"],
                                                 activation_fn=self.head_arch_params["activation_fn"])
                head_predictor.append(single_predictor)
        elif type(self.classifier_name) == str:
            head_predictor = get_predictor(arch=self.head_arch, in_features=self.in_features, num_tasks=self.num_tasks,
                                           inner_dim=self.head_arch_params["inner_dim"],
                                           dropout=self.head_arch_params["dropout"],
                                           activation_fn=self.head_arch_params["activation_fn"])
        else:
            raise Exception("error type in classifier_name ({}) and in_features ({})".format(type(self.classifier_name),
                                                                                             type(self.in_features)))
        return head_predictor

    def set_self_defined_head(self, self_defined_head):
        if type(self.classifier_name) == list or type(self.classifier_name) == tuple:
            for predictor_idx, item_classifier_name in enumerate(self.classifier_name):
                classifier = self.model
                if "." in item_classifier_name:
                    split_classifier_name = item_classifier_name.split(".")
                    for i, item in enumerate(split_classifier_name):
                        classifier = getattr(classifier, item)
                        if i == len(split_classifier_name) - 2:
                            setattr(classifier, split_classifier_name[-1], self_defined_head[predictor_idx])
                else:
                    setattr(self.model, item_classifier_name, self_defined_head[predictor_idx])
        elif "." in self.classifier_name:
            classifier = self.model
            split_classifier_name = self.classifier_name.split(".")
            for i, item in enumerate(split_classifier_name):
                classifier = getattr(classifier, item)
                if i == len(split_classifier_name) - 2:
                    setattr(classifier, split_classifier_name[-1], self_defined_head)
        else:
            setattr(self.model, self.classifier_name, self_defined_head)

