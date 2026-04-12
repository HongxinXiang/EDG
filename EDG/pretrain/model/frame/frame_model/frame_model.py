import torch
from torch import nn
from model.frame.frame_utils import get_timm_models
from timm.models.swin_transformer import default_cfgs
from model.model_utils import weights_init
from model.model_utils import get_classifier
import torchvision


class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()
        self.network = get_classifier(arch="arch1", in_features=in_features, num_tasks=out_features)

        self.apply(weights_init)

    def forward(self, x):
        logit = self.network(x)
        return logit


if __name__ == '__main__':
    # model = SwinTransformer(model_name="swinv2_large_window12to24_192to384_22kft1k")
    # print(model)

    batch_size = 8
    in_features = 512
    n_frame = 20
    frame1 = torch.rand(size=(batch_size, in_features))
    frame2 = torch.rand(size=(batch_size, in_features))
    frame = torch.cat([frame1, frame2], dim=1)

    axisClassifier = AxisClassifier(in_features=in_features * 2)
    rotationClassifier = RotationClassifier(in_features=in_features * 2)
    angleClassifier = AngleClassifier(in_features=in_features * 2, num_tasks=n_frame//3-1)

    output_axis = axisClassifier(frame)
    output_rotation = rotationClassifier(frame)
    output_angle = angleClassifier(frame)

    print(123)