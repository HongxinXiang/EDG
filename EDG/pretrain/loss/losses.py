import sys
import torch
from torch import nn
import torch.nn.functional as F


class MutualInformationLoss(nn.Module):
    def __init__(self, lamb=0.0, EPS=sys.float_info.epsilon):
        super(MutualInformationLoss, self).__init__()
        self.lamb = lamb
        self.EPS = EPS

    def forward(self, view1, view2):
        """Contrastive loss for maximizng the consistency"""
        _, k = view1.size()
        p_i_j = self.compute_joint(view1, view2)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

        p_i_j[(p_i_j < self.EPS).data] = self.EPS
        p_j[(p_j < self.EPS).data] = self.EPS
        p_i[(p_i < self.EPS).data] = self.EPS

        loss = - p_i_j * (torch.log(p_i_j) - (self.lamb + 1) * torch.log(p_j) - (self.lamb + 1) * torch.log(p_i))

        loss = loss.sum()

        return loss

    def compute_joint(self, view1, view2):
        """Compute the joint probability matrix P"""

        bn, k = view1.size()
        assert (view2.size(0) == bn and view2.size(1) == k)

        p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j


class SupConLoss(nn.Module):
    '''
    loss download originally from https://github.com/amazon-research/sccl/blob/5811042f48a5cb500a99834d4390cb2ef012386b/learners/contrastive_utils.py
    paper can download from https://arxiv.org/abs/2103.12953
    wechat report is from https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650423321&idx=2&sn=c10064b701687cf05872d59020679d85&scene=21#wechat_redirect
    '''
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, *views, labels=None, mask=None):
        """If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf.
        Args:
            views: view1, view2, ..., their shape must be same.
            labels: ground truth of shape [bsz, 1].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = self.get_forward_features_from_multi_view(*views)  # shape [bsz, n_views, ...].

        device = (features.device if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # [负无穷, 0]

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def get_forward_features_from_multi_view(self, *views):
        '''

        :param views: the shape for each view must be same: [batch, h_dim]
        :return: features, which can be input to function forward().
        '''
        assert len(views) >= 2, "the number of view must >= 2."
        for idx in range(len(views)):
            assert views[0].shape == views[idx].shape
        features = []
        for idx in range(len(views)):
            features.append(F.normalize(views[idx], dim=1).unsqueeze(1))
        features = torch.cat(features, dim=1)
        return features


if __name__ == '__main__':
    criterionI = SupConLoss(temperature=0.1, base_temperature=0.1)
    v1 = torch.rand(size=(8, 512))
    v2 = torch.rand(size=(8, 512))
    loss = criterionI(v1, v2, labels=None)
    print(loss)
