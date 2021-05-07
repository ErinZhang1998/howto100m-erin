from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch as th
import numpy as np

class MaxMarginRankingLoss(th.nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = th.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float().cuda()


    def forward(self, x):
        d = th.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1:
            max_margin = max_margin * self.mm_mask
        return max_margin.mean()

class TripletLoss(th.nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = th.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float().cuda()

    def generate_mask(self, labels):
        '''
        labels: (N,1)
        mask: (N,N,N)

        False if any:
        1) label i != label j
        2) label i == label k
        3) i == j
        '''
        labels = labels + 1
        N = len(labels)
        la_not_lp = labels[None, :] != labels[:, None]
        la_is_ln = labels[None, :] == labels[:, None]
        # print(labels.shape, la_not_lp.shape, la_is_ln.shape)
        la_not_lp = la_not_lp.view((N,N))
        la_is_ln = la_is_ln.view((N,N))
        mask1 = la_not_lp[:, :,None] + la_is_ln[:, None, :]

        ind_vec = th.arange(N).view((-1,1))
        a_eq_p = (ind_vec[None, :] == ind_vec[:, None]).view((N,N))
        a_eq_p = a_eq_p[:,:,None]
        all_false = (th.zeros(N) > 0).view((1,-1))
        all_false = all_false[None,:,:]
        mask2 = a_eq_p + all_false
        mask2 = mask2.to(mask1.device)

        mask = th.logical_not(mask1 + mask2)
        return mask

    def calculate_loss(self, pairwise_dist, labels):
        anchor_positive_dist = pairwise_dist[:, :, None] #th.unsqueeze(pairwise_dist, dim=2)
        anchor_negative_dist = pairwise_dist[:, None, :] #th.unsqueeze(pairwise_dist, dim=1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        mask = self.generate_mask(labels)
        triplet_loss = F.relu(triplet_loss) * mask

        return th.sum(triplet_loss) / th.sum(mask).item()

    def forward(self, pairwise_dist, labels):
        loss_tvv = self.calculate_loss(pairwise_dist, labels)
        loss_vtt = self.calculate_loss(pairwise_dist.T, labels)

        return loss_tvv + loss_vtt