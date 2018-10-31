import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_logits = self.cls.classify(x)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y]) #1000,10. 0,100,200 dupe
        x = ut.duplicate(x, self.y_dim) #1000,784. 0,100,200 dupe

        #100x10
        y_prior = torch.tensor([0.1]).expand_as(y_prob).to(device)
        #(batch size,)
        kl_ys = ut.kl_cat(y_logprob, y_prob, y_prior)
        kl_y = torch.mean(kl_ys)


        #1000 x 64. Still 0,100,200 corresponding...
        zqm, zqv = self.enc.encode(x, y)
        zpm = self.z_prior_m.expand_as(zqm)
        zpv = self.z_prior_v.expand_as(zqv)

        #so the zpm, zpv go as x quickly, y slowly
        #equivalent to y being the 0th dimension

        #(batch_size * y_dim,)
        kl_zs_flat = ut.kl_normal(zqm, zqv, zpm, zpv)
        kl_zs = kl_zs_flat.reshape(10,100).t()
        kl_zs_weighted = kl_zs * y_prob
        kl_z = kl_zs_weighted.sum(1).mean()

        #1000 x 64
        z = ut.sample_gaussian(zqm, zqv)

        #1000 x 784
        probs = self.dec.decode(z, y)
        #(batch_size * y_dim,)
        recs_flat = ut.log_bernoulli_with_logits(x, probs)
        recs = recs_flat.reshape(10,100).t()
        recs_weighted = recs * y_prob
        rec = -1.0 * recs_weighted.sum(1).mean()

        nelbos = kl_ys + kl_zs_weighted.sum(1) - recs_weighted.sum(1)
        nelbo = torch.mean(nelbos)


        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls.classify(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec.decode(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
