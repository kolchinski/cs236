import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        prior_m, prior_v = prior

        batch = x.shape[0]


        qm, qv = self.enc.encode(x)
        # Now draw Zs from the posterior qm/qv
        z = ut.sample_gaussian(qm,qv)

        l_posterior = ut.log_normal(z, qm, qv)
        multi_m = prior_m.expand(batch, *prior_m.shape[1:])
        multi_v = prior_v.expand(batch, *prior_v.shape[1:])
        l_prior = ut.log_normal_mixture(z, multi_m, multi_v)
        kls = l_posterior - l_prior
        kl = torch.mean(kls)

        probs = self.dec.decode(z)
        recs = ut.log_bernoulli_with_logits(x, probs)
        rec = -1.0 * torch.mean(recs)

        nelbo = kl + rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        prior_m, prior_v = prior

        batch = x.shape[0]
        multi_x = ut.duplicate(x, iw)

        qm, qv = self.enc.encode(x)
        multi_qm = ut.duplicate(qm, iw)
        multi_qv = ut.duplicate(qv, iw)

        # z will be (batch*iw x z_dim)
        # with sampled z's for a given x non-contiguous!
        z = ut.sample_gaussian(multi_qm,multi_qv)

        probs = self.dec.decode(z)
        recs = ut.log_bernoulli_with_logits(multi_x, probs)
        rec = -1.0 * torch.mean(recs)

        multi_m = prior_m.expand(batch*iw, *prior_m.shape[1:])
        multi_v = prior_v.expand(batch*iw, *prior_v.shape[1:])
        z_priors = ut.log_normal_mixture(z, multi_m, multi_v)
        x_posteriors = recs
        z_posteriors = ut.log_normal(z, multi_qm, multi_qv)

        kls = z_posteriors - z_priors
        kl = torch.mean(kls)

        log_ratios = z_priors + x_posteriors - z_posteriors
        # Should be (batch*iw, z_dim), batch ratios non contiguous

        unflat_log_ratios = log_ratios.reshape(iw, batch)

        niwaes = ut.log_mean_exp(unflat_log_ratios, 0)
        niwae = -1.0 * torch.mean(niwaes)


        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
