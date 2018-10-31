import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

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
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        qm, qv = self.enc.encode(x)
        pm = self.z_prior[0].expand(qm.shape)
        pv = self.z_prior[1].expand(qv.shape)
        kls = ut.kl_normal(qm, qv, pm, pv)
        kl = torch.mean(kls)

        z = ut.sample_gaussian(qm,qv)
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

        batch = x.shape[0]
        multi_x = ut.duplicate(x, iw)

        qm, qv = self.enc.encode(x)
        multi_qm = ut.duplicate(qm, iw)
        multi_qv = ut.duplicate(qv, iw)
        #multi_qm = qm.unsqueeze(1).expand(
        #    qm.shape[0], iw, *qm.shape[1:]).reshape(-1, *qm.shape[1:])
        #multi_qv = qv.unsqueeze(1).expand(
        #    qv.shape[0], iw, *qv.shape[1:]).reshape(-1, *qv.shape[1:])

        # z will be (batch*iw x z_dim)
        # with sampled z's for a given x non-contiguous!
        z = ut.sample_gaussian(multi_qm,multi_qv)

        probs = self.dec.decode(z)
        recs = ut.log_bernoulli_with_logits(multi_x, probs)
        rec = -1.0 * torch.mean(recs)

        multi_pm = self.z_prior[0].expand(multi_qm.shape)
        multi_pv = self.z_prior[1].expand(multi_qv.shape)

        z_priors = ut.log_normal(z, multi_pm, multi_pv)
        x_posteriors = recs
        z_posteriors = ut.log_normal(z, multi_qm, multi_qv)

        log_ratios = z_priors + x_posteriors - z_posteriors
        # Should be (batch*iw, z_dim), batch ratios non contiguous

        unflat_log_ratios = log_ratios.reshape(iw, batch)

        niwaes = ut.log_mean_exp(unflat_log_ratios, 0)
        niwae = torch.mean(niwaes)

        pm = self.z_prior[0].expand(qm.shape)
        pv = self.z_prior[1].expand(qv.shape)
        kls = ut.kl_normal(qm, qv, pm, pv)
        kl = torch.mean(kls)

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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
