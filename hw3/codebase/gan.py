import torch
from torch.nn import functional as F

def loss_nonsaturating(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    - g_loss (torch.Tensor): nonsaturating generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    #   - F.logsigmoid

    d_loss = -1./batch_size * (F.logsigmoid(d(x_real)).sum() + torch.log(1. - torch.sigmoid(d(g(z)))).sum())
    g_loss = -1./batch_size * F.logsigmoid(d(g(z))).sum()

    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def conditional_loss_nonsaturating(g, d, x_real, y_real, *, device):
    '''
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    - g_loss (torch.Tensor): nonsaturating conditional generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR CODE STARTS HERE

    d_loss = -1./batch_size * (F.logsigmoid(d(x_real, y_real)).sum() +
                               torch.log(1. - torch.sigmoid(d( g(z, y_fake), y_fake ))).sum())
    g_loss = -1./batch_size * F.logsigmoid(d(g(z, y_fake), y_fake)).sum()

    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def loss_wasserstein_gp(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    - g_loss (torch.Tensor): wasserstein generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)

    alpha = torch.rand(batch_size, 1, 1, 1)
    rx = alpha * g(z) + (1. - alpha) * x_real
    rx_grad = torch.autograd.grad(d(rx).sum(), rx, create_graph=True)
    grad_norms = (rx_grad[0]**2).sum(dim=(1,2,3)).sqrt()

    d_loss = d(g(z)) - d(x_real)
    d_loss = d_loss + 10. * (grad_norms - 1.)**2
    d_loss = d_loss.mean()

    g_loss = -1. * d(g(z)).mean()

    # YOUR CODE ENDS HERE

    return d_loss, g_loss
