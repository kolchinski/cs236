import argparse
import os
import time
import torch
import torchvision
import tqdm

import codebase.gan
import codebase.network

parser = argparse.ArgumentParser(description='Trains a simple GAN.')
parser.add_argument('--device', default='cpu',
                    help='device to run on ("cpu" or "cuda")')
parser.add_argument('--num_epochs', default=1, type=int,
                    help='number of epochs to run for')
parser.add_argument('--loss_type', default='nonsaturating',
                    help='loss to train the gan with (nonsaturating, wasserstein_gp)')
args = parser.parse_args()

out_dir = 'out_' + args.loss_type + '_conditional'
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

dataset = torchvision.datasets.FashionMNIST('./data',
              train=True, download=True,
              transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((.5,), (.5,))]))

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=64,
                                          shuffle=True, drop_last=True)

torchvision.utils.save_image((next(iter(data_loader))[0] + 1) / 2., 
                             '%s/real.png' % out_dir)

device = torch.device(args.device)
g = codebase.network.ConditionalGenerator().to(device)
d = codebase.network.ConditionalDiscriminator().to(device)
z_test = torch.randn(10, 1, g.dim_z).repeat(1, 10, 1).reshape(100, g.dim_z).to(device)
y_test = torch.arange(10).repeat(10).to(device)

g_optimizer = torch.optim.Adam(g.parameters(), 1e-3, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(d.parameters(), 1e-3, betas=(0.5, 0.999))

global_step = 0
for _ in range(args.num_epochs):
    for x_real, y_real in tqdm.tqdm(data_loader):
        x_real, y_real = x_real.to(device), y_real.to(device)
        
        if args.loss_type == 'nonsaturating':
            loss = codebase.gan.conditional_loss_nonsaturating
        else:
            raise NotImplementedError

        d_loss, g_loss = loss(g, d, x_real, y_real, device=device)

        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        global_step += 1

        if global_step % 50 == 0:
            with torch.no_grad():
                g.eval()
                x_test = (g(z_test, y_test) + 1) / 2.
                torchvision.utils.save_image(
                    x_test, '%s/fake_%04d.png' % (out_dir, global_step),
                    nrow=10)
                g.train()

    with torch.no_grad():
        torch.save((g, d), '%s/model_%04d.pt' % (out_dir, global_step))
    
