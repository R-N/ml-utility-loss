import torch.autograd as autograd

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import time

from torch.autograd import Variable
from .modules import FCDiscriminator, FCGenerator
from .ctabgan import Condvec, Sampler

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class LatentGAN:

    # input_size should be input_size + cond_vector
    def __init__(
        self, 
        input_size, 
        latent_dim, 
        transformer_output_info, 
        batch_size=256, 
        lr=0.0002, 
        b1=0.5, 
        b2=0.999, 
        n_critic=5, 
        decoder=None,
        scaler=None
    ):
        self.input_size = input_size
        self.generator = None
        self.discriminator = None
        self.latent_dim = latent_dim
        self.measurements = []
        # Loss weight for gradient penalty
        self.lambda_gp = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transformer_output_info=transformer_output_info
        self.batch_size=batch_size
        self.lr=lr
        self.b1=b1
        self.b2=b2
        self.n_critic=n_critic
        self.decoder=decoder
        self.scaler=scaler

    def prepare_training(self):

        # self.generator = FCGenerator(self.input_size, self.latent_dim + cond_generator.n_opt)
        self.generator = FCGenerator(self.input_size, self.latent_dim)
        # self.discriminator = FCDiscriminator(self.input_size + cond_generator.n_opt, batch_size=batch_size)
        self.discriminator = FCDiscriminator(self.input_size, batch_size=self.batch_size)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def fit(
        self, 
        latent_data, 
        original_data, 
        epochs=5, 
    ):
        
        assert len(latent_data) == len(original_data)
        
        cond_generator = Condvec(original_data, self.transformer_output_info)
        data_sampler = Sampler(original_data, self.transformer_output_info)

        self.cond_generator = cond_generator
        self.data_sampler = data_sampler
        
        steps = (len(latent_data) // self.batch_size) + 1

        loss_g  = torch.tensor([1.0]) # just for logging purposes, it is reassigned down
        loss_d = torch.tensor([1.0])  # just for logging purposes, it is reassigned down

        # Experimentation purposes

        start_time = time.time()

        adversarial_loss = torch.nn.BCELoss()
        valid = Variable(Tensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
        non_valid = Variable(Tensor(self.batch_size, 1).fill_(0.0), requires_grad=False)
        # Experimentation purposes

        for epoch in tqdm(range(epochs)):

            self.generator.train()
            self.discriminator.train()

            for i in range(steps):
                
                cond_vecs = cond_generator.sample_train(self.batch_size)
                c, m, col, opt = cond_vecs
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)

                original_idx = data_sampler.sample_idx(self.batch_size, col, opt)
                real = latent_data[original_idx]
                real = torch.from_numpy(real.astype('float32')).to(self.device)
                
                ### TRAIN DISCRIMINATOR
                self.optimizer_D.zero_grad()

                z = Tensor(np.random.uniform(0, 1, (self.batch_size, self.latent_dim))).to(self.device)
                # z = torch.cat([z , c], dim=1).to(self.device)
                z = torch.cat([z], dim=1).to(self.device)
                z = Variable(z).to(self.device)

                fake = self.generator(z).to(self.device)
                
                # fake_cat_d = torch.cat([fake, c], dim=1).to(self.device).to(self.device)
                fake_cat_d = torch.cat([fake], dim=1).to(self.device)
                # real_cat_d = torch.cat([Variable(real.type(Tensor)), c], dim=1).to(self.device)
                real_cat_d = torch.cat([Variable(real.type(Tensor))], dim=1).to(self.device)

                real_probability = self.discriminator(real_cat_d).to(self.device)
                fake_probability = self.discriminator(fake_cat_d).to(self.device)

                # Gradient penalty
                # gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_cat_d.data, fake_cat_d.data)
                gradient_penalty = 0
                # Adversarial loss
                loss_d = -torch.mean(real_probability) + torch.mean(fake_probability) + self.lambda_gp * gradient_penalty
                # loss_d = (-(torch.log(real_probability).mean()) - (torch.log(1. - fake_probability).mean()))

                # real_loss = adversarial_loss(real_probability, valid)
                # fake_loss = adversarial_loss(fake_probability, non_valid)
                # loss_d = (real_loss + fake_loss) / 2
                loss_d.backward()

                self.optimizer_D.step()

                ### TRAIN GENERATOR
                self.optimizer_G.zero_grad()

                if i % self.n_critic == 0:
                        # Generate a batch of data
                    fake = self.generator(z).to(self.device)
                    # fake_probability = self.discriminator(torch.cat([fake, c], dim=1).to(self.device))
                    fake_probability = self.discriminator(torch.cat([fake], dim=1).to(self.device))

                    # loss_g = adversarial_loss(fake_probability, valid)
                    # loss_g = -(torch.log(fake_probability + 1e-4).mean())
                    loss_g = -torch.mean(fake_probability)
                    loss_g.backward()

                    self.optimizer_G.step()

            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" 
                % (epoch + 1, epochs, loss_d.item(), loss_g.item())
            )


    def compute_gradient_penalty(self, D, real_samples, fake_samples):

        """Calculates the gradient penalty loss for WGAN GP"""

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(self.device)
        d_interpolates = D(interpolates)

        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def sample(self, n):
        
        # turning the generator into inference mode to effectively use running statistics in batch norm layers
        self.generator.eval()
        
        # generating synthetic data in batches accordingly to the total no. required
        steps = (n // self.batch_size) + 1
        data = []
        print("Number of steps: " + str(steps))
        for _ in tqdm(range(steps)):
            # generating synthetic data using sampled noise and conditional vectors
            ### Generating a batch
            z = Tensor(np.random.uniform(0, 1, (self.batch_size, self.latent_dim)))
            z_cond = Tensor(self.cond_generator.sample(self.batch_size))
            # z = torch.cat([z, z_cond], dim=1).to(self.device)
            z = torch.cat([z], dim=1).to(self.device)

            fake = self.generator(z).cpu().detach().numpy()
            data.append(fake)

        print("Sampled data length")
        print(len(data))
        data = np.concatenate(data[0:n])

        if self.scaler:
            data = self.scaler.inverse_transform(data)
        if self.decoder:
            data = self.decoder.decode(data, batch=True)
        return data