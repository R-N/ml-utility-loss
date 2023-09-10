import pandas as pd
from .ctabgan import Sampler, Condvec

from tqdm import tqdm
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn, optim
from .modules import FCDecoder, FCEncoder
from .process import preprocess, postprocess

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def handle_type(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if x.dim() == 1:
        x = [x]
    if isinstance(x, list):
        x = torch.stack(x)
    return x

class LatentTAE:

    def __init__(self,
                 embedding_size,
                 raw_csv_path="./data/Adult.csv",
                 test_ratio=0.20,
                 categorical_columns=['workclass', 'education', 'marital-status',
                                      'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
                 log_columns=[],
                 mixed_columns={'capital-loss': [0.0], 'capital-gain': [0.0]},
                 integer_columns=['age', 'fnlwgt', 'capital-gain',
                                  'capital-loss', 'hours-per-week'],
                 problem_type={"Classification": 'income'}):

        self.__name__ = 'AutoEncoder'
        self.raw_df = pd.read_csv(raw_csv_path)
        self.ae = AutoEncoder({"embedding_size": embedding_size, "log_interval": 5})
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.train_data = None
        self.loss = None

    def preprocess(self):
        self.data_prep, self.transformer, self.train_data = preprocess(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio
        )
        return self.train_data


    def fit(self, n_epochs, batch_size):

        self.preprocess()

        self.batch_size = batch_size

        data_dim = self.transformer.output_dim
        data_info = self.transformer.output_info

        print(f"DATA DIMENSION: {self.train_data.shape}")

        self.ae.train(self.train_data, data_dim, data_info, epochs=n_epochs, batch_size=batch_size)
        self.loss = self.ae.loss
        ##### TEST #####
        print("######## DEBUG ########")

        real = np.asarray(self.train_data[0:batch_size])

        latent = self.ae.encode(real)
        reconstructed = self.ae.decode(latent)
        l = reconstructed.cpu().detach().numpy()

        table_real = self.postprocess(real)
        table_recon = self.postprocess(l)

        print(table_real)
        print()
        print(table_recon)
        #### END OF TEST ####

    def encode(self, x):
        x = handle_type(x)
        return self.ae.encode(x)

    def decode(self, latent, batch=False):
        table = []
        batch_start = 0
        if batch:
            latent = latent if type(latent).__module__ == np.__name__ else latent.cpu().detach()

        steps = (len(latent) // self.batch_size) + 1

        for _ in range(steps):

            l = latent[batch_start: batch_start + self.batch_size]
            batch_start += self.batch_size
            if len(l) == 0: continue

            if not isinstance(l, Tensor):
                l = Tensor(l).to(self.ae.device)
            if l.device != self.ae.device:
                l = l.to(self.ae.device)

            if not batch:
                l = torch.cat(l).to(self.ae.device)
            reconstructed = self.ae.decode(l)
            reconstructed = reconstructed.cpu().detach().numpy()

            recon_inverse = self.transformer.inverse_transform(reconstructed)
            table_recon = self.data_prep.inverse_prep(recon_inverse)
            table.append(table_recon)

        return pd.concat(table)
    
    def postprocess(self, reconstructed):
        return postprocess(
            self.data_prep,
            self.transformer,
            reconstructed
        )

    def get_latent_dataset(self, as_numpy=False):

        latent_dataset = []
        print("Generating latent dataset")
        steps = (len(self.train_data) // self.batch_size) + 1
        curr = 0
        for _ in tqdm(range(steps)):
            data = self.train_data[curr : curr + self.batch_size]
            curr += self.batch_size
            if len(data) == 0: continue
            latent = self.encode(data).cpu().detach()
            latent = latent.numpy() if as_numpy else latent
            latent_dataset = [ *latent_dataset, *latent ]
        
        return np.asarray(latent_dataset) if as_numpy else torch.stack(latent_dataset)


class AENetwork(nn.Module):
    def __init__(self, args, input_dim):
        super(AENetwork, self).__init__()

        self.encoder = FCEncoder(args["embedding_size"], input_size=input_dim)
        self.decoder = FCDecoder(args["embedding_size"], input_size=input_dim)
        self.input_dim = input_dim

    def encode(self, x):
        x = handle_type(x)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        return self.decode(z)


class AutoEncoder(object):

    def __init__(self, args):
        self.args = args  # has to have 'embedding_size' and 'cuda' = True
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cond_generator = None
        self.last_loss = None

    def loss_function(self, recon_x, x, input_size):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction='sum')
        return F.mse_loss(recon_x, x.view(-1, input_size), reduction="sum")

    def encode(self, x):
        x = handle_type(x)
        x = x.to(self.device).view(-1, self.input_size).float()
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def train(self, data, output_dim: int, output_info, epochs, batch_size):

        data_sampler = Sampler(data, output_info)
        cond_generator = Condvec(data, output_info)

        col_size_d = output_dim
        self.input_size = col_size_d

        self.model = AENetwork(self.args, input_dim=col_size_d)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        train_loss = 0

        last_loss = 0

        steps = int(len(data) / batch_size)
        for e in tqdm(range(epochs)):
            for i in range(steps):
                # sample all conditional vectors for the training
                _, _, col, opt = cond_generator.sample_train(batch_size)

                # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator
                perm = np.arange(batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(batch_size, col[perm], opt[perm])

                batch = torch.from_numpy(real.astype('float32')).to(self.device)

                self.optimizer.zero_grad()

                recon_batch = self.model(batch).to(self.device)

                loss = self.loss_function(recon_batch, batch, input_size=self.input_size)
                loss.backward()

                train_loss += loss.item()
                self.optimizer.step()

                last_loss = (loss.item() / len(batch))

        print(last_loss)
        self.loss = last_loss
