import pandas as pd
from .ctabgan import Sampler, Condvec

from tqdm import tqdm
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn, optim
from .modules import FCDecoder, FCEncoder
from .preprocessing import DataPreprocessor

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def handle_type(x, device=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.Tensor(x).to(device)
    if x.dim() == 1:
        x = [x]
    if isinstance(x, list):
        x = torch.stack(x)
    if device and x.device != device:
        x = x.to(device)
    return x

class LatentTAE:

    def __init__(
        self,
        embedding_size,
        problem_type={"Classification": 'income'},
        categorical_columns = [],
        log_columns=[],
        integer_columns=[],
        mixed_columns={}, #dict(col: 0)
        batch_size=512,
        test_ratio=0.20,
    ):

        self.__name__ = 'AutoEncoder'
        self.ae = AutoEncoder({"embedding_size": embedding_size, "log_interval": 5})
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.batch_size=batch_size
        self.train_data = None
        self.loss = None
        self.data_preprocessor = DataPreprocessor(
            categorical_columns=self.categorical_columns,
            log_columns=self.log_columns,
            mixed_columns=self.mixed_columns,
            integer_columns=self.integer_columns
        )

    def fit_preprocessor(self, raw_df):
        self.data_preprocessor.fit(raw_df)

    def preprocess(self, raw_df):
        return self.data_preprocessor.preprocess(raw_df)


    def fit(self, df, n_epochs, preprocessed=False):

        if not preprocessed:
            df = self.preprocess(df)

        data_dim = self.data_preprocessor.output_dim
        data_info = self.data_preprocessor.output_info

        print(f"DATA DIMENSION: {df.shape}")

        self.ae.train(
            df, 
            data_dim, 
            data_info, 
            epochs=n_epochs, 
            batch_size=self.batch_size
        )
        self.loss = self.ae.loss
        ##### TEST #####
        print("######## DEBUG ########")

        real = np.asarray(df[0:self.batch_size])

        latent = self.ae.encode(real)
        reconstructed = self.ae.decode(latent)
        l = reconstructed.cpu().detach().numpy()

        table_real = self.postprocess(real)
        table_recon = self.postprocess(l)

        print(table_real)
        print()
        print(table_recon)
        #### END OF TEST ####

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

            l = handle_type(l, self.ae.device)

            if not batch:
                l = torch.cat(l).to(self.ae.device)

            reconstructed = self.ae.decode(l)
            reconstructed = reconstructed.cpu().detach().numpy()

            table_recon = self.postprocess(reconstructed)
            table.append(table_recon)

        return pd.concat(table)
    
    def postprocess(self, reconstructed):
        return self.data_preprocessor.postprocess(reconstructed)

    def encode(self, df, as_numpy=False, preprocessed=False):

        if not preprocessed:
            df = self.preprocess(df)

        latent_dataset = []
        print("Generating latent dataset")
        steps = (len(df) // self.batch_size) + 1
        curr = 0
        for _ in tqdm(range(steps)):
            data = df[curr : curr + self.batch_size]
            curr += self.batch_size
            if len(data) == 0: continue
            data = handle_type(data)
            latent = self.ae.encode(data).cpu().detach()
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
        x = x.to(self.device).view(-1, self.input_size).float() # 151, should be 152
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
