from .modules import Encoder, Decoder
import torch
from torch.nn import Module

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TVAEModel(Module):
    def __init__(self, 
        data_dim, 
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        device=DEFAULT_DEVICE
    ):
        super(TVAEModel, self).__init__()
        self.data_dim = data_dim
        self.embedding_dim = embedding_dim
        self.compress_dims=compress_dims
        self.decompress_dims=decompress_dims
        self.device=device
        self.encoder = Encoder(data_dim, compress_dims, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, decompress_dims, data_dim).to(device)
        
    def forward(self, real):
        mu, std, logvar = self.encoder(real)
        eps = torch.randn_like(std)
        emb = eps * std + mu
        rec, sigmas = self.decoder(emb)
        return rec, sigmas, mu, logvar
