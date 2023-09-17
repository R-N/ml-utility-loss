from .autoencoder import LatentTAE
from .gan import LatentGAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

import pickle
from tqdm import tqdm

import time
import torch
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

experiment_params = {
    "best_ae" : 1, # steps
    "embedding_size": 64,
    "raw_csv_path": "./data/Adult.csv",
    "test_ratio": 0.20, # this actually affects transformer dim
    "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
    "log_columns": [],
    "mixed_columns" : {'capital-loss': [0.0], 'capital-gain': [0.0]},
    "integer_columns" : ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
    "problem_type": {"Classification": 'income'}
}

def latent_gan_experiment(
    gan_latent_dim=16,
    gan_epochs=1,
    gan_n_critic=2,
    gan_batch_size=512,
    sample=1
):

    exp = dict(experiment_params)

    dataset_path = exp.pop("raw_csv_path")
    best_ae = exp["best_ae"]
    del exp["best_ae"]

    pickle_path = "./ae_pickles/" + dataset_path.replace("./data/", "").replace(".csv", f"_ae{exp['embedding_size']}_{best_ae}.pickle")

    print(f"Opening {pickle_path}")
    ae_pf = open(pickle_path, 'rb')
    ae = pickle.load(ae_pf)
    ae_pf.close()

    raw_df = pd.read_csv(dataset_path)

    # EVALUATING AUTO-ENCODER
    preprocessed = ae.preprocess(raw_df)
    latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file

    sscaler = StandardScaler()
    sscaler.fit(latent_data)

    lat_normalized = sscaler.transform(latent_data)
    gan = LatentGAN(exp["embedding_size"], latent_dim=gan_latent_dim)

    gan.fit(
        lat_normalized, 
        preprocessed, 
        transformer_output_info=ae.data_preprocessor.output_info, 
        epochs=gan_epochs, 
        batch_size=gan_batch_size, 
        n_critic=gan_n_critic, 
    )

    lat_normalized_synth = gan.sample(sample or len(latent_data))
    latent_data_synth = sscaler.inverse_transform(lat_normalized_synth)
    synth_df = ae.decode(latent_data_synth, batch=True)
    return synth_df

def ae_experiment(
    epochs = 1,
    ae_batch_size=512,
):

    exp = dict(experiment_params)
    dataset_path = exp.pop("raw_csv_path")
    
    raw_df = pd.read_csv(dataset_path)

    ae = LatentTAE(batch_size=ae_batch_size, **exp)
    ae.fit_preprocessor(raw_df)
    preprocessed = ae.preprocess(raw_df)
    ae.fit(preprocessed, n_epochs=epochs, preprocessed=True)
    

    latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file
    reconstructed_data = ae.decode(latent_data, batch=True)

    return ae



if __name__ == "__main__":
    ae_experiment()
    latent_gan_experiment()