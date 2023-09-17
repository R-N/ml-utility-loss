from .autoencoder import LatentTAE
from .gan import LatentGAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

import pickle
from tqdm import tqdm

import time
import torch
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

experiment_params = [
    {
        "best_ae" : 1, # steps
        "embedding_size": 64,
        "raw_csv_path": "./data/Adult.csv",
        "test_ratio": 0.20, # this actually affects transformer dim
        "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
        "log_columns": [],
        "mixed_columns" : {'capital-loss': [0.0], 'capital-gain': [0.0]},
        "integer_columns" : ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
        "problem_type": {"Classification": 'income'}
    },
]

def latent_gan_experiment(
        bottleneck=64,
        ae_epochs=1,
        ae_batch_size=512,
        gan_latent_dim=16,
        gan_epochs=1,
        gan_n_critic=2,
        gan_batch_size=512):

    for exp in experiment_params[:1]:
        exp = dict(exp)

        dataset_path = exp.pop("raw_csv_path")
        dataset_categories = exp["categorical_columns"]
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

        real_path = dataset_path
        decoded_path = dataset_path.replace("./data/", "./data/decoded/").replace(".csv", f"_decoded{exp['embedding_size']}_test.csv")

        reconstructed_data = ae.decode(latent_data, batch=True)
        reconstructed_data.to_csv(decoded_path, index=False)

        real_path = real_path
        fake_paths = [ decoded_path ]

        sscaler = StandardScaler()
        sscaler.fit(latent_data)

        lat_normalized = sscaler.transform(latent_data)
        gan = LatentGAN(exp["embedding_size"], latent_dim=gan_latent_dim)

        def measure(x):
            df = gan.sample(len(latent_data), ae, sscaler)
            df.to_csv(dataset_path.replace(".csv", "_fake.csv"), index=False)
            print(df)

        gan.fit(
            lat_normalized, 
            preprocessed, 
            transformer_output_info=ae.data_preprocessor.output_info, 
            epochs=gan_epochs, 
            batch_size=gan_batch_size, 
            n_critic=gan_n_critic, 
            callback=measure
        )

        gan_pf = open("./gan_pickles/" + dataset_path.replace("./data/", "").replace(".csv", f"_gan{gan_latent_dim}_{gan_epochs}.pickle"), 'wb')
        pickle.dump(gan, gan_pf)
        gan_pf.close()

def ae_experiment(
    epochs = 1,
    ae_batch_size=512,
):

    
    for exp in experiment_params[:1]:
        exp = dict(exp)

        dataset_path = exp.pop("raw_csv_path")
        best_ae = exp["best_ae"]
        del exp["best_ae"]
        
        print(f"Training on {dataset_path}")
        start_time = time.time()
        raw_df = pd.read_csv(dataset_path)

        ae = LatentTAE(**exp)
        ae.fit_preprocessor(raw_df)
        preprocessed = ae.preprocess(raw_df)
        latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file
        ae.fit(raw_df, n_epochs=epochs, batch_size=ae_batch_size)
        time_to_train = time.time() - start_time
        print("--- %s seconds ---" % (time_to_train))

        ae_pf = open("./ae_pickles/" + dataset_path.replace("./data/", "").replace(".csv", f"_ae{exp['embedding_size']}_{epochs}.pickle"), 'wb')
        pickle.dump(ae, ae_pf)
        ae_pf.close()

        real_path = dataset_path
        decoded_path = dataset_path.replace("./data/", "./data/decoded/").replace(".csv", f"_decoded{exp['embedding_size']}_{epochs}.csv")

        reconstructed_data = ae.decode(latent_data, batch=True)

        reconstructed_data.to_csv(decoded_path, index=False)


if __name__ == "__main__":
    ae_experiment()
    latent_gan_experiment()