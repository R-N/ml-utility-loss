from .autoencoder import LatentTAE
from .gan import LatentGAN
from sklearn.preprocessing import StandardScaler
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
    ae,
    df,
    gan_latent_dim=16,
    gan_epochs=1,
    gan_n_critic=2,
    gan_batch_size=512,
    sample=1,
):

    # EVALUATING AUTO-ENCODER
    preprocessed = ae.preprocess(df)
    latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file

    sscaler = StandardScaler()
    sscaler.fit(latent_data)

    lat_normalized = sscaler.transform(latent_data)
    gan = LatentGAN(ae.embedding_size, latent_dim=gan_latent_dim)

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
    df,
    epochs = 1,
    batch_size=512,
    embedding_size = 64,
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
    log_columns = [],
    mixed_columns = {'capital-loss': [0.0], 'capital-gain': [0.0]},
    integer_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
):
    ae = LatentTAE(
        batch_size=batch_size,
        embedding_size = embedding_size,
        categorical_columns = categorical_columns,
        log_columns=log_columns,
        integer_columns=integer_columns,
        mixed_columns=mixed_columns, #dict(col: 0)
    )
    ae.fit_preprocessor(df)
    preprocessed = ae.preprocess(df)
    ae.fit(preprocessed, n_epochs=epochs, preprocessed=True)
    

    latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file
    reconstructed_data = ae.decode(latent_data, batch=True)
    return ae



if __name__ == "__main__":
    ae_experiment()
    latent_gan_experiment()