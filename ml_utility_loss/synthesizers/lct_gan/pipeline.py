from .autoencoder import LatentTAE
from .gan import LatentGAN
from sklearn.preprocessing import StandardScaler
import torch

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def create_gan(
    ae,
    df,
    latent_dim=16,
    epochs=1,
    n_critic=2,
    batch_size=512,
    sample=1,
):

    # EVALUATING AUTO-ENCODER
    preprocessed = ae.preprocess(df)
    latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file

    sscaler = StandardScaler()
    sscaler.fit(latent_data)

    lat_normalized = sscaler.transform(latent_data)
    gan = LatentGAN(
        ae.embedding_size, 
        transformer_output_info=ae.data_preprocessor.output_info, 
        latent_dim=latent_dim,
        batch_size=batch_size, 
        n_critic=n_critic, 
        decoder=ae,
        scaler=sscaler
    )

    gan.fit(
        lat_normalized, 
        preprocessed, 
        epochs=epochs, 
    )

    synth_df = gan.sample(sample or len(df))

    return gan, synth_df

def create_ae(
    df,
    categorical_columns=[],
    log_columns=[],
    mixed_columns={}, #dict(col: [0.0])
    integer_columns=[],
    epochs=1,
    batch_size=512,
    embedding_size=64,
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
    return ae, reconstructed_data
