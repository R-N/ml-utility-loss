from .autoencoder import LatentTAE
from .gan import LatentGAN
from sklearn.preprocessing import StandardScaler
import torch
from .params.default import AE_PARAMS, GAN_PARAMS
from ...util import filter_dict_2

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def create_gan(
    ae,
    df,
    latent_dim=16,
    epochs=1,
    n_critic=2,
    batch_size=512,
    lr=0.0002,
    sample=None,
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
        lr=lr,
        scaler=sscaler
    )

    gan.fit(
        lat_normalized, 
        preprocessed, 
        epochs=epochs, 
    )

    n = sample or len(df)
    synth_df = gan.sample(n)[:n]

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
    lr=1e-3,
):
    ae = LatentTAE(
        batch_size=batch_size,
        embedding_size = embedding_size,
        categorical_columns = categorical_columns,
        log_columns=log_columns,
        integer_columns=integer_columns,
        mixed_columns=mixed_columns, #dict(col: 0)
        lr=lr,
    )
    ae.fit_preprocessor(df)
    preprocessed = ae.preprocess(df)
    ae.fit(preprocessed, n_epochs=epochs, preprocessed=True)
    

    latent_data = ae.encode(preprocessed, preprocessed=True) # could be loaded from file
    reconstructed_data = ae.decode(latent_data, batch=True)
    return ae, reconstructed_data


def create_ae_2(
    datasets,
    cat_features=[],
    mixed_features={},
    longtail_features=[],
    integer_features=[],
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    if isinstance(datasets, tuple):
        train, test, *_ = datasets
    else:
        train = datasets

    ae_kwargs = filter_dict_2(kwargs, AE_PARAMS)

    ae, recon = create_ae(
        train,
        categorical_columns = cat_features,
        mixed_columns = mixed_features,
        integer_columns = integer_features,
        log_columns=longtail_features,
        **ae_kwargs
    )
    return ae, recon

def create_gan_2(
    ae,
    datasets,
    **kwargs
):
    if isinstance(datasets, tuple):
        train, test, *_ = datasets
    else:
        train = datasets

    gan_kwargs = filter_dict_2(kwargs, GAN_PARAMS)
    gan, synth = create_gan (
        ae, train,
        sample=None,
        **gan_kwargs
    )
    return gan, synth
