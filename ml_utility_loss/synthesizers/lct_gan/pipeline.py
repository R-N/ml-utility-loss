from .autoencoder import LatentTAE
from .gan import LatentGAN
from ...scalers import StandardScaler
import torch
from .params.default import AE_PARAMS, GAN_PARAMS
from ...util import filter_dict_2
import os

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
    mlu_trainer=None,
    train=True,
    g_state_path=None,
    d_state_path=None,
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
        scaler=sscaler,
        mlu_trainer=mlu_trainer,
    )

    if not train:
        return gan, None
    if g_state_path and os.path.exists(g_state_path):
        gan.generator.load_state_dict(torch.load(g_state_path))
        if d_state_path and os.path.exists(d_state_path):
            gan.discriminator.load_state_dict(torch.load(d_state_path))
    else:
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
    mlu_trainer=None,
    preprocess_df=None,
    train=True,
    state_path=None
):
    preprocess_df = preprocess_df if preprocess_df is not None else df
    ae = LatentTAE(
        batch_size=batch_size,
        embedding_size = embedding_size,
        categorical_columns = categorical_columns,
        log_columns=log_columns,
        integer_columns=integer_columns,
        mixed_columns=mixed_columns, #dict(col: 0)
        lr=lr,
        mlu_trainer=mlu_trainer,
    )
    ae.fit_preprocessor(preprocess_df)
    
    if not train:
        return ae, None

    if state_path and os.path.exists(state_path):
        ae.ae.model.load_state_dict(torch.load(state_path))
    else:
        preprocessed = ae.preprocess(df)
        ae.fit(
            preprocessed, 
            n_epochs=epochs, 
            preprocessed=True, 
        )
    

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
    mlu_trainer=None,
    preprocess_df=None,
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
        mlu_trainer=mlu_trainer,
        preprocess_df=preprocess_df,
        **ae_kwargs
    )
    return ae, recon

def create_gan_2(
    ae,
    datasets,
    mlu_trainer=None,
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
        mlu_trainer=mlu_trainer,
        **gan_kwargs
    )
    return gan, synth
