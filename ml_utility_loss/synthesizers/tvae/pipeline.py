from .wrapper import TVAE

def main(
    real_data,
    discrete_columns,
    epochs=2,
    samples=100
):
    tvae = TVAE(epochs=epochs)
    tvae.fit(real_data, discrete_columns)

    # Create synthetic data
    synthetic_data = tvae.sample(samples)


def train_2(
    datasets,
    cat_features=[],
    checkpoint_dir=None,
    log_dir=None,
    trial=None,
    **kwargs
):
    if isinstance(datasets, tuple):
        train, test, *_ = datasets
    else:
        train = datasets

    for x in ["compress", "decompress"]:
        kwargs[f"{x}_dims"] = [
            kwargs[f"{x}_dims"] 
            for i in range(
                kwargs.pop(f"{x}_depth")
            )
        ]

    tvae = TVAE(**kwargs)
    tvae.fit(train, cat_features)

    return tvae