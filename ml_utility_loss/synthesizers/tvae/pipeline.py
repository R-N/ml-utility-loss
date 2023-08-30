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