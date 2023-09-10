from .preprocessing import DataPrep, DataTransformer

def preprocess(
    raw_df,
    categorical_columns=[],
    log_columns=[],
    mixed_columns=[],
    integer_columns=[],
    problem_type=[],
    test_ratio=0,
    data_prep=None,
    transformer=None,
):
    if not data_prep:
        data_prep = DataPrep(
            raw_df,
            categorical_columns,
            log_columns,
            mixed_columns,
            integer_columns,
            problem_type,
            test_ratio
        )

    if not transformer:
        transformer = DataTransformer(
            train_data=data_prep.df,
            categorical_list=data_prep.column_types["categorical"],
            mixed_dict=data_prep.column_types["mixed"]
        )
        transformer.fit()

    train_data = transformer.transform(data_prep.df.values)
    return data_prep, transformer, train_data

def postprocess(
    data_prep,
    transformer,
    reconstructed
):
    recon_inverse = transformer.inverse_transform(reconstructed)
    table_recon = data_prep.inverse_prep(recon_inverse)
    return table_recon
