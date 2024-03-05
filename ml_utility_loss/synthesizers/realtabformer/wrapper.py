"""The REaLTabFormer implements the model training and data processing
for tabular and relational data.
"""
import json
import logging
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics.pairwise import manhattan_distances
from .data_utils import map_input_ids
from torchinfo import summary

# from sklearn.metrics import accuracy_score
from transformers import ( 
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

import ml_utility_loss.realtabformer as realtabformer

from .data_utils import (
    ModelFileName,
    ModelType,
    SpecialTokens,
    TabularArtefact,
    build_vocab,
    make_dataset,
    make_dataset_2,
    process_data,
)
from .rtf_analyze import SyntheticDataBench
from .rtf_exceptions import SampleEmptyLimitError
from .rtf_sampler import TabularSampler
from .rtf_trainer import ResumableTrainer
from .rtf_validators import ObservationValidator
from .util import validate_get_device


class REaLTabFormer:
    def __init__(
        self,
        model_type: str = ModelType.tabular,
        tabular_config: Optional[GPT2Config] = None,
        checkpoints_dir: str = "rtf_checkpoints",
        samples_save_dir: str = "rtf_samples",
        epochs: int = 100,
        batch_size: int = 8,
        random_state: int = 1029,
        train_size: float = 1,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0,
        mask_rate: float = 0,
        numeric_nparts: int = 1,
        numeric_precision: int = 4,
        numeric_max_len: int = 10,
        mlu_trainer=None,
        **training_args_kwargs,
    ) -> None:
        self.model: PreTrainedModel = None
        self.mlu_trainer = mlu_trainer

        # This will be set during and will also be deleted after training.
        self.dataset = None

        if model_type not in ModelType.types():
            self._invalid_model_type(model_type)

        self.model_type = model_type

        if self.model_type == ModelType.tabular:
            self._init_tabular(tabular_config)
        #elif self.model_type == ModelType.relational:
        #    raise Exception("Relational")
        else:
            self._invalid_model_type(self.model_type)

        self.checkpoints_dir = Path(checkpoints_dir)
        self.samples_save_dir = Path(samples_save_dir)
        self.epochs = epochs
        self.batch_size = batch_size

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

        self.training_args_kwargs = dict(
            evaluation_strategy="steps",
            output_dir=self.checkpoints_dir.as_posix(),
            metric_for_best_model="loss",  # This will be replaced with "eval_loss" if `train_size` < 1
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            remove_unused_columns=True,
            logging_steps=100,
            save_steps=100,
            eval_steps=100,
            load_best_model_at_end=True,
            save_total_limit=2,
            optim="adamw_torch",
            report_to="none",
        )

        # Remove experiment params from `training_args_kwargs`
        for p in [
            "output_dir",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
        ]:
            if p in training_args_kwargs:
                warnings.warn(
                    f"Argument {p} was passed in training_args_kwargs but will be ignored..."
                )
                training_args_kwargs.pop(p)

        self.training_args_kwargs.update(training_args_kwargs)

        self.train_size = train_size
        self.mask_rate = mask_rate

        self.columns: List[str] = []
        self.column_dtypes: Dict[str, type] = {}
        self.column_has_missing: Dict[str, bool] = {}
        self.drop_na_cols: List[str] = []
        self.processed_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.datetime_columns: List[str] = []
        self.vocab: Dict[str, dict] = {}
        # Output length for generator model
        # including special tokens.
        self.tabular_max_length = None
        #self.relational_max_length = None
        # Number of derived columns for the relational
        # and tabular data after performing the data transformation.
        # This will be used as record size validator in the
        # sampling stage.
        self.tabular_col_size = None
        #self.relational_col_size = None

        # This stores the transformation
        # parameters for numeric columns.
        self.col_transform_data: Optional[Dict] = None

        # This is the col_transform_data
        # for the relational models's in_df.
        self.in_col_transform_data: Optional[Dict] = None

        self.col_idx_ids: Dict[int, list] = {}

        self.random_state = random_state

        self.numeric_nparts = numeric_nparts
        self.numeric_precision = numeric_precision
        self.numeric_max_len = numeric_max_len

        # A unique identifier for the experiment set after the
        # model is trained.
        self.experiment_id = None
        self.trainer_state = None

        # Target column, when set, a copy of the column values will be
        # implicitly placed at the beginning of the dataframe.
        self.target_col = None

        self.realtabformer_version = realtabformer.__version__

    def _invalid_model_type(self, model_type):
        raise ValueError(
            f"Model type: {model_type} is not valid. REaLTabFormer in this codebase only supports `tabular`."
        )

    def _init_tabular(self, tabular_config):
        if tabular_config is not None:
            warnings.warn(
                "The `bos_token_id`, `eos_token_id`, and `vocab_size` attributes will \
                    be replaced when the `.fit` method is run."
            )
        else:
            # Default is 12, use 6 for distill-gpt2 as default
            tabular_config = GPT2Config(n_layer=6)

        self.tabular_config = tabular_config
        self.model = None


    def _extract_column_info(self, df: pd.DataFrame) -> None:
        # Track the column order of the original data
        self.columns = df.columns.to_list()

        # Store the dtypes of the columns
        self.column_dtypes = df.dtypes.astype(str).to_dict()

        # Track which variables have missing values
        self.column_has_missing = (df.isnull().sum() > 0).to_dict()

        # Get the columns where there should be no missing values
        self.drop_na_cols = [
            col for col, has_na in self.column_has_missing.items() if not has_na
        ]

        # Identify the numeric columns. These will undergo
        # special preprocessing.
        self.numeric_columns = df.select_dtypes(include=np.number).columns.to_list()

        # Identify the datetime columns. These will undergo
        # special preprocessing.
        self.datetime_columns = df.select_dtypes(include="datetime").columns.to_list()

    def _generate_vocab(self, df: pd.DataFrame) -> dict:
        return build_vocab(df, special_tokens=SpecialTokens.tokens(), add_columns=False)

    def _check_model(self):
        assert self.model is not None, "Model is None. Train the model first!"

    def _split_train_eval_dataset(self, dataset: Dataset):
        test_size = 1 - self.train_size
        if test_size > 0:
            dataset = dataset.train_test_split(
                test_size=test_size, seed=self.random_state
            )
            dataset["train_dataset"] = dataset.pop("train")
            dataset["eval_dataset"] = dataset.pop("test")

            # Override `metric_for_best_model` from "loss" to "eval_loss"
            self.training_args_kwargs["metric_for_best_model"] = "eval_loss"
            # Make this explicit so that no assumption is made on the
            # direction of the metric improvement.
            self.training_args_kwargs["greater_is_better"] = False
        else:
            dataset = dict(train_dataset=dataset)
            self.training_args_kwargs["evaluation_strategy"] = "no"
            self.training_args_kwargs["load_best_model_at_end"] = False

        return dataset

    def fit(
        self,
        df: pd.DataFrame,
        #in_df: Optional[pd.DataFrame] = None,
        #join_on: Optional[str] = None,
        resume_from_checkpoint: Union[bool, str] = False,
        device="cuda",
        num_bootstrap: int = 500,
        frac: float = 0.165,
        frac_max_data: int = 10000,
        qt_max: Union[str, float] = 0.05,
        qt_max_default: float = 0.05,
        qt_interval: int = 100,
        qt_interval_unique: int = 100,
        distance: manhattan_distances = manhattan_distances,
        quantile: float = 0.95,
        n_critic: int = 5,
        n_critic_stop: int = 2,
        gen_rounds: int = 3,
        sensitivity_max_col_nums: int = 20,
        use_ks: bool = False,
        full_sensitivity: bool = False,
        sensitivity_orig_frac_multiple: int = 4,
        orig_samples_rounds: int = 5,
        load_from_best_mean_sensitivity: bool = False,
        target_col: str = None,
        fit_preprocess=True,
    ):
        assert len(df.index.unique()) == len(df.index), "Index must be unique"

        device = validate_get_device(device)

        # Set target col for teacher forcing
        self.target_col = target_col

        # Set the seed for, *hopefully*, replicability.
        # This may cause an unexpected behavior when using
        # the resume_from_checkpoint option.
        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        if self.model_type == ModelType.tabular:
            if n_critic <= 0:
                trainer = self._fit_tabular(df, device=device, fit_preprocess=fit_preprocess)
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer = self._train_with_sensitivity(
                    df,
                    device,
                    num_bootstrap=num_bootstrap,
                    frac=frac,
                    frac_max_data=frac_max_data,
                    qt_max=qt_max,
                    qt_max_default=qt_max_default,
                    qt_interval=qt_interval,
                    qt_interval_unique=qt_interval_unique,
                    distance=distance,
                    quantile=quantile,
                    n_critic=n_critic,
                    n_critic_stop=n_critic_stop,
                    gen_rounds=gen_rounds,
                    resume_from_checkpoint=resume_from_checkpoint,
                    sensitivity_max_col_nums=sensitivity_max_col_nums,
                    use_ks=use_ks,
                    full_sensitivity=full_sensitivity,
                    sensitivity_orig_frac_multiple=sensitivity_orig_frac_multiple,
                    orig_samples_rounds=orig_samples_rounds,
                    load_from_best_mean_sensitivity=load_from_best_mean_sensitivity,
                    fit_preprocess=fit_preprocess
                )

            del self.dataset

        #elif self.model_type == ModelType.relational:
        #    raise Exception("Relational")
        else:
            self._invalid_model_type(self.model_type)

        try:
            self.experiment_id = f"id{int((time.time() * 10 ** 10)):024}"
            torch.cuda.empty_cache()

            return trainer
        except Exception as exception:
            if device == torch.device("cuda"):
                del self.model
                torch.cuda.empty_cache()
                self.model = None

            raise exception

    def _train_with_sensitivity(
        self,
        df: pd.DataFrame,
        device: str = "cuda",
        num_bootstrap: int = 500,
        frac: float = 0.165,
        frac_max_data: int = 10000,
        qt_max: Union[str, float] = 0.05,
        qt_max_default: float = 0.05,
        qt_interval: int = 100,
        qt_interval_unique: int = 100,
        distance: manhattan_distances = manhattan_distances,
        quantile: float = 0.95,
        n_critic: int = 5,
        n_critic_stop: int = 2,
        gen_rounds: int = 3,
        sensitivity_max_col_nums: int = 20,
        use_ks: bool = False,
        resume_from_checkpoint: Union[bool, str] = False,
        full_sensitivity: bool = False,
        sensitivity_orig_frac_multiple: int = 4,
        orig_samples_rounds: int = 5,
        load_from_best_mean_sensitivity: bool = False,
        fit_preprocess=True,
    ):
        assert gen_rounds >= 1

        _frac = min(frac, frac_max_data / len(df))
        if frac != _frac:
            warnings.warn(
                f"The frac ({frac}) set results to a sample larger than \
                    frac_max_data={frac_max_data}. Setting frac to {_frac}."
            )
            frac = _frac

        trainer: Trainer = None
        dup_rate = df.duplicated().mean()

        if isinstance(qt_max, str):
            if qt_max == "compute":
                # The idea behind this is if the empirical has
                # natural duplicates, we can use that as
                # basis for what a typical rate for duplicates a
                # random sample should have. Any signidican excess
                # from this indicates overfitting.
                # The choice of dividing the duplicate rate by 2
                # is arbitrary but reasonable to prevent delayed
                # stopping when overfitting.
                dup_rate = dup_rate / 2
                qt_max = dup_rate if dup_rate > 0 else qt_max_default
            else:
                raise ValueError(f"Unexpected qt_max value: {qt_max}")
        elif not isinstance(qt_max, str) and dup_rate >= qt_max:
            warnings.warn(
                f'The qt_max ({qt_max}) set is lower than the duplicate \rate ({dup_rate}) in \
                    the data. This will not give a reliable early stopping condition. Consider \
                        using qt_max="compute" argument.'
            )

        if dup_rate == 0:
            # We do this because for data without unique values, we
            # expect that a generated sample should have equal likelihood
            # in the minimum distance with the hold out.
            warnings.warn(
                f"Duplicate rate ({dup_rate}) in the data is zero. The `qt_interval` will be set \
                    to qt_interval_unique={qt_interval_unique}."
            )
            qt_interval = qt_interval_unique

        # Estimate the sensitivity threshold
        print("Computing the sensitivity threshold...")

        if not full_sensitivity:
            # Dynamically compute the qt_interval to fit the data
            # if the resulting sample has lower resolution.
            # For example, we can't use qt_interval=1000 if the number
            # of samples left at qt_max of the distance matrix is less than
            # 1000.
            # The formula means:
            # - 2       -> accounts for the fact that we concatenate the rows and columns
            #             of the distance matrix.
            # - frac    -> the proportion of the training data that is used to compute the
            #             the distance matrix.
            # - qt_max  -> the maximum quantile of assessment.
            # We divide by 2 to increase the resolution a bit
            _qt_interval = min(qt_interval, (2 * frac * len(df) * qt_max) // 2)
            _qt_interval = max(_qt_interval, 2)
            _qt_interval = int(_qt_interval)

            if _qt_interval < qt_interval:
                warnings.warn(
                    f"qt_interval adjusted from {qt_interval} to {_qt_interval}..."
                )
                qt_interval = _qt_interval

        # Computing this here before splitting may have some data
        # leakage issue, but it should be almost negligible. Doing
        # the computation of the threshold on the full data with the
        # train size aligned will give a more reliable estimate of
        # the sensitivity threshold.
        sensitivity_values = SyntheticDataBench.compute_sensitivity_threshold(
            train_data=df,
            num_bootstrap=num_bootstrap,
            # Divide by two so that the train_data in this computation matches the size
            # of the final df used to train the model. This is essential so that the
            # sensitivity_threshold value is consistent with the val_sensitivity.
            # Concretely, the computation of the distribution of min distances is
            # relative to the number of training observations.
            # The `frac` in  this method corresponds to the size of both the test and the
            # synthetic samples.
            frac=frac / 2,
            qt_max=qt_max,
            qt_interval=qt_interval,
            distance=distance,
            return_values=True,
            quantile=quantile,
            max_col_nums=sensitivity_max_col_nums,
            use_ks=use_ks,
            full_sensitivity=full_sensitivity,
            sensitivity_orig_frac_multiple=sensitivity_orig_frac_multiple,
        )
        sensitivity_threshold = np.quantile(sensitivity_values, quantile)
        mean_sensitivity_value = np.mean(sensitivity_values)
        best_mean_sensitivity_value = np.inf

        assert isinstance(sensitivity_threshold, float)
        print("Sensitivity threshold:", sensitivity_threshold, "qt_max:", qt_max)

        # # Create a hold out sample for the discriminator model
        # hold_df = df.sample(frac=frac, random_state=self.random_state)
        # df = df.loc[df.index.difference(hold_df.index)]

        # Start training
        logging.info("Start training...")

        # Remove existing checkpoints
        for chkp in self.checkpoints_dir.glob("checkpoint-*"):
            shutil.rmtree(chkp, ignore_errors=True)

        sensitivity_scores = []
        bdm_path = self.checkpoints_dir / TabularArtefact.best_disc_model
        mean_closest_bdm_path = (
            self.checkpoints_dir / TabularArtefact.mean_best_disc_model
        )
        not_bdm_path = self.checkpoints_dir / TabularArtefact.not_best_disc_model
        last_epoch_path = self.checkpoints_dir / TabularArtefact.last_epoch_model

        # Remove existing artefacts in the best model dir
        shutil.rmtree(bdm_path, ignore_errors=True)
        bdm_path.mkdir(parents=True, exist_ok=True)

        shutil.rmtree(mean_closest_bdm_path, ignore_errors=True)
        mean_closest_bdm_path.mkdir(parents=True, exist_ok=True)

        shutil.rmtree(not_bdm_path, ignore_errors=True)
        not_bdm_path.mkdir(parents=True, exist_ok=True)

        shutil.rmtree(last_epoch_path, ignore_errors=True)
        last_epoch_path.mkdir(parents=True, exist_ok=True)

        last_epoch = 0
        not_best_val_sensitivity = np.inf

        if resume_from_checkpoint:
            chkp_list = sorted(
                self.checkpoints_dir.glob("checkpoint-*"), key=os.path.getmtime
            )
            if chkp_list:
                # Get the most recent checkpoint based on
                # creation time.
                chkp = chkp_list[-1]
                trainer_state = json.loads((chkp / "trainer_state.json").read_text())
                last_epoch = math.ceil(trainer_state["epoch"])

                trainer = self._fit_tabular(
                    df,
                    device=device,
                    num_train_epochs=last_epoch,
                    target_epochs=self.epochs,
                    fit_preprocess=fit_preprocess
                )

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        for p_epoch in range(last_epoch, self.epochs, n_critic):
            gen_total = int(len(df) * frac)

            num_train_epochs = min(p_epoch + n_critic, self.epochs)
            # Perform the discriminator sampling every `n_critic` epochs
            if trainer is None:
                trainer = self._fit_tabular(
                    df,
                    device=device,
                    num_train_epochs=num_train_epochs,
                    target_epochs=self.epochs,
                    fit_preprocess=fit_preprocess
                )
                trainer.train(resume_from_checkpoint=False)
            else:
                trainer = self._build_tabular_trainer(
                    device=device,
                    num_train_epochs=num_train_epochs,
                    target_epochs=self.epochs,
                )
                trainer.train(resume_from_checkpoint=True)

            try:
                # Generate samples
                gen_df = self.sample(n_samples=gen_rounds * gen_total, device=device)
            except SampleEmptyLimitError:
                # Continue training if the model is still not
                # able to generate stable observations.
                continue

            val_sensitivities = []

            if full_sensitivity:
                for _ in range(gen_rounds):
                    hold_df = df.sample(n=gen_total)

                    for g_idx in range(gen_rounds):
                        val_sensitivities.append(
                            SyntheticDataBench.compute_sensitivity_metric(
                                original=df.loc[df.index.difference(hold_df.index)],
                                synthetic=gen_df.iloc[
                                    g_idx * gen_total : (g_idx + 1) * gen_total
                                ],
                                test=hold_df,
                                qt_max=qt_max,
                                qt_interval=qt_interval,
                                distance=distance,
                                max_col_nums=sensitivity_max_col_nums,
                                use_ks=use_ks,
                            )
                        )
            else:
                for g_idx in range(gen_rounds):
                    for _ in range(orig_samples_rounds):
                        original_df = df.sample(
                            n=sensitivity_orig_frac_multiple * gen_total, replace=False
                        )
                        hold_df = df.loc[df.index.difference(original_df.index)]
                        hold_df = hold_df.sample(
                            n=gen_total, replace=False
                        )

                        val_sensitivities.append(
                            SyntheticDataBench.compute_sensitivity_metric(
                                original=original_df,
                                synthetic=gen_df.iloc[
                                    g_idx * gen_total : (g_idx + 1) * gen_total
                                ],
                                test=hold_df,
                                qt_max=qt_max,
                                qt_interval=qt_interval,
                                distance=distance,
                                max_col_nums=sensitivity_max_col_nums,
                                use_ks=use_ks,
                            )
                        )

            val_sensitivity = np.mean(val_sensitivities)

            sensitivity_scores.append(val_sensitivity)

            if val_sensitivity < sensitivity_threshold:
                # Just save the model while the
                # validation sensitivity is still within
                # the accepted range.
                # This way we can load the acceptable
                # model back when the threshold is breached.
                trainer.save_model(bdm_path.as_posix())
                trainer.state.save_to_json((bdm_path / "trainer_state.json").as_posix())

            elif not_best_val_sensitivity > (val_sensitivity - sensitivity_threshold):
                print("Saving not-best model...")
                trainer.save_model(not_bdm_path.as_posix())
                trainer.state.save_to_json(
                    (not_bdm_path / "trainer_state.json").as_posix()
                )
                not_best_val_sensitivity = val_sensitivity - sensitivity_threshold

            _delta_mean_sensitivity_value = abs(
                mean_sensitivity_value - val_sensitivity
            )

            if _delta_mean_sensitivity_value < best_mean_sensitivity_value:
                best_mean_sensitivity_value = _delta_mean_sensitivity_value
                trainer.save_model(mean_closest_bdm_path.as_posix())
                trainer.state.save_to_json(
                    (mean_closest_bdm_path / "trainer_state.json").as_posix()
                )

            print(
                f"Critic round: {p_epoch + n_critic}, \
                    sensitivity_threshold: {sensitivity_threshold}, \
                        val_sensitivity: {val_sensitivity}, \
                            val_sensitivities: {val_sensitivities}"
            )

            if len(sensitivity_scores) > n_critic_stop:
                n_no_improve = 0
                for sensitivity_score in sensitivity_scores[-n_critic_stop:]:
                    # We count no improvement if the score is not
                    # better than the best, and that the score is not
                    # better than the previous score.
                    if sensitivity_score > sensitivity_threshold:
                        n_no_improve += 1

                if n_no_improve == n_critic_stop:
                    print("Stopping training, no improvement in critic...")
                    break

        # Save last epoch artefacts before loading the best model.
        trainer.save_model(last_epoch_path.as_posix())
        trainer.state.save_to_json((last_epoch_path / "trainer_state.json").as_posix())

        loaded_model_path = None

        if not load_from_best_mean_sensitivity:
            if (bdm_path / "pytorch_model.bin").exists():
                loaded_model_path = bdm_path
        else:
            if (mean_closest_bdm_path / "pytorch_model.bin").exists():
                loaded_model_path = mean_closest_bdm_path

        if loaded_model_path is None:
            # There should always be at least one `mean_closest_bdm_path` but
            # in case it doesn't exist, try loading from `not_bdm_path`.
            warnings.warn(
                "No best model was saved. Loading the closest model to the sensitivity_threshold."
            )
            loaded_model_path = not_bdm_path

        self.model = self.model.from_pretrained(loaded_model_path.as_posix())
        self.trainer_state = json.loads(
            (loaded_model_path / "trainer_state.json").read_text()
        )

        return trainer
    
    def fit_preprocess(self, df):
        self._extract_column_info(df)
        df, self.col_transform_data = process_data(
            df,
            numeric_max_len=self.numeric_max_len,
            numeric_precision=self.numeric_precision,
            numeric_nparts=self.numeric_nparts,
            target_col=self.target_col,
        )
        #print("col_transform_data", self.col_transform_data)
        self.vocab = self._generate_vocab(df)
        #print("vocab", self.vocab)
        self.processed_columns = df.columns.to_list()
        #print("processed_columns", len(self.processed_columns), self.processed_columns)
        self.tabular_col_size = df.shape[0]
        #print("tabular_col_size", self.tabular_col_size)

        # NOTE: the index starts at zero, but should be adjusted
        # to account for the special tokens. For tabular data,
        # the index should start at 1.
        self.col_idx_ids = {
            ix: self.vocab["column_token_ids"][col]
            for ix, col in enumerate(self.processed_columns)
        }
        #print("col_idx_ids", self.col_idx_ids)

    
    def preprocess(self, df, fit=True):
        if fit:
            self.fit_preprocess(df)
        df, col_transform_data = process_data(
            df,
            numeric_max_len=self.numeric_max_len,
            numeric_precision=self.numeric_precision,
            numeric_nparts=self.numeric_nparts,
            target_col=self.target_col,
            col_transform_data=self.col_transform_data,
        )
        #print("col_transform_data", col_transform_data)

        processed_columns = df.columns.to_list()
        #print(len(processed_columns), processed_columns)

        return df
    
    def map_input_ids(self, df):
        df = map_input_ids(
            df, 
            self.vocab, mask_rate=self.mask_rate, return_token_type_ids=False,
            remove_columns=df.columns
        )
        self.tabular_max_length = len(df["input_ids"].iloc[0])
        return df
    
    def make_dataset(self, df, preprocess=True, fit_preprocess=True, map_ids=True, two=True):
        if preprocess:
            df = self.preprocess(df, fit=fit_preprocess)
        # Load the dataframe into a HuggingFace Dataset
        f = make_dataset_2 if two else make_dataset
        dataset = f(
            df, 
            self.vocab, mask_rate=self.mask_rate, return_token_type_ids=False, 
            map_ids=map_ids
        )

        # Store the sequence length for the processed data
        self.tabular_max_length = len(dataset["input_ids"][0])
        return dataset

    def _fit_tabular(
        self,
        df: pd.DataFrame,
        device="cuda",
        num_train_epochs: int = None,
        target_epochs: int = None,
        fit_preprocess=True,
    ) -> Trainer:
        dataset = self.make_dataset(df, fit_preprocess=fit_preprocess)

        # Create train-eval split if specified
        self.dataset = self._split_train_eval_dataset(dataset)

        # Set up the config and the model
        self.tabular_config.bos_token_id = self.vocab["token2id"][SpecialTokens.BOS]
        self.tabular_config.eos_token_id = self.vocab["token2id"][SpecialTokens.EOS]
        self.tabular_config.vocab_size = len(self.vocab["id2token"])

        # Make sure that we have at least the number of
        # columns in the transformed data as positions.
        if self.tabular_config.n_positions < len(self.vocab["column_token_ids"]):
            self.tabular_config.n_positions = 128 + len(self.vocab["column_token_ids"])

        self.model = GPT2LMHeadModel(self.tabular_config)

        # Tell pytorch to run this model on the GPU.
        device = torch.device(device)
        if device == torch.device("cuda"):
            print("Model is cuda! Device is", device)
            self.model.cuda()
        else:
            print("Model is not cuda! Device is", device)
        print(summary(self.model, input_size=(3, 24,), depth=3, dtypes=['torch.IntTensor'], device="cpu"))

        return self._build_tabular_trainer(
            device=device,
            num_train_epochs=num_train_epochs,
            target_epochs=target_epochs,
        )

    def _build_tabular_trainer(
        self,
        device="cuda",
        num_train_epochs: int = None,
        target_epochs: int = None,
    ) -> Trainer:
        device = torch.device(device)

        # Set TrainingArguments and the Trainer
        logging.info("Set up the TrainingArguments and the Trainer...")
        training_args_kwargs: Dict[str, Any] = dict(self.training_args_kwargs)

        default_args_kwargs = dict(
            fp16=(
                device == torch.device("cuda")
            ),  # Use fp16 by default if using cuda device
        )

        for k, v in default_args_kwargs.items():
            if k not in training_args_kwargs:
                training_args_kwargs[k] = v

        if num_train_epochs is not None:
            training_args_kwargs["num_train_epochs"] = num_train_epochs

        # # NOTE: The `ResumableTrainer` will default to its original
        # # behavior (Trainer) if `target_epochs`` is None.
        # # Set the `target_epochs` to `num_train_epochs` if not specified.
        # if target_epochs is None:
        #     target_epochs = training_args_kwargs.get("num_train_epochs")

        callbacks = None
        if training_args_kwargs["load_best_model_at_end"]:
            callbacks = [
                EarlyStoppingCallback(
                    self.early_stopping_patience, self.early_stopping_threshold
                )
            ]

        #disable wandb
        training_args_kwargs["report_to"] = "none"

        assert self.dataset
        trainer = ResumableTrainer(
            target_epochs=target_epochs,
            save_epochs=None,
            model=self.model,
            args=TrainingArguments(**training_args_kwargs),
            data_collator=None,  # Use the default_data_collator
            callbacks=callbacks,
            mlu_trainer=self.mlu_trainer,
            sampler=self,
            batch_size=self.batch_size,
            **self.dataset,
        )

        return trainer

    def sample(
        self,
        n_samples: int = None,
        #input_unique_ids: Optional[Union[pd.Series, List]] = None,
        #input_df: Optional[pd.DataFrame] = None,
        #input_ids: Optional[torch.tensor] = None,
        gen_batch: Optional[int] = 128,
        device: str = "cuda",
        seed_input: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        save_samples: Optional[bool] = False,
        constrain_tokens_gen: Optional[bool] = True,
        validator: Optional[ObservationValidator] = None,
        continuous_empty_limit: int = 10,
        suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        #related_num: Optional[Union[int, List[int]]] = None,
        raw=False,
        **generate_kwargs,
    ) -> pd.DataFrame:
        self._check_model()
        device = validate_get_device(device)

        # Clear the cache
        torch.cuda.empty_cache()

        if self.model_type == ModelType.tabular:
            assert n_samples
            assert self.tabular_max_length is not None
            assert self.tabular_col_size is not None
            assert self.col_transform_data is not None

            tabular_sampler = TabularSampler.sampler_from_model(
                rtf_model=self, device=device
            )
            synth_df = tabular_sampler.sample_tabular(
                n_samples=n_samples,
                gen_batch=gen_batch,
                device=device,
                seed_input=seed_input,
                constrain_tokens_gen=constrain_tokens_gen,
                validator=validator,
                continuous_empty_limit=continuous_empty_limit,
                suppress_tokens=suppress_tokens,
                forced_decoder_ids=forced_decoder_ids,
                raw=raw,
                **generate_kwargs,
            )

        #elif self.model_type == ModelType.relational:
        #    raise Exception("Relational")

        if save_samples:
            samples_fname = (
                self.samples_save_dir
                / f"rtf_{self.model_type}-exp_{self.experiment_id}-{int(time.time())}-samples_{synth_df.shape[0]}.pkl"
            )
            samples_fname.parent.mkdir(parents=True, exist_ok=True)
            synth_df.to_pickle(samples_fname)

        return synth_df

    def postprocess(
        self, 
        synth_sample,
        device: str = "cuda",
        validator: Optional[ObservationValidator] = None,
    ):
        #self._check_model()
        try:
            device = validate_get_device(device)
        except AttributeError:
            pass

        # Clear the cache
        torch.cuda.empty_cache()

        if self.model_type == ModelType.tabular:
            assert self.tabular_max_length is not None
            assert self.tabular_col_size is not None
            assert self.col_transform_data is not None

            tabular_sampler = TabularSampler.sampler_from_model(
                rtf_model=self, device=device, ignore_model=True
            )
            synth_sample = tabular_sampler.processes_sample(
                sample_outputs=synth_sample,
                vocab=self.vocab,
                validator=validator,
            )
        return synth_sample

    def predict(
        self,
        data: pd.DataFrame,
        target_col: str,
        target_pos_val: Any = None,
        batch: int = 32,
        obs_sample: int = 30,
        fillunk: bool = True,
        device: str = "cuda",
        disable_progress_bar: bool = True,
        **generate_kwargs,
    ) -> pd.Series:

        assert (
            self.model_type == ModelType.tabular
        ), "The predict method is only implemented for tabular data..."
        self._check_model()
        device = validate_get_device(device)
        batch = min(batch, data.shape[0])

        # Clear the cache
        torch.cuda.empty_cache()

        tabular_sampler = TabularSampler.sampler_from_model(self, device=device)


        return tabular_sampler.predict(
            data=data,
            target_col=target_col,
            target_pos_val=target_pos_val,
            batch=batch,
            obs_sample=obs_sample,
            fillunk=fillunk,
            device=device,
            disable_progress_bar=disable_progress_bar,
            **generate_kwargs,
        )

    def save(self, path: Union[str, Path], allow_overwrite: Optional[bool] = False):
        """Save REaLTabFormer Model

        Saves the model weights and a configuration file in the given directory.
        Args:
            path: Path where to save the model
        """
        self._check_model()
        assert self.experiment_id is not None

        if isinstance(path, str):
            path = Path(path)

        # Add experiment id to the save path
        path = path / self.experiment_id

        config_file = path / ModelFileName.rtf_config_json
        model_file = path / ModelFileName.rtf_model_pt

        if path.is_dir() and not allow_overwrite:
            if config_file.exists() or model_file.exists():
                raise ValueError(
                    "This directory is not empty, and contains either a config or a model."
                    " Consider setting `allow_overwrite=True` if you want to overwrite these."
                )
            else:
                warnings.warn(
                    f"Directory {path} exists, but `allow_overwrite=False`."
                    " This will raise an error next time when the model artifacts \
                        exist on this directory"
                )

        path.mkdir(parents=True, exist_ok=True)

        # Save attributes
        rtf_attrs = self.__dict__.copy()
        rtf_attrs.pop("mlu_trainer")
        rtf_attrs.pop("model")

        # We don't need to store the `parent_config`
        # since a saved model should have the weights loaded from
        # the trained model already.
        for ignore_key in [
            "parent_vocab",
            "parent_gpt2_config",
            "parent_gpt2_state_dict",
            "parent_col_transform_data",
        ]:
            if ignore_key in rtf_attrs:
                rtf_attrs.pop(ignore_key)

        # GPT2Config is not JSON serializable, let us manually
        # extract the attributes.
        if rtf_attrs.get("tabular_config"):
            rtf_attrs["tabular_config"] = rtf_attrs["tabular_config"].to_dict()

        if rtf_attrs.get("relational_config"):
            raise Exception("Relational")

        rtf_attrs["checkpoints_dir"] = rtf_attrs["checkpoints_dir"].as_posix()
        rtf_attrs["samples_save_dir"] = rtf_attrs["samples_save_dir"].as_posix()

        config_file.write_text(json.dumps(rtf_attrs))

        # Save model weights
        torch.save(self.model.state_dict(), model_file.as_posix())

        if self.model_type == ModelType.tabular:
            # Copy the special model checkpoints for
            # tabular models.
            for artefact in TabularArtefact.artefacts():
                print("Copying artefacts from:", artefact)
                if (self.checkpoints_dir / artefact).exists():
                    shutil.copytree(
                        self.checkpoints_dir / artefact,
                        path / artefact,
                        dirs_exist_ok=True,
                    )

    @classmethod
    def load_from_dir(cls, path: Union[str, Path], config_file=ModelFileName.rtf_config_json, model_file=ModelFileName.rtf_model_pt):
        """Load a saved REaLTabFormer model

        Load trained REaLTabFormer model from directory.
        Args:
            path: Directory where REaLTabFormer model is saved
        Returns:
            REaLTabFormer instance
        """

        if isinstance(path, str):
            path = Path(path)

        config_file = path / config_file
        model_file = path / model_file

        assert path.is_dir(), f"Directory {path} does not exist."
        assert config_file.exists(), f"Config file {config_file} does not exist."
        assert model_file.exists(), f"Model file {model_file} does not exist."

        # Load the saved attributes
        rtf_attrs = json.loads(config_file.read_text())

        # Create new REaLTabFormer model instance
        try:
            realtf = cls(model_type=rtf_attrs["model_type"])
        except KeyError:
            # Back-compatibility for saved models
            # before the support for relational data
            # was implemented.
            realtf = cls(model_type="tabular")

        # Set all attributes and handle the
        # special case for the GPT2Config.
        for k, v in rtf_attrs.items():
            if k == "gpt_config":
                # Back-compatibility for saved models
                # before the support for relational data
                # was implemented.
                v = GPT2Config.from_dict(v)
                k = "tabular_config"

            elif k == "tabular_config":
                v = GPT2Config.from_dict(v)

            elif k == "relational_config":
                raise Exception("Relational")

            elif k in ["checkpoints_dir", "samples_save_dir"]:
                v = Path(v)

            elif k == "vocab":
                if realtf.model_type == ModelType.tabular:
                    # Cast id back to int since JSON converts them to string.
                    v["id2token"] = {int(ii): vv for ii, vv in v["id2token"].items()}
                #elif realtf.model_type == ModelType.relational:
                 #   raise Exception("Relational")
                else:
                    raise ValueError(f"Invalid model_type: {realtf.model_type}")

            elif k == "col_idx_ids":
                v = {int(ii): vv for ii, vv in v.items()}

            setattr(realtf, k, v)

        # Implement back-compatibility for REaLTabFormer version < 0.0.1.8.2
        # since the attribute `col_idx_ids` is not implemented before.
        if "col_idx_ids" not in rtf_attrs:
            if realtf.model_type == ModelType.tabular:
                realtf.col_idx_ids = {
                    ix: realtf.vocab["column_token_ids"][col]
                    for ix, col in enumerate(realtf.processed_columns)
                }
            #elif realtf.model_type == ModelType.relational:
            #    raise Exception("Relational")

        # Load model weights
        if realtf.model_type == ModelType.tabular:
            realtf.model = GPT2LMHeadModel(realtf.tabular_config)
        #elif realtf.model_type == ModelType.relational:
        #    raise Exception("Relational")
        else:
            raise ValueError(f"Invalid model_type: {realtf.model_type}")

        realtf.model.load_state_dict(
            torch.load(model_file.as_posix(), map_location="cpu")
        )

        return realtf
