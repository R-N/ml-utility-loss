import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from torch import nn
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    logging,
)
from transformers.integrations import is_fairscale_available
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from transformers.utils import is_sagemaker_mp_enabled

logger = logging.get_logger(__name__)


class SaveEpochEndCallback(TrainerCallback):
    """This callback forces a checkpoint save at each epoch end."""

    def __init__(self, save_epochs: int = None) -> None:
        super().__init__()

        self.save_epochs = save_epochs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_epochs is not None:
            control.should_save = math.ceil(state.epoch) % self.save_epochs == 0
        else:
            control.should_save = True

        return control


class MLUtilityCallback(TrainerCallback):
    """This callback forces a checkpoint save at each epoch end."""

    def __init__(self, sampler, mlu_trainer=None):
        super().__init__()
        self.mlu_trainer = mlu_trainer
        self.sampler = sampler
        self.epoch = -1

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.epoch += 1
        if self.mlu_trainer and self.epoch%self.mlu_trainer.t_steps == 0:
            for i in range(self.mlu_trainer.n_steps):
                n_samples = self.mlu_trainer.n_samples
                batch_size=self.mlu_trainer.sample_batch_size
                samples = self.sampler.sample(
                    n_samples=n_samples,
                    gen_batch=batch_size,
                    raw=True,
                )
                self.mlu_trainer.step(samples)




class ResumableTrainer(Trainer):
    """This trainer makes the scheduler consistent over pauses
    in the training. The scheduler should return values similar
    to when a training is done either intermittently or continuously
    over the `target_epochs`.
    """

    def __init__(
        self,
        target_epochs: int = None,
        save_epochs: int = None,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
        mlu_trainer=None,
        sampler=None,
    ):
        # Declare here for typing
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None

        if callbacks is None:
            callbacks = []

        callbacks.append(SaveEpochEndCallback(save_epochs=save_epochs))
        mlu_callback = None
        if mlu_trainer:
            mlu_callback = MLUtilityCallback(sampler, mlu_trainer)
            callbacks.append(mlu_callback)

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.target_epochs = target_epochs
        model = self.model or model
        mlu_callback.mlu_trainer.create_optim(model.parameters())
        mlu_callback.mlu_trainer.set_embedding(model.transformer.wte)

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """

        if self.lr_scheduler is None:
            if self.target_epochs is not None:
                # Compute the max_steps based from the
                # `target_epochs`.
                train_dataloader = self.get_train_dataloader()
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = (
                    len_dataloader // self.args.gradient_accumulation_steps
                )
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

                max_steps = math.ceil(self.target_epochs * num_update_steps_per_epoch)
                num_training_steps = max_steps

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
        #return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        return super().training_step(model, inputs)
