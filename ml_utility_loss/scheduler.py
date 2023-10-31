from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR, ReduceLROnPlateau as _ReduceLROnPlateau
import optuna
from torch import inf
from copy import deepcopy
import torch

class PretrainingScheduler:
    """
        Based on ReduceLROnPlateau
    """
    def __init__(
        self, 
        min_size=32, max_size=inf, size_factor=2, 
        min_batch_size=4, max_batch_size=64, batch_size_factor=0.5,
        min_aug=0, max_aug=0, aug_step=0.1, aug_inc=1,
        mode='min', patience=10, cooldown=0,
        threshold=1e-4, threshold_mode='rel', 
        eps=1e-8, 
        model=None,
        verbose=False,
    ):

        self.min_size = min_size
        self.max_size = max_size
        self.cur_size = min_size
        self.size_factor = size_factor

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.cur_batch_size = max_batch_size
        self.batch_size_factor = batch_size_factor

        self.min_aug = min_aug
        self.max_aug = max_aug
        self.cur_aug = min_aug
        self.aug_step = aug_step
        self.aug_inc = aug_inc
        self.aug_counter = 0

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.model = model
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets counters."""
        self.best = self.mode_worse
        self.best_state = None
        self.last_epoch = -1
        self.best_epoch = -1
        self.aug_counter = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.stopped = False

    def step(self, train_loss, val_loss=None, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        train_loss = train_loss.item() if torch.is_tensor(train_loss) else train_loss
        val_loss = val_loss.item() if torch.is_tensor(val_loss) else val_loss
        assert not isinstance(train_loss, int), "got int train_loss"
        assert not isinstance(val_loss, int), "got int val_loss"
        metrics = val_loss if val_loss is not None else train_loss
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        elif epoch == self.last_epoch:
            return False

        if self.is_better(current, self.best):
            self.new_best(current, epoch=epoch)
        else:
            print(current, "not better than", self.best)
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            return self._increase_size(epoch=epoch)

        self.last_epoch = epoch
        return False
    
    def new_best(self, current, epoch=None):
        epoch = epoch if epoch is not None else self.last_epoch+1
        self.best = current
        self.num_bad_epochs = 0
        self.best_epoch = epoch
        print("Found new best")
        if self.model:
            if self.best_state is not None:
                del self.best_state
            print("Saving new best state")
            self.best_state = deepcopy(self.model.state_dict())

    def get_size(self):
        return min(self.max_size, self.cur_size)

    def get_batch_size(self):
        return max(self.min_batch_size, self.cur_batch_size)

    def get_aug(self):
        return min(self.max_aug, self.cur_aug)

    def _increase_size(self, epoch=None):
        epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
        updated = False

        if (self.check_done()):
            self.stop()
            print(f"Epoch {epoch_str}: done.")
            return False

        old_size = self.cur_size
        self.cur_size = min(self.max_size, int(self.cur_size * self.size_factor))

        old_batch_size = self.cur_batch_size
        self.cur_batch_size = max(self.min_batch_size, int(self.cur_batch_size * self.batch_size_factor))

        if old_size < self.cur_size:
            updated = True
            if self.verbose:
                print(f"Epoch {epoch_str}: increase size to {self.cur_size}.")
        if old_batch_size < self.cur_batch_size:
            updated = True
            if self.verbose:
                print(f"Epoch {epoch_str}: increase batch size to {self.cur_batch_size}.")

        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0

        if self.aug_inc:
            self.aug_counter += 1
            if self.aug_counter >= self.aug_inc:
                updated = updated or self._increase_aug(epoch=epoch)

        if (not updated and self.check_done()):
            self.stop()
            print(f"Epoch {epoch_str}: done. best_epoch={self.best_epoch}")
            return False

        return updated

    def _increase_aug(self, epoch=None):
        updated = False
        old_aug = self.cur_aug
        self.cur_aug = min(self.max_aug, self.cur_aug + self.aug_step)
        self.aug_counter = 0
        if old_aug < self.cur_aug:
            updated = True
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print('Epoch {}: increase aug to {:.4e}.'.format(epoch_str, self.cur_aug))
        self.check_done()
        return updated

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

    def check_done(self):
        if self.cur_size >= self.max_size and self.cur_batch_size <= self.min_batch_size and self.cur_aug >= self.max_aug and self.num_bad_epochs > self.patience:
            self.stop()
        return self.stopped

    def stop(self):
        if not self.stopped:
            if self.model:
                self.model.load_state_dict(self.best_state)
            self.stopped = True


class ReduceLROnPlateau(_ReduceLROnPlateau):
    def __init__(self, *args, factor=0.1, patience=10, cooldown=2, min_lr=1e-7, **kwargs):
        super().__init__(*args, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr, **kwargs)

    def _reduce_lr(self, epoch):
        updated = False
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                updated = True
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
        log = "Learning rate stuck"
        if not updated:
            self.stop()
            print(log)

    def stop(self):
        if not self.stopped:
            #if self.model:
            #    self.model.load_state_dict(self.best_state)
            self.stopped = True

class OneCycleLR:
    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, div_factor=25, autodecay=0.5):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.div_factor = max(25, div_factor)
        print("max_lr", self.max_lr)
        print("div_factor", self.div_factor)
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = epochs
        print("onecycle max_epochs", epochs)
        self.epochs = 0
        self.scheduler = None
        self.autodecay = autodecay
        self.create()

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def update_max_lr(self, max_lr, initial_lr=None, div_factor=None):
        if div_factor:
            self.div_factor = div_factor
        elif initial_lr:
            self.div_factor = max_lr / initial_lr
        elif self.initial_lr < max_lr:
            self.div_factor = max_lr / self.initial_lr
            self.div_factor = max(self.div_factor, 25)
        else:
            self.div_factor = 25
        self.max_lr = max_lr
        print("max_lr", self.max_lr)
        print("div_factor", self.div_factor)

    def create(self, last_epoch=-1):
        self.scheduler = _OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.max_epochs,
            last_epoch=last_epoch
        )
        return self.scheduler

    @property
    def initial_lr(self):
        return self.max_lr / self.div_factor

    @property
    def state_dict(self):
        return self.scheduler.state_dict

    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)

    def reset(self):
        """
        self.scheduler.last_epoch = -1
        self.scheduler.step()
        """
        if self.autodecay:
            self.update_max_lr(
                self.initial_lr * (
                    self.div_factor ** self.autodecay
                )
            )
        self.create()
        self.epochs = 0

    def step(self, *args, **kwargs):
        ret = self.scheduler.step()
        self.epochs += 1
        if self.epochs >= self.max_epochs:
            self.reset()
        return ret
