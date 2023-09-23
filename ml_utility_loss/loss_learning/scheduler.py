
from torch import inf

class PretrainingScheduler:
    """
        Based on ReduceLROnPlateau
    """
    def __init__(
        self, 
        min_size=32, max_size=inf, size_factor=2, 
        min_aug=0, max_aug=1.0, aug_step=0.1, aug_inc=1,
        mode='min', patience=10, cooldown=0,
        threshold=1e-4, threshold_mode='rel', 
        eps=1e-8, 
        verbose=False,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.cur_size = min_size
        self.size_factor = size_factor

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
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets counters."""
        self.best = self.mode_worse
        self.aug_counter = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._increase_size(epoch)

    def get_size(self):
        return min(self.max_size, self.cur_size)

    def get_aug(self):
        return min(self.max_aug, self.cur_aug)

    def _increase_size(self, epoch=None):
        old_size = self.cur_size
        self.cur_size = min(self.max_size, self.cur_size * self.size_factor)
        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0
        if old_size < self.cur_size and self.verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print('Epoch {}: increase size to {:.4e}.'.format(epoch_str, self.cur_size))

        if self.aug_inc:
            self.aug_counter += 1
            if self.aug_counter >= self.aug_inc:
                self._increase_aug(epoch=epoch)

    def _increase_aug(self, epoch=None):
        old_aug = self.cur_aug
        self.cur_aug = min(self.max_aug, self.cur_aug + self.aug_step)
        self.aug_counter = 0
        if old_aug < self.cur_aug and self.verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print('Epoch {}: increase aug to {:.4e}.'.format(epoch_str, self.cur_aug))

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

    @property
    def is_done(self):
        return self.cur_size >= self.max_size and self.cur_aug >= self.max_aug
