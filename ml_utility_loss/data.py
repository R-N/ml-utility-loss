import torch
import torch.utils.data

#https://github.com/pytorch/pytorch/issues/15849

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

