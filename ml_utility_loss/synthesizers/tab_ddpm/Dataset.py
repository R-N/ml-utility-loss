
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict, List
import torch
import numpy as np
from .util import TaskType

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]
DATASET_TYPES = ["X_num", "X_cat", "y"]
DATASET_TYPE_INDEX = {
    "X_num": 0,
    "X_cat": 1,
    "y": 2
}


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]

@dataclass(frozen=False)
class Dataset:
    #datasets: ArrayDict
    train_set: ArrayDict
    val_set: Optional[ArrayDict]
    test_set: Optional[ArrayDict]
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION
    
    @property
    def has_num(self):
        return "X_num" in self.train_set and self.train_set["X_num"] is not None
    
    @property
    def has_cat(self):
        return "X_cat" in self.train_set and self.train_set["X_cat"] is not None

    @property
    def n_num_features(self) -> int:
        return 0 if not self.has_num else self.train_set["X_num"].shape[1]

    @property
    def num_numerical_features(self) -> int:
        return self.n_num_features

    @property
    def n_cat_features(self) -> int:
        return 0 if not self.has_cat else self.train_set['X_cat'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1
    """
    @property
    def train_set(self):
        return self.datasets["train"] if "train" in self.datasets else None
        
    @property
    def val_set(self):
        return self.datasets["val"] if "val" in self.datasets else None
        
    @property
    def test_set(self):
        return self.datasets["test"] if "test" in self.datasets else None
    """

    @property
    def train_category_sizes(self) -> List[int]:
        return [] if not self.has_cat else get_category_sizes(self.train_set["X_cat"])

    