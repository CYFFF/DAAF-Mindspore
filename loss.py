import math
import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from src.model_utils.config import config


class Triplet_Loss(nn.Cell):
    def __init__(self):
        super(Triplet_Loss, self).__init__()
        self.abs = 0

    def construct(self, base, target):


        return 0