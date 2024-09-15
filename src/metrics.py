
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
from torch import Tensor
from utils import dice_coef

def update_metrics(K: int, total_pred_seg: Tensor, total_gt_seg: Tensor):
    
    dice_metric = dice_coef(total_pred_seg, total_gt_seg) # the dice metric
    
    # Hausdorff metric
    
    
    # Sensitivity metric
    
    