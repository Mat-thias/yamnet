from typing import Tuple
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import torch


def valid_pred_true_aggregation(
    pred:torch.Tensor,
    true:torch.Tensor,
    aggregation:str = "max"
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pred excepts the framed prediction while true expects the full audio class (0 or 1)
    assert pred.ndim > 2, f"pred should have ndim > 2, received {pred.ndim}"
    assert true.ndim > 1, f"true should have ndim > 1, received {true.ndim}"
    aggregation_methods = ["max", "avg"]
    assert aggregation in aggregation_methods, \
        f"aggregation method must be one of {aggregation_methods}, received {aggregation}"

    # aggregating the framed predictions
    pred = pred.max(dim=-2).values if aggregation == "max" else\
           pred.mean(dim=-2) if aggregation == "avg" else\
           None
    pred = pred.view(-1, pred.size(-1))
    true = true.view(-1, true.size(-1))

    assert pred.size() == true.size(), (
        f"pred and true shape is not compatable, received pred: {pred.size()} after aggregation "
        f"and received true shape: {true.size()}"
    )
    return pred, true


def  mAP(pred:torch.Tensor, true:torch.Tensor, aggregation:str = "max") -> float:
    """Returns the mean average precision of a batch prediction."""

    pred, true = valid_pred_true_aggregation(pred, true, aggregation)
    # classes without any true values are skipped
    valid_class = (true.sum(dim=0) > 0)
    pred = pred[:, valid_class]
    true = true[:, valid_class]

    num_sample, num_class = pred.size()

    sorted_indices = torch.argsort(pred, dim=0, descending=True)
    sorted_true = torch.gather(true, dim=0, index=sorted_indices)
    tp = torch.cumsum(sorted_true, dim=0)
    precision = tp / torch.arange(1, num_sample+1, device=pred.device).unsqueeze(1)
    recall = tp / true.sum(dim=0)
    delta = torch.diff(recall, dim=0, prepend=recall.new_zeros(1, num_class))
    return torch.sum(precision * delta).item() / num_class


def confusion_matrix(
    pred:torch.Tensor, true:torch.Tensor,
    aggregation:str = "max", threshold:float = 0.5
):
    """Returns the confusion matrix of a batch prediction."""

    pred, true = valid_pred_true_aggregation(pred, true, aggregation)
    pred = (pred > threshold).int()
    _, num_class = pred.size()

    confusion_matrices = [
        sk_confusion_matrix(y_pred=pred[:, c].numpy(), y_true=true[:, c].numpy(), labels=[0, 1])
        for c in range(num_class)
    ]
    return confusion_matrices

