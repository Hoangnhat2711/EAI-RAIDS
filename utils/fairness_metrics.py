import torch
from sklearn.metrics import confusion_matrix
import numpy as np

def get_privileged_unprivileged_indices(sensitive_attrs, privileged_group, unprivileged_group):
    """
    Identifies the indices corresponding to privileged and unprivileged groups
    within the sensitive attributes tensor.

    Args:
        sensitive_attrs (torch.Tensor): A tensor of sensitive attributes, typically one-hot encoded
                                        or multi-hot encoded, with shape (num_samples, num_sensitive_attribute_categories).
        privileged_group (int or list): The index (or list of indices) in the sensitive_attrs tensor
                                        that represents the privileged group.
        unprivileged_group (int or list): The index (or list of indices) in the sensitive_attrs tensor
                                          that represents the unprivileged group.

    Returns:
        tuple: A tuple containing two `torch.Tensor`:
            - privileged_indices (torch.Tensor): Indices of samples belonging to the privileged group.
            - unprivileged_indices (torch.Tensor): Indices of samples belonging to the unprivileged group.
    """
    if isinstance(privileged_group, int):
        privileged_indices = (sensitive_attrs[:, privileged_group] == 1).nonzero(as_tuple=True)[0]
    else:
        # Handle multi-hot encoding for privileged group
        mask = torch.zeros(len(sensitive_attrs), dtype=torch.bool)
        for group_idx in privileged_group:
            mask |= (sensitive_attrs[:, group_idx] == 1)
        privileged_indices = mask.nonzero(as_tuple=True)[0]

    if isinstance(unprivileged_group, int):
        unprivileged_indices = (sensitive_attrs[:, unprivileged_group] == 1).nonzero(as_tuple=True)[0]
    else:
        # Handle multi-hot encoding for unprivileged group
        mask = torch.zeros(len(sensitive_attrs), dtype=torch.bool)
        for group_idx in unprivileged_group:
            mask |= (sensitive_attrs[:, group_idx] == 1)
        unprivileged_indices = mask.nonzero(as_tuple=True)[0]

    return privileged_indices, unprivileged_indices

def calculate_statistical_parity_difference(y_pred, sensitive_attrs, privileged_group, unprivileged_group):
    """
    Calculates the Statistical Parity Difference (SPD).

    SPD measures the difference in the proportion of positive predictions (Y_pred=1)
    between the unprivileged group and the privileged group.
    Formula: SPD = P(Y_pred=1 | S=unprivileged) - P(Y_pred=1 | S=privileged)

    Args:
        y_pred (torch.Tensor): Predicted labels (0 or 1) as a tensor.
        sensitive_attrs (torch.Tensor): Tensor of sensitive attributes (one-hot encoded or multi-hot).
        privileged_group (int or list): Index/indices of the privileged group in sensitive_attrs.
        unprivileged_group (int or list): Index/indices of the unprivileged group in sensitive_attrs.

    Returns:
        float: The calculated Statistical Parity Difference. Returns np.nan if a group is empty.
    """
    privileged_indices, unprivileged_indices = get_privileged_unprivileged_indices(
        sensitive_attrs, privileged_group, unprivileged_group
    )

    if len(privileged_indices) == 0 or len(unprivileged_indices) == 0:
        return np.nan # Cannot calculate if a group is empty

    prob_positive_privileged = y_pred[privileged_indices].float().mean()
    prob_positive_unprivileged = y_pred[unprivileged_indices].float().mean()

    return prob_positive_unprivileged - prob_positive_privileged

def calculate_equal_opportunity_difference(y_true, y_pred, sensitive_attrs, privileged_group, unprivileged_group):
    """
    Calculates the Equal Opportunity Difference (EOD).

    EOD measures the difference in True Positive Rate (TPR) between the unprivileged
    group and the privileged group, focusing on cases where the true label is positive (Y_true=1).
    Formula: EOD = TPR_unprivileged - TPR_privileged, where TPR = P(Y_pred=1 | Y_true=1, S=group)

    Args:
        y_true (torch.Tensor): True labels (0 or 1) as a tensor.
        y_pred (torch.Tensor): Predicted labels (0 or 1) as a tensor.
        sensitive_attrs (torch.Tensor): Tensor of sensitive attributes (one-hot encoded or multi-hot).
        privileged_group (int or list): Index/indices of the privileged group in sensitive_attrs.
        unprivileged_group (int or list): Index/indices of the unprivileged group in sensitive_attrs.

    Returns:
        float: The calculated Equal Opportunity Difference. Returns np.nan if a group is empty.
    """
    privileged_indices, unprivileged_indices = get_privileged_unprivileged_indices(
        sensitive_attrs, privileged_group, unprivileged_group
    )

    if len(privileged_indices) == 0 or len(unprivileged_indices) == 0:
        return np.nan

    # Filter for true positives (y_true = 1)
    true_positive_privileged_count = ((y_pred[privileged_indices] == 1) & (y_true[privileged_indices] == 1)).sum().float()
    total_positive_privileged_count = (y_true[privileged_indices] == 1).sum().float()
    tpr_privileged = true_positive_privileged_count / total_positive_privileged_count if total_positive_privileged_count > 0 else 0

    true_positive_unprivileged_count = ((y_pred[unprivileged_indices] == 1) & (y_true[unprivileged_indices] == 1)).sum().float()
    total_positive_unprivileged_count = (y_true[unprivileged_indices] == 1).sum().float()
    tpr_unprivileged = true_positive_unprivileged_count / total_positive_unprivileged_count if total_positive_unprivileged_count > 0 else 0

    return tpr_unprivileged - tpr_privileged

def calculate_average_odds_difference(y_true, y_pred, sensitive_attrs, privileged_group, unprivileged_group):
    """
    Calculates the Average Odds Difference (AOD).

    AOD is the average of the difference in False Positive Rate (FPR) and True Positive Rate (TPR)
    between the unprivileged group and the privileged group.
    Formula: AOD = 0.5 * [(FPR_unprivileged - FPR_privileged) + (TPR_unprivileged - TPR_privileged)]

    Args:
        y_true (torch.Tensor): True labels (0 or 1) as a tensor.
        y_pred (torch.Tensor): Predicted labels (0 or 1) as a tensor.
        sensitive_attrs (torch.Tensor): Tensor of sensitive attributes (one-hot encoded or multi-hot).
        privileged_group (int or list): Index/indices of the privileged group in sensitive_attrs.
        unprivileged_group (int or list): Index/indices of the unprivileged group in sensitive_attrs.

    Returns:
        float: The calculated Average Odds Difference. Returns np.nan if a group is empty.
    """
    privileged_indices, unprivileged_indices = get_privileged_unprivileged_indices(
        sensitive_attrs, privileged_group, unprivileged_group
    )

    if len(privileged_indices) == 0 or len(unprivileged_indices) == 0:
        return np.nan

    # Calculate TPRs
    true_positive_privileged_count = ((y_pred[privileged_indices] == 1) & (y_true[privileged_indices] == 1)).sum().float()
    total_positive_privileged_count = (y_true[privileged_indices] == 1).sum().float()
    tpr_privileged = true_positive_privileged_count / total_positive_privileged_count if total_positive_privileged_count > 0 else 0

    true_positive_unprivileged_count = ((y_pred[unprivileged_indices] == 1) & (y_true[unprivileged_indices] == 1)).sum().float()
    total_positive_unprivileged_count = (y_true[unprivileged_indices] == 1).sum().float()
    tpr_unprivileged = true_positive_unprivileged_count / total_positive_unprivileged_count if total_positive_unprivileged_count > 0 else 0

    # Calculate FPRs (False Positive Rate)
    false_positive_privileged_count = ((y_pred[privileged_indices] == 1) & (y_true[privileged_indices] == 0)).sum().float()
    total_negative_privileged_count = (y_true[privileged_indices] == 0).sum().float()
    fpr_privileged = false_positive_privileged_count / total_negative_privileged_count if total_negative_privileged_count > 0 else 0

    false_positive_unprivileged_count = ((y_pred[unprivileged_indices] == 1) & (y_true[unprivileged_indices] == 0)).sum().float()
    total_negative_unprivileged_count = (y_true[unprivileged_indices] == 0).sum().float()
    fpr_unprivileged = false_positive_unprivileged_count / total_negative_unprivileged_count if total_negative_unprivileged_count > 0 else 0

    aod = 0.5 * ( (fpr_unprivileged - fpr_privileged) + (tpr_unprivileged - tpr_privileged) )
    return aod

def evaluate_fairness_metrics(y_true, y_pred, sensitive_attrs, privileged_groups, unprivileged_groups):
    """
    Calculates and returns a dictionary of various fairness metrics.

    This function iterates through specified sensitive attribute groups (e.g., 'sex', 'race')
    and computes Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD),
    and Average Odds Difference (AOD) for each.

    Args:
        y_true (torch.Tensor): True labels (0 or 1) as a tensor.
        y_pred (torch.Tensor): Predicted labels (0 or 1) as a tensor.
        sensitive_attrs (torch.Tensor): Tensor of sensitive attributes (one-hot encoded or multi-hot).
                                        Shape (num_samples, num_sensitive_attribute_categories).
        privileged_groups (dict): A dictionary where keys are sensitive feature names (e.g., 'sex')
                                  and values are the index (or list of indices) in `sensitive_attrs`
                                  representing the privileged group for that feature.
                                  Example: {'sex': 0, 'race': 0}
        unprivileged_groups (dict): A dictionary similar to `privileged_groups` but for the unprivileged groups.
                                    Example: {'sex': 1, 'race': [1,2,3,4]}

    Returns:
        dict: A dictionary containing fairness metrics, with keys like '{feature_name}_SPD',
              '{feature_name}_EOD', '{feature_name}_AOD'.
    """
    metrics = {}
    for sf_name in privileged_groups:
        privileged_group = privileged_groups[sf_name]
        unprivileged_group = unprivileged_groups[sf_name]

        spd = calculate_statistical_parity_difference(y_pred, sensitive_attrs, privileged_group, unprivileged_group)
        eod = calculate_equal_opportunity_difference(y_true, y_pred, sensitive_attrs, privileged_group, unprivileged_group)
        aod = calculate_average_odds_difference(y_true, y_pred, sensitive_attrs, privileged_group, unprivileged_group)

        metrics[f'{sf_name}_SPD'] = spd
        metrics[f'{sf_name}_EOD'] = eod
        metrics[f'{sf_name}_AOD'] = aod

    return metrics
