import torch
import torch.nn.functional as F
import torch.nn as nn # Import nn for KLDivLoss

def lwf_loss(student_outputs, teacher_outputs, temperature=2.0):
    """
    Calculates the Learning without Forgetting (LwF) distillation loss.

    LwF aims to prevent catastrophic forgetting by encouraging the current model
    (student) to produce similar outputs to the previous model (teacher) on old tasks.
    This is achieved through a Kullback-Leibler (KL) divergence loss between
    the softened softmax probabilities of the student and teacher models.

    Args:
        student_outputs (torch.Tensor): Logits from the current model (student).
        teacher_outputs (torch.Tensor): Logits from the previous task's trained model (teacher).
        temperature (float, optional): Temperature for softmax distillation. A higher temperature
                                       produces a softer probability distribution. Defaults to 2.0.

    Returns:
        torch.Tensor: The LwF distillation loss, scaled by the square of the temperature.
    """
    # Ensure teacher_outputs are detached to prevent gradients flowing back to the teacher model
    teacher_outputs = teacher_outputs.detach()

    # Apply softmax with temperature and then log_softmax for numerical stability
    # KL_divergence expects log-probabilities for the first argument and probabilities for the second
    loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1)
    ) * (temperature * temperature) # Scale by T^2 as per original paper

    return loss
