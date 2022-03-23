import torch
import torch.nn.functional as F

def student_loss(student_logits, teacher_logits, labels, alpha=.9, tau=4):
    # print(student_logits)
    p_s_tilde = F.softmax(student_logits/tau, dim=-1)
    # print(p_s_tilde.shape)
    p_t_tilde = F.softmax(teacher_logits/tau, dim=-1)
    L_KD = (tau ** 2) * F.cross_entropy(p_s_tilde, p_t_tilde, size_average=False)
    L_cls = F.cross_entropy(student_logits, labels)
    return alpha * L_cls + (1-alpha) * L_KD