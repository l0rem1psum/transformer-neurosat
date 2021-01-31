import torch


def compute_loss(logits, labels, parameters):
    # x = logits, z = labels
    # max(x, 0) - x * z + log(1 + exp(-abs(x)))
    predict_costs = torch.nn.functional.relu(logits) - logits * labels + torch.log1p(torch.exp(-torch.abs(logits)))
    predict_cost = torch.mean(predict_costs)
    l2_cost = torch.zeros([])#.cuda()
    for p in parameters:
        l2_cost += torch.sum(p ** 2) / 2
    return predict_cost + 1e-6 * l2_cost