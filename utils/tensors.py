import torch


def overload_diag(input: torch.Tensor):
    input[:, 0, :, :, :] = input[:, 0, :, :, :] + 0.000001
    input[:, 4, :, :, :] = input[:, 4, :, :, :] + 0.0000001
    input[:, 8, :, :, :] = input[:, 8, :, :, :] + 0.00000001

    return input


def compute_fa(eigen_values: torch.Tensor):
    num = torch.pow((eigen_values[:, 0, :, :, :] - eigen_values[:, 1, :, :, :]), 2) + \
          torch.pow((eigen_values[:, 1, :, :, :] - eigen_values[:, 2, :, :, :]), 2) + \
          torch.pow((eigen_values[:, 0, :, :, :] - eigen_values[:, 2, :, :, :]), 2)
    denom = 2 * (torch.pow(eigen_values[:, 0, :, :, :], 2) + torch.pow(eigen_values[:, 1, :, :, :], 2) + torch.pow(
        eigen_values[:, 2, :, :, :], 2))

    return torch.clamp(torch.pow(num / denom, 0.5), 0, 1)