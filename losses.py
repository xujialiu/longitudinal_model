from sympy import sequence
import torch
import numpy as np


class CrossEntropySurvLoss(object):
    # CREDIT TO https://github.com/mahmoodlab/MCAT/blob/b9cca63be83c67de7f95308d54a58f80b78b0da1/utils/utils.py
    def __init__(self, beta=0.15):
        self.beta = beta

    def __call__(self, hazards, S, Y, c, obs_times, beta=None):
        if beta is None:
            return self.ce_loss(hazards, S, Y, c, obs_times, beta=self.beta)
        else:
            return self.ce_loss(hazards, S, Y, c, obs_times, beta=beta)

    @staticmethod
    def ce_loss(hazards, S, Y, c, obs_times=None, beta=0.15, eps=1e-7):
        batch_size = len(Y)

        Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1

        if S is None:
            S = torch.cumprod(
                1 - hazards, dim=1
            )  # surival is cumulative product of 1 - hazards

        S_padded = torch.cat([torch.ones_like(c), S], 1)
        # batch_size, seq_len, _ = S.shape
        # ones = torch.ones(batch_size, seq_len, 1).to(S.device)
        # S_padded = torch.cat([ones, S], dim=2)

        reg = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y) + eps)
            + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
        ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (
            1 - c
        ) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
        loss = (1 - beta) * ce_l + beta * reg
        loss = loss.mean()

        return loss


class NLLSurvLoss(object):
    def __init__(self, beta=0.15):
        self.beta = beta

    def __call__(self, hazards, S, Y, c, obs_times, beta=None):
        if beta is None:
            return self.nll_loss(hazards, S, Y, c, obs_times, beta=self.beta)
        else:
            return self.nll_loss(hazards, S, Y, c, obs_times, beta=beta)

    @staticmethod
    def nll_loss(hazards, S, Y, c, obs_times, beta=0.15, eps=1e-7):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
        if S is None:
            S = torch.cumprod(
                1 - hazards, dim=1
            )  # surival is cumulative product of 1 - hazards

        S_padded = torch.cat(
            [torch.ones_like(c), S], 1
        )  # S(-1) = 0, all patients are alive from (-inf, 0) by definition

        uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
            + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
        censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - beta) * neg_l + beta * uncensored_loss
        loss = loss.mean()
        return loss


class CoxSurvLoss(object):
    def __call__(hazards, S, Y, c, obs_times, beta, eps, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        device = S.device()

        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c)
        )

        return loss_cox
