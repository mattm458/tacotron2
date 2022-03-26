from torch import nn


class ConcordanceCorrelationCoefficientLoss(nn.Module):
    def forward(self, pred, actual):
        pred_mean = pred.mean()
        actual_mean = actual.mean()

        covariance = ((pred - pred_mean) * (actual - actual_mean)).sum() / len(pred)

        ccc = (2 * covariance) / (
            pred.var(False) + actual.var(False) + ((pred_mean - actual_mean) ** 2)
        )

        return 1 - ccc
