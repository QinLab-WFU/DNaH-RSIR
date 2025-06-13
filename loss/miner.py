from argparse import Namespace

import torch
from torch import nn

from .AdaQuadrupletLoss_miner import get_all_quadruplets
from .utils import distance


class AdaQuadrupletMiner(nn.Module):
    """
    Returns quadruplets that violate the margin
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.type_of_distance = args.type_of_distance
        self.type_of_quadruplets = args.type_of_quadruplets
        self.what_is_hard = args.what_is_hard

        self.epsilon = args.epsilon
        self.k_delta = args.k_delta

    def update(self, dist, dist_type):
        # dist = dist.cpu().numpy()
        if dist_type == "ap_an":
            # Eq. (7): ε(t) = μΔ(t)/K_Δ
            self.epsilon = nn.functional.relu(dist.mean() / self.k_delta)
        else:
            raise NotImplementedError(f"not support: {dist_type}")

    def forward(self, logits, labels):
        quadruplets = get_all_quadruplets(labels)
        if quadruplets is None:
            return None

        # mat = distance(logits, self.type_of_distance)
        mat = distance(logits.detach(), self.type_of_distance)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]
        margin1 = in_dists - ij_dists
        margin2 = in_dists - ik_dists

        delta = torch.cat((margin1, margin2))
        self.update(delta, "ap_an")

        if self.what_is_hard == "one":
            opt = lambda x, y: x | y
        elif self.what_is_hard == "all":
            opt = lambda x, y: x & y
        else:
            raise NotImplementedError(f"not support: {self.what_is_hard}")

        if self.type_of_quadruplets == "easy":
            threshold_condition = opt(margin1 > self.epsilon, margin2 > self.epsilon)

        else:
            threshold_condition = opt(margin1 <= self.epsilon, margin2 <= self.epsilon)
            if self.type_of_quadruplets == "hard":
                threshold_condition &= opt(margin1 <= 0, margin2 <= 0)
            elif self.type_of_quadruplets == "semi-hard":
                threshold_condition &= opt(margin1 > 0, margin2 > 0)
            else:
                # 2
                pass  # here is "all"

        if not threshold_condition.any():
            return None

        return quadruplets[threshold_condition]


if __name__ == "__main__":
    _labels = torch.tensor(
        [
            [1, 1, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, 0, 1],  # 3
            [1, 0, 0],  # 4
        ]
    )
    _logits = torch.randn((5, 4), requires_grad=True)
    print(get_all_quadruplets(_labels, False))

    _args = Namespace(
        type_of_distance="cosine",
        type_of_quadruplets="all",
        what_is_hard="one",
        epsilon=0.25,
        beta=0,
        k_delta=2,
        k_an=2,
        calc_loss_an=False
    )
    _miner = AdaQuadrupletMiner(_args)
    quadruplets = _miner(_logits, _labels)
    print(quadruplets)
