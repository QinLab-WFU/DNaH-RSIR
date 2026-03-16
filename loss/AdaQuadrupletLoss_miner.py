import torch
from torch import nn

from .utils import distance


def get_all_quadruplets(labels, need_jk=False):
    """
    get quadruplets like (A, P1, P2, N)
    just fit AP1, AP2, AN & P1!=P2
    note P1P2 may negative, P1N, P2N may positive, just leave another quadruplet to constrain the margin
    Args:
        multi-hot labels
    Returns:
        quadruplets: shape is n_quadruplets x 4
        Sjk: see "C. Multi-Label Based Hashing" of paper
    """
    sames = (labels @ labels.T > 0).byte()
    diffs = sames ^ 1
    sames.fill_diagonal_(0)

    # mining anchor, positive1, positive2
    I, J, K = torch.where(
        sames.unsqueeze(2) * sames.unsqueeze(1) * torch.triu(1 - torch.eye(sames.shape[0], device=labels.device))
    )

    if I.numel() == 0:
        # print("I is None")
        return None

    # finding negatives & gen quadruplets
    N = diffs[I].nonzero()
    if N.numel() == 0:
        # print("N is None")
        return None
    idx = N[:, 0]
    quadruplets = torch.hstack((I[idx].unsqueeze(1), J[idx].unsqueeze(1), K[idx].unsqueeze(1), N[:, 1].unsqueeze(1)))
    # assert (sames[J[idx], K[idx]] == (labels @ labels.T > 0).byte()[J[idx], K[idx]]).all()
    return quadruplets if not need_jk else (quadruplets, sames[J[idx], K[idx]])


class QuadrupletMarginMiner(nn.Module):
    """
    Returns quadruplets that violate the margin
    """

    def __init__(self, margin, type_of_distance, type_of_quadruplets, what_is_hard):
        super().__init__()
        self.margin = margin
        self.type_of_distance = type_of_distance
        self.type_of_quadruplets = type_of_quadruplets
        self.what_is_hard = what_is_hard

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
        # print("margin1:", margin1 <= self.margin)
        # print("margin2:", margin2 <= self.margin)
        # violation1 = ij_dists - in_dists + self.margin
        # violation2 = ik_dists - in_dists + self.margin
        # print("violation:", nn.functional.relu(violation1)+nn.functional.relu(violation2))

        if self.what_is_hard == "one":
            opt = lambda x, y: x | y
        elif self.what_is_hard == "all":
            opt = lambda x, y: x & y
        else:
            raise NotImplementedError(f"not support: {self.what_is_hard}")

        if self.type_of_quadruplets == "easy":
            threshold_condition = opt(margin1 > self.margin, margin2 > self.margin)
        else:
            threshold_condition = opt(margin1 <= self.margin, margin2 <= self.margin)
            if self.type_of_quadruplets == "hard":
                threshold_condition &= opt(margin1 <= 0, margin2 <= 0)
            elif self.type_of_quadruplets == "semi-hard":
                threshold_condition &= opt(margin1 > 0, margin2 > 0)
            else:
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
    _miner = QuadrupletMarginMiner(0.25, "cosine", "all", "one")
    quadruplets = _miner(_logits, _labels)
    print(quadruplets)
