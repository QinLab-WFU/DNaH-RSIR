from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from .utils import distance
from .miner import AdaQuadrupletMiner


class AdaQuadrupletLoss(nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super().__init__()
        self.type_of_distance = args.type_of_distance
        self.need_cnt = kwargs.pop("need_cnt", False)
        self.miner = AdaQuadrupletMiner(args)

    def forward(self, logits, labels):
        quadruplets = self.miner(logits, labels)
        if quadruplets is None:
            print("no quadruplets")
            return (0, 0) if self.need_cnt else 0

        # get the updated params
        epsilon = self.miner.epsilon

        mat = distance(logits, self.type_of_distance)
        I, J, K, N = quadruplets[:, 0], quadruplets[:, 1], quadruplets[:, 2], quadruplets[:, 3]
        ij_dists = mat[I, J]
        ik_dists = mat[I, K]
        in_dists = mat[I, N]

        # 少了jk之间的计算
        violation1 = ij_dists - in_dists + epsilon
        violation2 = ik_dists - in_dists + epsilon    # epsilon 为 mad

        losses = F.relu(violation1) + F.relu(violation2)
        loss = torch.mean(losses)

        return (loss, quadruplets.size(0)) if self.need_cnt else loss


if __name__ == "__main__":
    pass
