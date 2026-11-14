import numpy as np
import torch
from ....base import HardLabelAttackBase

"""
Chen and Gu,
RayS: A Ray Searching Method for Hard-label Adversarial Attack
ACM SIGKDD (KDD), 2020

Adapted from https://github.com/uclaml/RayS (RayS_Single.py)
MIT License, Copyright (c) 2020 Jinghui Chen, Quanquan Gu

Changes for this repo:
- every model query goes through check_adversary() -> inject(), so GWAD sees it
  (and the 'batch' adaptive attack can interleave benign queries)
- the search runs in [0, 1] pixel space like the original; the loader gives normalised
  images, so they are converted with mu / std and normalised again before each query
- one block flip + its binary search per attack() call; the common run() loop of
  BlackBoxAtackCommonBase enforces the query budget
"""


class RayS(HardLabelAttackBase):
    def __init__(self, device, model, q_budgets=[10000], epsilon=8/255, order=np.inf, lin_search_rad=10,
                 tol=1e-3, mu=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261), stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp='inf' if order == np.inf else 'l2',
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='dec')
        self.name = 'rays'
        self.epsilon = epsilon          # early stopping threshold (pixel space, `order` norm)
        self.order = order
        self.lin_search_rad = lin_search_rad
        self.tol = tol
        self.mu = torch.tensor(mu).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    # -- pixel <-> model input space --
    def to_pixel(self, x):
        return x * self.std + self.mu

    def to_model(self, p):
        return (p - self.mu) / self.std

    def get_xadv(self, v, d):
        return torch.clamp(self.p + d * v, 0., 1.)

    def budget_left(self):
        return self.query_cnt < self.query_max

    def search_succ(self, p_adv):
        # one hard-label query through GWAD
        return self.check_adversary(self.to_model(p_adv))

    # -- RayS --
    def lin_search(self, sgn):
        for d in range(1, self.lin_search_rad + 1):
            if not self.budget_left():
                break
            if self.search_succ(self.get_xadv(sgn, d)):
                return d
        return np.inf

    def binary_search(self, sgn):
        sgn_norm = float(torch.norm(sgn))
        sgn_unit = sgn / sgn_norm

        d_start = 0.
        if self.d_t < np.inf:
            # already have a result: the new direction is only useful if it is adversarial at d_t
            if not self.search_succ(self.get_xadv(sgn_unit, self.d_t)):
                return False
            d_end = self.d_t
        else:
            # first run, look for any adversarial distance along sgn
            d = self.lin_search(sgn)
            if d == np.inf:
                return False
            d_end = d * sgn_norm

        while (d_end - d_start) > self.tol and self.budget_left():
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(sgn_unit, d_mid)):
                d_end = d_mid
            else:
                d_start = d_mid

        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        return False

    def attack(self, sm, x_best):
        # flip the signs of one block of the current ray direction, then binary search along it
        block_num = 2 ** self.block_level
        block_size = int(np.ceil(self.dim / block_num))
        start, end = self.block_ind * block_size, min(self.dim, (self.block_ind + 1) * block_size)

        attempt = self.sgn_t.clone().view(-1)
        attempt[start:end] *= -1.
        self.binary_search(attempt.view(self.shape))

        self.block_ind += 1
        if self.block_ind == block_num or end == self.dim:
            self.block_level += 1
            self.block_ind = 0

        if self.d_t == np.inf:
            return sm, None
        if sm == 0:
            # first adversarial example found
            sm = 1
            self.query_cnt2 = self.query_cnt
        return sm, self.to_model(self.x_final)

    def stop_criteria(self, x, x_adv):
        # original early stopping (dist <= epsilon), without the extra query of the common version
        if not self.stop or x_adv is None:
            return False
        dist = torch.norm((self.x_final - self.p).flatten(), self.order)
        return float(dist) <= self.epsilon

    def untarget(self, image, label):
        self.targeted = False
        self.label = label
        self.shape = image.size()
        self.mu = self.mu.to(image.device)
        self.std = self.std.to(image.device)

        self.x = image.detach().clone()
        self.p = self.to_pixel(self.x)
        self.dim = int(np.prod(self.shape[1:]))
        # run() clamps the result to [min, max]; use the bounds of the valid pixel range
        self.min = float(self.to_model(torch.zeros_like(self.p)).min())
        self.max = float(self.to_model(torch.ones_like(self.p)).max())

        self.query_cnt = 0
        self.query_cnt2 = 0
        self.d_t = np.inf
        self.sgn_t = torch.ones_like(self.p)
        self.x_final = self.p.clone()
        self.block_level = 0
        self.block_ind = 0

        adv, q0, q1 = self.run(None)
        return adv, q0, q1, self.adaptive.query_cnt

    def target(self, image, label, example):
        raise ValueError("targeted attack is not supported yet")
