# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the PPO-Lag algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange


@registry.register
class PPOLag(PolicyGradient, Lagrange):
    """specific functions"""

    def __init__(self, algo='ppo-lag', clip=0.2, **cfgs):
        """Initialize PPO-Lag algorithm."""
        self.clip = clip
        PolicyGradient.__init__(self, algo=algo, **cfgs)
        Lagrange.__init__(self, **self.cfgs['lagrange_cfgs'])

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/Lagrange', self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        """Compute policy loss."""
        dist, _log_p = self.actor_critic.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Ensure that Lagrange Multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += (
            penalty * (torch.max(ratio * data['cost_adv'], ratio_clip * data['cost_adv'])).mean()
        )
        loss_pi /= 1 + penalty

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        """Update."""
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(Jc)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)
