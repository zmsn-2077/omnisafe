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
"""Implementation of the Natural PG algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.policy_gradient import PolicyGradient
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class NaturalPG(PolicyGradient):
    """Class-specific functions"""

    def __init__(self, algo: str = 'npg', **cfgs):
        PolicyGradient.__init__(self, algo=algo, **cfgs)
        cfgs = cfgs['cfgs']
        self.cg_damping = cfgs['cg_damping']
        self.cg_iters = cfgs['cg_iters']
        self.target_kl = cfgs['target_kl']
        self.fvp_obs = cfgs['fvp_obs']

    def search_step_size(self, step_dir, g_flat, p_dist, data):
        """
        Natural Policy Gradient use full step_size
        """
        accept_step = 1
        return step_dir, accept_step

    def algorithm_specific_logs(self):
        self.logger.log_tabular('Misc/AcceptanceStep')
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/FinalStepNorm')
        self.logger.log_tabular('Misc/gradient_norm')
        self.logger.log_tabular('Misc/xHx')
        self.logger.log_tabular('Misc/H_inv_g')

    def Fvp(self, p):
        """
        Build the Hessian-vector product based on an approximation of the KL-divergence.
        For details see John Schulman's PhD thesis (pp. 40) http://joschu.net/docs/thesis.pdf
        """
        self.actor_critic.pi.net.zero_grad()
        q_dist = self.actor_critic.pi.dist(self.fvp_obs)
        with torch.no_grad():
            p_dist = self.actor_critic.pi.dist(self.fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.actor_critic.pi.net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * p).sum()
        grads = torch.autograd.grad(kl_p, self.actor_critic.pi.net.parameters(), retain_graph=False)
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        # average --->
        distributed_utils.mpi_avg_torch_tensor(flat_grad_grad_kl)
        return flat_grad_grad_kl + p * self.cg_damping

    def update(self):
        """
        Update actor, critic, running statistics
        """
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Update Policy Network
        self.update_policy_net(data)
        # Update Value Function
        self.update_value_net(data=data)
        if self.cfgs['use_cost']:
            self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_policy_net(self, data):
        """update policy network"""
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.actor_critic.pi.net)
        self.actor_critic.pi.net.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = distributed_utils.mpi_avg(loss_pi.item())
        loss_v = self.compute_loss_v(data['obs'], data['target_v'])
        self.loss_v_before = distributed_utils.mpi_avg(loss_v.item())
        p_dist = self.actor_critic.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.pi.net)
        g_flat = get_flat_gradients_from(self.actor_critic.pi.net)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.search_step_size(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            data=data,
        )

        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.pi.net, new_theta)

        with torch.no_grad():
            q_dist = self.actor_critic.pi.dist(data['obs'])
            kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(data=data)

        self.logger.store(
            **{
                'Values/Adv': data['act'].numpy(),
                'Entropy': pi_info['ent'],
                'KL': kl,
                'PolicyRatio': pi_info['ratio'],
                'Loss/Pi': self.loss_pi_before,
                'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/StopIter': 1,
                'Misc/FinalStepNorm': torch.norm(final_step_dir).numpy(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(g_flat).numpy(),
                'Misc/H_inv_g': x.norm().item(),
            }
        )
