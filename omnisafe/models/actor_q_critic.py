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

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

from omnisafe.models.mlp_actor import MLPActor
from omnisafe.models.mlp_categorical_actor import MLPCategoricalActor
from omnisafe.models.mlp_cholesky_actor import MLPCholeskyActor
from omnisafe.models.mlp_gaussian_actor_off import OMLPGaussianActor
from omnisafe.models.model_utils import build_mlp_network
from omnisafe.models.online_mean_std import OnlineMeanStd
from omnisafe.models.q_critic import Q_Critic


class Actor_Q_Critic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        pi_type,
        standardized_obs: bool,
        scale_rewards: bool,
        shared_weights: bool,
        ac_kwargs: dict,
        weight_initialization_mode='kaiming_uniform',
    ) -> None:
        super().__init__()

        self.obs_shape = observation_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) if standardized_obs else None
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.ac_kwargs = ac_kwargs

        # build policy and value functions
        if isinstance(action_space, Box):
            if pi_type == 'dire':
                actor_fn = MLPActor
            elif pi_type == 'chol':
                actor_fn = MLPCholeskyActor
            else:
                actor_fn = OMLPGaussianActor
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            actor_fn = MLPCategoricalActor
            act_dim = action_space.n
        else:
            raise ValueError

        obs_dim = observation_space.shape[0]

        # Use for shared weights
        layer_units = [obs_dim] + ac_kwargs['pi']['hidden_sizes']

        activation = ac_kwargs['pi']['activation']
        if shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=activation,
                weight_initialization_mode=weight_initialization_mode,
                output_activation=activation,
            )
        else:
            shared = None
        self.pi = actor_fn(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_limit=self.act_limit,
            shared=shared,
            weight_initialization_mode=weight_initialization_mode,
            **ac_kwargs['pi'],
        )
        self.v = Q_Critic(obs_dim, act_dim, shared=shared, **ac_kwargs['val'])
        self.v_ = Q_Critic(obs_dim, act_dim, shared=shared, **ac_kwargs['val'])

    def step(self, obs, determinstic=False):
        """
        If training, this includes exploration noise!
        Expects that obs is not pre-processed.
        Args:
            obs, , description
        Returns:
            action, value, log_prob(action)
        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            if isinstance(self.pi, MLPActor):
                a = self.pi.predict(obs, determinstic=determinstic)
            else:
                a, logp_a = self.pi.predict(obs, determinstic=determinstic)
            v = self.v(obs, a)
            a = np.clip(a.numpy(), -self.act_limit, self.act_limit)

        return a, v.numpy(), logp_a.numpy()

    def anneal_exploration(self, frac):
        """update internals of actors
            1) Updates exploration parameters for Gaussian actors update log_std
        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1
        """
        if hasattr(self.pi, 'set_log_std'):
            self.pi.set_log_std(1 - frac)
