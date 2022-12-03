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

import torch

from omnisafe.models.actor_critic import ActorCritic
from omnisafe.models.critic import Critic


class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic is a wrapper around ActorCritic that adds a cost critic to the model."""

    def __init__(
        self,
        observation_space,
        action_space,
        scale_rewards,
        standardized_obs,
        model_cfgs,
    ):
        ActorCritic.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            standardized_obs=standardized_obs,
            scale_rewards=scale_rewards,
            ac_kwargs=model_cfgs.ac_kwargs,
            shared_weights=model_cfgs.shared_weights,
        )

        self.cost_critic = Critic(
            obs_dim=self.obs_shape[0],
            shared=None,
            activation=model_cfgs.ac_kwargs.val.activation,
            hidden_sizes=model_cfgs.ac_kwargs.val.hidden_sizes,
        )

    def step(self, obs, deterministic=False):
        """Produce action, value, log_prob(action).
        If training, this includes exploration noise!

        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: do the updates at the end of batch!
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            value = self.reward_critic(obs)
            cost_value = self.cost_critic(obs)

            action, logp_a = self.actor.predict(obs, deterministic=deterministic)

        return action.numpy(), value.numpy(), cost_value.numpy(), logp_a.numpy()
