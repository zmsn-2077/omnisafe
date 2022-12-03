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
"""env_wrapper"""

import safety_gymnasium
import torch


class EnvWrapper:
    """env_wrapper"""

    def __init__(self, env_id, render_mode='none'):
        # check env_id is str
        self.env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.env_id = env_id
        self.render_mode = render_mode
        self.metadata = self.env.metadata  # TODO: change to __getattr__

        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env._max_episode_steps
        else:
            self.max_ep_len = 1000
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = None
        self.curr_o, info = self.env.reset(seed=self.seed)
        self.rand_a = True
        self.ep_steps = 1000
        self.ep_ret = 0
        self.ep_costs = 0
        self.ep_len = 0
        self.deterministic = False

    def make(self):
        """create environments"""
        return self.env

    def reset(self, seed=None):
        """reset environment"""
        self.curr_o, info = self.env.reset(seed=seed)
        return self.curr_o, info

    def render(self):
        """render environment"""
        return self.env.render()

    def set_seed(self, seed):
        """set environment seed"""
        self.seed = seed

    def set_rollout_cfgs(self, **kwargs):
        """set rollout configs"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def step(self, action):
        """engine step"""
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return next_obs, reward, cost, terminated, truncated, info

    def roll_out(self, agent, buf, logger):
        """collect data and store to experience buffer."""
        obs, _ = self.env.reset()
        # print(info)  ## need do
        ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
        for step_i in range(self.local_steps_per_epoch):
            action, value, cost_value, logp = agent.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, cost, done, truncated, info = self.step(action)
            ep_ret += reward
            ep_costs += (self.cost_gamma**ep_len) * cost
            ep_len += 1

            # Save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buf
            buf.store(
                obs=obs,
                act=action,
                rew=reward,
                val=value,
                logp=logp,
                cost=cost,
                cost_val=cost_value,
            )

            # Store values for statistic purpose
            if self.use_cost == True:
                logger.store(**{'Values/V': value, 'Values/C': cost_value})
            else:
                logger.store(**{'Values/V': value})

            # Update observation
            obs = next_obs

            timeout = ep_len == self.max_ep_len
            terminal = done or timeout or truncated
            epoch_ended = step_i == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, value, cost_value, _ = agent(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value, cost_value = 0.0, 0.0

                # Automatically compute GAE in buffer
                buf.finish_path(value, cost_value, penalty_param=float(self.penalty_param))

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len,
                            'Metrics/EpCost': ep_costs,
                        }
                    )
                ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
                obs, info = self.env.reset()

    def roll_out_off(self, ac, buf, logger, use_cost):
        """collect data and store to experience buffer."""
        for t in range(self.ep_steps):
            ep_ret = self.ep_ret
            ep_len = self.ep_len
            ep_costs = self.ep_costs
            o = self.curr_o
            a, v, cv, logp = ac.step(torch.as_tensor(o, dtype=torch.float32), self.deterministic)
            # Store values for statistic purpose
            if use_cost == True:
                logger.store(**{'Values/V': v, 'Values/C': cv})
            else:
                logger.store(**{'Values/V': v})
            if self.rand_a:
                a = self.env.action_space.sample()
            # Step the env
            o2, r, c, d, truncated, info = self.step(a)
            ep_ret += r
            ep_costs += c
            ep_len += 1
            self.ep_len = ep_len
            self.ep_ret = ep_ret
            self.ep_costs = ep_costs
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            self.curr_o = o2
            if not self.deterministic:
                d = False if ep_len >= self.max_ep_len else d
                buf.store(o, a, r, c, o2, d)
                if d or ep_len >= self.max_ep_len:
                    logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len,
                            'Metrics/EpCosts': ep_costs,
                        }
                    )
                    self.curr_o, info = self.env.reset(seed=self.seed)
                    self.ep_ret, self.ep_costs, self.ep_len = (
                        0,
                        0,
                        0,
                    )
            else:
                if d or ep_len >= self.max_ep_len:
                    logger.store(
                        **{
                            'Test/EpRet': ep_ret,
                            'Test/EpLen': ep_len,
                            'Test/EpCosts': ep_costs,
                        }
                    )
                    self.curr_o, info = self.env.reset(seed=self.seed)
                    self.ep_ret, self.ep_cost, self.ep_len = (
                        0,
                        0,
                        0,
                    )
                    return
