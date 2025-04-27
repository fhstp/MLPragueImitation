# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.rich import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "auto"
    """device to be used"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: object = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    buffer_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    use_targets: bool = False
    """Whether or not to use target nets"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 0
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.6
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    auto_target_entropy: bool = False
    """whether or not to choose the target entropy automatically"""
    target_entropy: float = -1.0
    """The target entropy when not chosen automatically"""


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            env.observation_space.shape[0] + env.action_space.shape[0], 32
        )
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, net=None):
        super().__init__()
        self.net = None
        if net is None:
            self.fc1 = nn.Linear(env.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc_mean = nn.Linear(32, np.prod(env.action_space.shape))
            self.fc_logstd = nn.Linear(32, np.prod(env.action_space.shape))
        else:
            self.net = net

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        if self.net is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            mean = self.fc_mean(x)
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                log_std + 1
            )  # From SpinUp / Denis Yarats
        else:
            mean, log_std = self.net(x)

        return mean, log_std

    def get_action(self, x, bias_actor=None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if bias_actor is not None:
            mean, log_std = bias_actor(x)
            std = log_std.exp()
            bias_normal = torch.distributions.Normal(mean, std)
            bias_log_prob = bias_normal.log_prob(x_t)
            # Enforcing Action Bound
            bias_log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob -= bias_log_prob.sum(1, keepdim=True)
        return action, log_prob, mean


def default_phi(x):
    return x


def default_regularizer(x):
    return x**2 / 10


class IQLearn:
    def __init__(
        self,
        env: gym.Env,
        phi: Callable[[torch.Tensor], torch.Tensor] | None = None,
        regularizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        online_size: int = 256,
        actor_net: nn.Module | None = None,
        q_cls: type = SoftQNetwork,
        sac_args: Args | dict[str, Any] | None = None,
    ):
        if sac_args is None:
            self.args = Args()
        elif type(sac_args) == dict:
            self.args = Args(**sac_args)
        elif type(sac_args) == Args:
            self.args = sac_args

        self.online_size = online_size
        self.demonstration_buffer = None

        self.set_env(env)

        if phi is None:
            self.phi = default_phi
        else:
            self.phi = phi
        if regularizer is None:
            self.regularizer = default_regularizer
        else:
            self.regularizer = regularizer

        if self.args.track:
            import wandb

            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        self.setup_writer()

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        assert isinstance(
            self.env.action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        assert isinstance(
            self.env.observation_space, gym.spaces.Box
        ), "only continuous observation space is supported"

        self.actor = Actor(self.env, actor_net).to(self.args.device)
        self.bias_actor = None
        self.qf1 = q_cls(self.env).to(self.args.device)
        self.qf2 = q_cls(self.env).to(self.args.device)
        if self.args.use_targets:
            self.qf1_target = q_cls(self.env).to(self.args.device)
            self.qf2_target = q_cls(self.env).to(self.args.device)
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
        else:
            self.qf1_target = self.qf1
            self.qf2_target = self.qf2

        # Automatic entropy tuning
        if self.args.autotune:
            if self.args.auto_target_entropy:
                self.target_entropy = -torch.prod(
                    torch.Tensor(self.env.action_space.shape).to(self.args.device)
                ).item()
            else:
                self.target_entropy = self.args.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha
            self.a_optimizer = None
        self.recreate_optimizers()

        self.env.observation_space.dtype = np.float32  # type: ignore
        self.reset_replay_buffer()
        self.start_time = time.time()

        self.n_updates = 0

    def set_bias_actor(self, actor):
        self.bias_actor = actor

    def recreate_optimizers(self):
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.args.policy_lr
        )
        if self.args.autotune:
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)

    def reset_replay_buffer(self):
        self.rb = ReplayBuffer(
            self.args.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.args.device,
            handle_timeout_termination=False,
        )

    def set_env(self, env):
        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)  # type: ignore
        self.setup_writer()

    def setup_writer(self):
        self.run_name = f"{self.env.spec.id if self.env is not None and self.env.spec is not None else ''}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self.args).items()]
                )
            ),
        )

    def set_env(self, env):
        self.env = env
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)

    def set_demonstration_buffer(self, demonstration_buffer):
        self.demonstration_buffer = demonstration_buffer

    def learn(self, timesteps: int):
        # ALGO LOGIC: put action logic here
        if self.online_size > 0:
            obs, _ = self.env.reset(seed=np.random.randint(2147483647))

        for _ in tqdm(range(timesteps)):
            self.n_updates += 1
            if self.online_size > 0:
                if self.n_updates < self.args.learning_starts:
                    action = np.array(self.env.action_space.sample())
                else:
                    action, _, _ = self.actor.get_action(
                        torch.Tensor(obs).unsqueeze(0).to(self.args.device)
                    )
                    action = action.detach().cpu().numpy()[0]

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, termination, truncated, info = self.env.step(action)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in info:
                    for info in info["final_info"]:
                        # print(
                        #     f"self.n_updates={self.n_updates}, episodic_return={info['episode']['r']}"
                        # )
                        self.writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            self.n_updates,
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            self.n_updates,
                        )
                        break

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                self.rb.add(obs, next_obs, action, reward, termination, info)  # type: ignore

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
                if termination or truncated:
                    obs, _ = self.env.reset(seed=np.random.randint(2147483647))

            # ALGO LOGIC: training.
            if self.n_updates > self.args.learning_starts:
                data = self.demonstration_buffer.sample(self.args.batch_size)

                loss, demonstration_loss, mixed_loss, regularizer_loss = (
                    self.update_critic(data)
                )
                if (
                    self.n_updates % self.args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    actor_loss, alpha_loss = self.update_policy(data)

                # update the target networks
                if (
                    self.n_updates % self.args.target_network_frequency == 0
                    and self.args.use_targets
                ):
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if self.n_updates % 100 == 0:
                    self.writer.add_scalar(
                        "losses/critic_loss",
                        loss.item(),
                        self.n_updates,
                    )
                    self.writer.add_scalar(
                        "losses/demonstration_loss",
                        demonstration_loss.item(),
                        self.n_updates,
                    )
                    self.writer.add_scalar(
                        "losses/mixed_loss", mixed_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/regularizer_loss",
                        regularizer_loss.item(),
                        self.n_updates,
                    )
                    self.writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), self.n_updates  # type: ignore
                    )
                    self.writer.add_scalar("losses/alpha", self.alpha, self.n_updates)
                    # print(
                    #     "SPS:", int(self.n_updates / (time.time() - self.start_time))
                    # )
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(self.n_updates / (time.time() - self.start_time)),
                        self.n_updates,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), self.n_updates  # type: ignore
                        )

    def get_values(self, observations, actions=None):
        if actions is None:
            actions, _, _ = self.actor.get_action(observations)

        qf1_a_values = self.qf1(observations, actions).view(-1)
        qf2_a_values = self.qf2(observations, actions).view(-1)
        return torch.min(qf1_a_values, qf2_a_values).unsqueeze(1)

    def update_critic(self, data, live_data=None):
        demonstration_loss = (
            self.get_values(data.observations, data.actions)
            - (1 - data.dones)
            * self.args.gamma
            * self.get_values(data.next_observations).detach()
        )
        mixed_loss = (
            self.get_values(data.observations)
            - (1 - data.dones)
            * self.args.gamma
            * self.get_values(data.next_observations).detach()
        )
        if live_data is not None:
            live_loss = (
                self.get_values(live_data.observations)
                - (1 - live_data.dones)
                * self.args.gamma
                * self.get_values(live_data.next_observations).detach()
            )
        else:
            live_loss = []  # hack so live_loss has len()

        data_normalizer = (len(mixed_loss) + len(live_loss)) / (len(demonstration_loss))

        regularizer_loss = self.regularizer(mixed_loss).mean()
        if live_data is not None:
            regularizer_loss += self.regularizer(live_loss).mean()  # type: ignore
        regularizer_loss += self.regularizer(demonstration_loss).mean()

        demonstration_loss = self.phi(demonstration_loss).mean()
        mixed_loss = (
            mixed_loss.mean() + (0 if live_data is None else live_loss.mean())  # type: ignore
        ) / data_normalizer
        regularizer_loss = regularizer_loss.mean()

        loss = demonstration_loss - mixed_loss - regularizer_loss
        loss = -loss  # maximize

        # optimize the model
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss, demonstration_loss, mixed_loss, regularizer_loss

    def update_policy(self, data):
        actor_loss = 0
        alpha_loss = 0
        for _ in range(
            self.args.policy_frequency
        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
            pi, log_pi, _ = self.actor.get_action(data.observations)
            qf1_pi = self.qf1(data.observations, pi)
            qf2_pi = self.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.args.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(data.observations)
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi + self.target_entropy)
                ).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        return actor_loss, alpha_loss

    def predict(self, obs: torch.Tensor | np.ndarray, deterministic: bool = False):
        if type(obs) == np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.args.device)
        obs = obs.unsqueeze(0)  # type: ignore
        action, _, mean = self.actor.get_action(obs)
        prediction = mean if deterministic else action
        prediction = prediction.detach().cpu().numpy()
        prediction = prediction[0]
        return prediction, None  # return None for consistency with sb3

    def sac_learn(self, steps):
        obs, _ = self.env.reset()
        for _ in tqdm(range(steps)):
            # ALGO LOGIC: put action logic here
            # if self.n_updates < self.args.learning_starts:
            #     action = np.array(self.env.action_space.sample())
            # else:
            action, _, _ = self.actor.get_action(
                torch.Tensor(obs).unsqueeze(0).to(self.args.device)
            )
            action = action.detach().cpu().numpy()[0]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncated, info = self.env.step(action)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if termination or truncated:
                if info is not None:
                    self.writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], self.n_updates
                    )
                    self.writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], self.n_updates
                    )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            self.rb.add(obs, next_obs, action, reward, termination, info)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if termination or truncated:
                obs, _ = self.env.reset()

            # ALGO LOGIC: training.
            if self.n_updates > self.args.learning_starts:
                data = self.rb.sample(self.args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations, self.bias_actor
                    )
                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if (
                    self.n_updates % self.args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        self.args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(
                            data.observations, self.bias_actor
                        )
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        # actor_loss = (-min_qf_pi).mean()
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (
                                -self.log_alpha.exp() * (log_pi + self.target_entropy)
                            ).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if self.n_updates % self.args.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data
                            + (1 - self.args.tau) * target_param.data
                        )

                if self.n_updates % 100 == 0:
                    self.writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf1_loss", qf1_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf2_loss", qf2_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, self.n_updates
                    )
                    self.writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), self.n_updates
                    )
                    self.writer.add_scalar("losses/alpha", self.alpha, self.n_updates)
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(self.n_updates / (time.time() - self.start_time)),
                        self.n_updates,
                    )
                    if self.args.autotune:
                        self.writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), self.n_updates
                        )
            self.n_updates += 1

    def close(self):
        self.env.close()
        self.writer.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["writer"]
        del state["env"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.writer = None
        self.env = None
