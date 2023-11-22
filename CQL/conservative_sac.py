from ml_collections import ConfigDict
import numpy as np
import torch
import copy
from torch import nn as nn
import torch.nn.functional as F
from vae import VAE
from model import Scalar, soft_target_update, extend_and_repeat
from utils import prefix_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConservativeSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0                   # SAC entropy temparature
        config.use_automatic_entropy_tuning = True      # whether SAC entropy temperature
        config.backup_entropy = False                   # Bellman Backup does not contain \log\pi 
        config.target_entropy = 0.0                     # Target Entropy
        config.policy_lr = 1e-4                         # Policy Learning Rate = 1e-4
        config.qf_lr = 3e-4                             # Value Learning Rate = 3e-4
        config.optimizer_type = 'adam'                  # Optimizer
        config.soft_target_update_rate = 5e-3           # Target Update Rate = 5e-3
        config.target_update_period = 1                 # Target Update Frequencey = 1
        config.use_cql = True                           # Whether CQL (for offline RL)
        config.cql_n_actions = 10                       # CQL Sampled Action Num = 10 for calculation of log-exp
        config.cql_importance_sample = True             # CQL Importance Sample
        config.cql_lagrange = False                     # CQL entropy temperature   (False for MuJoco, True for AntMaze)
        config.cql_target_action_gap = 1.0              # CQL Target Action Gap
        config.cql_temp = 1.0                           # CQL Temperature
        config.cql_min_q_weight = 5.0                   # CQl min Q weight \alpha
        config.cql_max_target_backup = False            # CQL max target backup     (False for Mujoco, True for AntMaze, but actually backup_entropy=False)
        config.cql_clip_diff_min = -np.inf              # -200 for AntMaze, -np.inf for MuJoCo
        config.cql_clip_diff_max = np.inf

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2, state_dim, action_dim, max_action):
        self.config = ConservativeSAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )
        
        self.vae = VAE(
            state_dim, action_dim, latent_dim=action_dim*2, max_action=max_action, hidden_dim=750
        ).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mean, _ = self.policy(state, deterministic=True)
        return mean.cpu().data.numpy().flatten()
    
    def select_exploration_action(self, state, strategy="default", order="uq", candidate_num=100, topk=10, mean=None, std=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if strategy == "default":
            action, _ = self.policy(state, deterministic=False)
        else:
            states = state.repeat(candidate_num, 1)
            candidate_action, _ = self.policy(states, deterministic=False)
            if order == "uq":
                normalized_states = (states - mean)/std
                Q = self.vae.elbo_loss(normalized_states, candidate_action, beta=0.5).detach()
                Q, idx = torch.topk(Q, topk)
                states, candidate_action = states[idx], candidate_action[idx]
                density_uncertainty = torch.min(
                    self.qf1(states, candidate_action),
                    self.qf2(states, candidate_action)
                )
            elif order == "qu":
                Q = torch.min(
                self.qf1(states, candidate_action),
                self.qf2(states, candidate_action),
                )
                Q, idx = torch.topk(Q, topk)
                states, candidate_action = states[idx], candidate_action[idx]
                normalized_states = (states - mean)/std
                density_uncertainty = self.vae.elbo_loss(normalized_states, candidate_action, beta=0.5).detach()
            else:
                raise NotImplementedError("Order is not implemented!")
            if strategy == "greedy":
                idx = torch.argmax(density_uncertainty)
                action = candidate_action[idx]
            elif strategy == "e-greedy":
                if np.random.random() > self.epsilon:
                    idx = np.random.choice(self.topk)
                else:
                    idx = torch.argmax(density_uncertainty)
                action = candidate_action[idx]
            elif strategy == "prob":
                probs = F.softmax(density_uncertainty)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()
                action = candidate_action[idx]
            else:
                raise NotImplementedError("The selected strategy is not implemented yet!")
        return action.cpu().data.numpy().flatten()
    
    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, replay_buffer, batch_size=256, strategy="default", lamda=0.05, prob=0.05, bc=False):
        self._total_steps += 1
        
        observations, actions, next_observations, rewards, not_dones = replay_buffer.sample(batch_size)
        normalized_observations = (observations - replay_buffer.mean) / replay_buffer.std

        new_actions, log_pi = self.policy(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        if bc:
            log_probs = self.policy.log_prob(observations, actions)
            policy_loss = (alpha*log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.qf1(observations, new_actions),
                self.qf2(observations, new_actions),
            )
            policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        if self.config.cql_max_target_backup:
            new_next_actions, next_log_pi = self.policy(next_observations, repeat=self.config.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_qf1(next_observations, new_next_actions),
                    self.target_qf2(next_observations, new_next_actions),
                ),
                dim=-1
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.policy(next_observations)
            target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions),
                self.target_qf2(next_observations, new_next_actions),
            )

        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        td_target = rewards.flatten() + not_dones.flatten() * self.config.discount * target_q_values

        qf1_loss = F.mse_loss(q1_pred, td_target.detach())
        qf2_loss = F.mse_loss(q2_pred, td_target.detach())


        ### CQL
        if not self.config.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            
            # Uncertainty-Guided Exploitation
            uncertainty = self.vae.elbo_loss(normalized_observations, actions).detach()
            if strategy == "greedy":
                _, idx = torch.topk(uncertainty, k=int(batch_size*prob))
            elif strategy == "prob":
                probs = F.softmax(uncertainty)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample([int(batch_size*prob)])
            else:
                raise NotImplementedError("Not Implemented Yet!")
            
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            filter_num = int(batch_size*prob)
            cql_random_actions = actions.new_empty((filter_num, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            cql_current_actions, cql_current_log_pis = self.policy(observations[idx], repeat=self.config.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_observations[idx], repeat=self.config.cql_n_actions)
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

            cql_q1_rand = self.qf1(observations[idx], cql_random_actions)
            cql_q2_rand = self.qf2(observations[idx], cql_random_actions)
            cql_q1_current_actions = self.qf1(observations[idx], cql_current_actions)
            cql_q2_current_actions = self.qf2(observations[idx], cql_current_actions)
            cql_q1_next_actions = self.qf1(observations[idx], cql_next_actions)
            cql_q2_next_actions = self.qf2(observations[idx], cql_next_actions)

            cql_cat_q1 = torch.cat(
                [cql_q1_rand, torch.unsqueeze(q1_pred[idx], 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand, torch.unsqueeze(q2_pred[idx], 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
            )
            cql_std_q1 = torch.std(cql_cat_q1, dim=1)
            cql_std_q2 = torch.std(cql_cat_q2, dim=1)

            if self.config.cql_importance_sample:
                random_density = np.log(0.5 ** action_dim)
                cql_cat_q1 = torch.cat(
                    [cql_q1_rand - random_density,
                     cql_q1_next_actions - cql_next_log_pis.detach(),
                     cql_q1_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand - random_density,
                     cql_q2_next_actions - cql_next_log_pis.detach(),
                     cql_q2_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )

            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config.cql_temp, dim=1) * self.config.cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config.cql_temp, dim=1) * self.config.cql_temp

            """Subtract the log likelihood of data"""
            cql_qf1_diff = torch.clamp(
                cql_qf1_ood - q1_pred[idx],
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean() * lamda
            cql_qf2_diff = torch.clamp(
                cql_qf2_ood - q2_pred[idx],
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean() * lamda

            if self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                alpha_prime_loss = observations.new_tensor(0.0)
                alpha_prime = observations.new_tensor(0.0)


            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        
        # VAE training
        recon, mean, std = self.vae(normalized_observations, actions)
        train_states_actions = torch.cat([normalized_observations, actions], dim=-1)
        recon_loss = F.mse_loss(recon, train_states_actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )


        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

        if self.config.use_cql:
            metrics.update(prefix_metrics(dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            ), 'cql'))

        return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps
    
    def save(self, filename):
        torch.save(self.qf1.state_dict(), filename + "_qf1")
        torch.save(self.qf2.state_dict(), filename + "_qf2")
        torch.save(self.qf_optimizer.state_dict(), filename + "_qf_optimizer")
        
        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")


    def load(self, filename):
        self.qf1.load_state_dict(torch.load(filename + "_qf1"))
        self.qf2.load_state_dict(torch.load(filename + "_qf2"))
        # self.qf_optimizer.load_state_dict(torch.load(filename + "_qf_optimizer"))
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.policy.load_state_dict(torch.load(filename + "_policy"))
        # self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))
