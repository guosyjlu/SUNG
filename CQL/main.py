from copy import deepcopy
import numpy as np
import gym
import torch
import d4rl
import argparse
from conservative_sac import ConservativeSAC
from model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from utils import define_flags_with_default, set_random_seed, get_user_flags, ReplayBuffer, OORB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_policy(policy, env_name, seed_offset=2023, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = np.array(state).reshape(1,-1)
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	metrics = {"evaluation/D4RL score": d4rl_score, "evaluation/Return": avg_reward}
	return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
	# Experiment
    parser.add_argument("--policy", default="SUNG-CQL")
    parser.add_argument("--env", default="hopper-medium-replay-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_model", default=True)
    parser.add_argument("--load_model", default=True)
	# SAC
    parser.add_argument("--reward_scale", default=1.0, type=float)                  # 10.0 for AntMaze
    parser.add_argument("--reward_bias", default=0.0, type=float)                   # -5.0 for AntMaze
    parser.add_argument("--clip_action", default=0.99, type=float)
    parser.add_argument("--policy_arch", default='256-256', type=str)
    parser.add_argument("--qf_arch", default='256-256', type=str)                   # 256-256-256 for AntMaze
    parser.add_argument("--orthogonal_init", default=False)                         # True for AntMaze
    parser.add_argument("--policy_log_std_multiplier", default=1.0, type=float)     # 0.0 for AntMaze
    parser.add_argument("--policy_log_std_offset", default=-1.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_epochs", default=int(1e5), type=int)
    parser.add_argument("--bc_epochs", default=0, type=int)                         # 20 for AntMaze
    parser.add_argument("--n_train_step_per_epoch", default=1, type=int)
    parser.add_argument("--eval_period", default=int(1e3), type=int)    
    # Ours
    parser.add_argument("--exploration", default="prob")                            # Choice of default, greedy, prob
    parser.add_argument("--candidate_num", default=100, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--constrained", default="prob")                            # Choice of default, greedy, prob
    parser.add_argument("--order", default="uq")                                    # Choice of uq or qu
    parser.add_argument("--lamda", default=0.1, type=float)                         # 0.05 for halfcheetah-random-v2
    parser.add_argument("--prob", default=0.1, type=float)                          # 0.05 for halfcheetah-random-v2
    FLAGS = parser.parse_args()
    
    variant = vars(FLAGS)
    
    set_random_seed(FLAGS.seed)
        
    file_name = f"{FLAGS.policy}_{FLAGS.env}_{FLAGS.seed}"
    env = gym.make(FLAGS.env)
    dataset = d4rl.qlearning_dataset(env)
        
    # Note that here is the difference 
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    replay_buffer = OORB(state_dim, action_dim, int(3e6), int(2e4), 0.1)
    replay_buffer.convert_D4RL(dataset)
    mean, std = replay_buffer.mean, replay_buffer.std

    policy = TanhGaussianPolicy(
        state_dim,
        action_dim,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )
        
    qf1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)
    qf2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)
        
    kwargs = {
        'target_entropy': -np.prod(env.action_space.shape).item(),
        'use_cql': True        # CQL for online finetuning
    }
    if FLAGS.constrained == "default":
        kwargs['use_cql'] = False       # SAC for online finetuning
        
    sac = ConservativeSAC(kwargs, policy, qf1, qf2, target_qf1, target_qf2, state_dim, action_dim, max_action)
    if FLAGS.load_model:
        sac.load(filename=f"../offline_models/CQL_{FLAGS.env}_{FLAGS.seed}")
    sac.vae.load_state_dict(torch.load(f"../models/vae_{FLAGS.env[:-3]}_100000.pt"))
    sac.torch_to_device(device)
        
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    for epoch in range(FLAGS.n_epochs): 
        episode_timesteps += 1
        action = sac.select_exploration_action(
            np.array(state).reshape(1, -1),
            FLAGS.exploration,
            FLAGS.order,
            FLAGS.candidate_num,
            FLAGS.topk,
            mean, std
        ).clip(-max_action, max_action)
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
            
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
            
        state = next_state
        episode_reward += reward
            
        train_metrics = sac.train(replay_buffer, FLAGS.batch_size, FLAGS.constrained, FLAGS.lamda, FLAGS.prob,  bc=False)

        if done:
            print(f"Total T: {epoch + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
        if (epoch + 1) % FLAGS.eval_period == 0:
            print(f"Epoch: {epoch + 1}")
            print(train_metrics)
            evaluation_metrics = eval_policy(sac, FLAGS.env)
            if FLAGS.save_model: sac.save(f"./models/{file_name}")
            metrics = {**train_metrics, **evaluation_metrics}
            print(metrics)