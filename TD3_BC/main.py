import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import utils
import TD3
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, mean, std, evaluation_step, seed_offset=2023, eval_episodes=10):
    begin = time.perf_counter()
    eval_env = gym.make(env_name)
    eval_env.seed(seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    end = time.perf_counter()

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    
    evaluation_outputs = {
        # "evaluation/step": int(evaluation_step),
        "evaluation/time": end - begin,
        "evaluation/D4RL score": d4rl_score,
        "evaluation/Return": avg_reward
    }
    return evaluation_outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="SUNG-TD3+BC")		# Policy name
    parser.add_argument("--env", default="hopper-medium-replay-v2")    	# OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)    			# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e3, type=int)    	# How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", default=True)    			# Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")    		# Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)    			# Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)    	# Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)    				# Discount factor
    parser.add_argument("--tau", default=0.005)    					# Target network update rate
    parser.add_argument("--policy_noise", default=0.2)    			# Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)    			# Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)    	# Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--normalize", default=True)
    # OORB
    parser.add_argument("--offline_size", default=int(3e6))
    parser.add_argument("--online_size", default=int(2e4))
    parser.add_argument("--p", default=0.1)
    # Density Guided Exploration
    parser.add_argument("--strategy", default="prob")		            # Choice of default, greedy and prob
    parser.add_argument("--candidate_num", default=100)
    parser.add_argument("--topk", default=10)
    parser.add_argument("--order", default="qu")
    # Density Constrained Exploitation
    parser.add_argument("--constrained", default="greedy")              # Choice of default, greedy and prob
    parser.add_argument("--filter_prob", default=0.05)
    parser.add_argument("--lamda", default=0.05)
    parser.add_argument("--annealing", default=False)                   # Anealing     
    parser.add_argument("--lb_percent", default=0.0)                    # Achieving Lower Bound Percent
           
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
        
    # Loading wandb
    variant = vars(args)

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "expl_noise": args.expl_noise,
        # Optimistic Exploration
        "strategy": args.strategy,
        "candidate_num": args.candidate_num,
        "topk": args.topk,
        "order": args.order,
        # Adaptive Exploitation
        "constrained": args.constrained,
        "filter_prob": args.filter_prob,
        "lamda": args.lamda,
    }

    # Initialize policy
    policy = TD3.TD3(**kwargs)
        
    # Initialize vae model
    load_file_path = f"../models/vae_{args.env[:-3]}_100000.pt"
    policy.vae.load_state_dict(torch.load(load_file_path))

    if args.load_model != "":
        policy_file = "TD3_BC_"+args.env+"_" + str(args.seed) if args.load_model == "default" else args.load_model
        policy.load(f"../offline_models/{policy_file}")
        print(f"Model Loaded with {policy_file}!")

    replay_buffer = utils.OORB(state_dim, action_dim, args.offline_size, args.online_size, args.p)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.state_mean, replay_buffer.state_std
    else:
        mean, std = 0, 1

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    evaluations = []
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        normalized_state = (np.array(state).reshape(1, -1) - mean)/std
            
        if args.strategy == "default":
            action = (
				policy.select_action(normalized_state)
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
		    ).clip(-max_action, max_action)
        else:
            action = policy.select_exploration_action(normalized_state)
        action = action.clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
            
        # Filter annealing
        if args.annealing:
            policy.lamda = args.lamda - (args.lamda - args.lamda * args.lb_percent) * t / args.max_timesteps

        # Train agent after collecting sufficient data
        training_outputs = policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluation_outputs = eval_policy(policy, args.env, mean, std, (t+1)//args.eval_freq)
            if args.save_model:
                policy.save(f"./models/{file_name}")
            outputs = {**training_outputs, **evaluation_outputs}
            print(outputs)

