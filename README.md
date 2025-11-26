This is the official implementation of paper "A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning" (TNNLS 2025).

# SUNG

Our codebase is mainly built on top of these implementations:

- TD3+BC (https://github.com/sfujim/TD3_BC), 
- TD3 (https://github.com/sfujim/TD3),
- CQL (https://github.com/young-geng/CQL),
- BR (https://github.com/shlee94/Off2OnRL).

## Usage 

***Firstly***, pretrain the offline RL agents with offline RL algorithms, such as TD3+BC and CQL, and save the last checkpoint of policy network and value network.

***Then***, train the VAE estimator with provided scripts:

```bash
python train_vae.py
```

This will automatically save the checkpoints of the trained VAE.

***Finally***, perform online finetuning with provided scripts:

``` bash
python main.py
```

Or you can also run all the included experiments via:

```bash
bash run_experiments.sh
```

### Requirements

- Python 3.8
- PyTorch 1.10.0
- Gym 0.19.0
- MuJoCo 2.2.0
- mujoco-py 2.1.2.14
- d4rl

## Cite

Please cite our work if you find it useful:

```
@article{sung,
  author={Guo, Siyuan and Sun, Yanchao and Hu, Jifeng and Huang, Sili and Chen, Hechang and Piao, Haiyin and Sun, Lichao and Chang, Yi},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={A Simple Unified Uncertainty-Guided Framework for Offline-to-Online Reinforcement Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TNNLS.2025.3633997}
}
```
