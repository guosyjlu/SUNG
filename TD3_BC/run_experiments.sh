CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env halfcheetah-random-v2 > halfcheetah-random-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env hopper-random-v2 > hopper-random-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env walker2d-random-v2 > walker2d-random-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env halfcheetah-medium-v2 > halfcheetah-medium-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env hopper-medium-v2 > hopper-medium-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env walker2d-medium-v2 > walker2d-medium-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env halfcheetah-medium-replay-v2 > halfcheetah-medium-replay-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env hopper-medium-replay-v2 > hopper-medium-replay-v2.result 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env walker2d-medium-replay-v2 > walker2d-medium-replay-v2.result 2>&1 &
