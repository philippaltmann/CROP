import setuptools; import argparse; import os; import random
from ai_safety_gym import factory, SafetyWrapper; 
from algorithm import TrainableAlgorithm
from benchmark import *
from crop import *

# General Arguments
parser = argparse.ArgumentParser()
parser.add_argument('method', nargs='+', help='The algorithm to use', choices=[*ALGS, *CROPS])
parser.add_argument( '--env', nargs='+', default=[], metavar="Environment", help='The name and spec and of the safety environments to train and test the agent. Usage: --env NAME, CONFIG, N_TRAIN, N_TEST')
parser.add_argument('-s', dest='seed', type=int, help='The random seed. If not specified a free seed [0;999] is randomly chosen')
parser.add_argument('--load', type=str, help='Path to load the model.')
parser.add_argument('--test', help='Run in test mode (dont write log files).', action='store_true')
parser.add_argument('--path', default='results', help='The base path, defaults to `results`')

# Training Arguments
parser.add_argument('-t', dest='timesteps', type=float, help='Number of timesteps to learn the model (eg 10e4)')
parser.add_argument('-ts', dest='maxsteps', type=float, default=10e5, help='Maximum timesteps to learn the model (eg 10e4), using early stopping')
parser.add_argument('--reward-threshold', type=float, help='Threshold for 100 episode mean return to stop training.')

# Policy Optimization Arguments
parser.add_argument('--n-steps', type=int, help='The length of rollouts to perform policy updates on')

# Get arguments 
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}; 
_default = lambda d,c: c + d[len(c):]

method = args.pop('method'); wrapper, name = SafetyWrapper, '' 
algorithm, model = eval(method.pop(0)), None
if len(method): wrapper, name = eval(method[0])(*method[1:])

# Generate Envs and Algorithm TODO: remove n ?
env = dict(zip(['name','spec'], _default(['DistributionalShift','Train'], args.pop('env'))))
envs, sparse = factory(wrapper=wrapper, **env)

path = args.pop('path')
if args.pop('test'): path = None
else: path = f"{path}/{'/'.join(list(env.values()))}/{algorithm.__name__ } {' '.join(method)}"
# else: path = f"{path}/{'/'.join(list(env.values()))}/{algorithm.__name__ }{name}"

# Extract training parameters & merge model args
reward_threshold = envs['train'].get_attr('reward_threshold')[0] 
stop_on_reward = args.pop('reward_threshold',reward_threshold) if any(arg in ['maxsteps', 'reward_threshold'] for arg in args) else None
timesteps = args.pop('timesteps', args.pop('maxsteps'))
print(f"Training {algorithm.__name__ } {name} in {' '.join(list(env.values()))}") 
print(f"Stopping training at threshold {stop_on_reward}") if stop_on_reward else print(f"Training for {timesteps} steps")

#Create, train & save model 
args = {'envs': envs, 'path': path, **args} # 'seed': seed,
# 'silent': path is None,
load = args.pop('load', False); #load = base_path(seed,load) if load else False
model:TrainableAlgorithm = algorithm.load(load=load, **args) if load else algorithm(**args) #device='cpu',

envs['train'].seed(model.seed); [env.seed(model.seed) for _,env in envs['test'].items()]
model.learn(total_timesteps=timesteps, stop_on_reward=stop_on_reward, reset_num_timesteps = not load)
if path: model.save()
