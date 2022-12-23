import setuptools; import argparse; import os; import random
from ai_safety_gym import factory; 
from algorithm import TrainableAlgorithm
from benchmark import *; from crop import *

# General Arguments
parser = argparse.ArgumentParser()
parser.add_argument('method', nargs='+', help='The algorithm to use', choices=[*ALGS, *CROPS])
parser.add_argument( '--env', default='Train', metavar="Environment", help='The spec and of the safety environments to train and test the agent.')
# parser.add_argument( '--env', nargs='+', default=[], metavar="Environment", help='The name and spec and of the safety environments to train and test the agent. Usage: --env NAME, CONFIG, N_TRAIN, N_TEST')
parser.add_argument('-s', dest='seed', type=int, help='The random seed. If not specified a free seed [0;999] is randomly chosen')
parser.add_argument('--load', type=str, help='Path to load the model.')
parser.add_argument('--test', help='Run in test mode (dont write log files).', action='store_true')
parser.add_argument('--path', default='results', help='The base path, defaults to `results`')

# Training Arguments
parser.add_argument('-t', dest='timesteps', type=float, help='Number of timesteps to learn the model (eg 10e4)')
parser.add_argument('-ts', dest='maxsteps', type=float, default=10e5, help='Maximum timesteps to learn the model (eg 10e4), using early stopping')

# Get arguments 
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}; 

method = args.pop('method'); method.append('FullyObservable') if len(method)<2 else None
algorithm, wrapper = eval(method.pop(0)), eval(method[0])(*method[1:])
print(algorithm)
print(wrapper)

name, spec = 'DistributionalShift', args.pop('env')
envs, sparse = factory(name=name, spec=spec, wrapper=wrapper)
path = None if args.pop('test') else f"{args.pop('path')}/{spec}/{algorithm.__name__ }/{' '.join(method)}"

# Extract training parameters & merge model args
stop_on_reward = envs['train'].get_attr('reward_threshold')[0] if 'maxsteps'in args else None
timesteps = args.pop('timesteps', args.pop('maxsteps'))
print(f"Training {algorithm.__name__ } {name} in {name} {spec}") 
print(f"Stopping training at threshold {stop_on_reward}") if stop_on_reward else print(f"Training for {timesteps} steps")

#Create, train & save model 
args = {'envs': envs, 'path': path, **args} # 'seed': seed, # 'silent': path is None,
load = args.pop('load', False); #load = base_path(seed,load) if load else False
model:TrainableAlgorithm = algorithm.load(load=load, **args) if load else algorithm(**args) #device='cpu',

envs['train'].seed(model.seed); [env.seed(model.seed) for _,env in envs['test'].items()]
model.learn(total_timesteps=timesteps, stop_on_reward=stop_on_reward, reset_num_timesteps = not load)
if path: model.save()
