import setuptools; import argparse
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
parser.add_argument('--stop', help='Stop at reward threshold.', action='store_true')
parser.add_argument('--path', default='results', help='The base path, defaults to `results`')
parser.add_argument('-d',  dest='device', default='cuda', choices=['cuda','cpu'])

# Get arguments 
args = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}; 

method = args.pop('method'); algorithm = eval(method.pop(0))
method = method or ['Observation', 'Full']; wrapper = eval(method[0])(*method[1:])

name, spec = 'DistributionalShift', args.pop('env')
envs, sparse = factory(name=name, spec=spec, wrapper=wrapper)
base = args.pop('path')
path = None if args.pop('test') else f"{base}/{spec}/{algorithm.__name__ }/{' '.join(method[::-1])}"

# Extract training parameters & merge model args
duration = lambda stop, spec: ([steps for key, steps in {
  'stop': {'Train': 1e6, 'Maze': 1e6}, 'time': {'Train': 15e4, 'Maze7': 1e5, 'Mazes7': 2e5, 'Maze11': 15e4, 'Mazes11': 3e5}
}['stop' if stop else 'time'].items() if key in spec][0], envs['train'].get_attr('reward_threshold')[0] if stop else False)
timesteps, reward = duration(args.pop('stop'), spec)
print(f"Training {algorithm.__name__ } {' '.join(method[::-1])} in {name} {spec} for {timesteps:.0f} steps {f'until {reward} is reached' if reward else ''}") 

#Create, train & save model 
args = {'envs': envs, 'path': path, **args}; load = args.pop('load', False);
model:TrainableAlgorithm = algorithm.load(load=load, **args) if load else algorithm(**args)
envs['train'].seed(model.seed); [env.seed(model.seed) for _,env in envs['test'].items()]
model.learn(total_timesteps=timesteps, stop_on_reward=reward, reset_num_timesteps = not load)
if path: model.save()
print("Done")