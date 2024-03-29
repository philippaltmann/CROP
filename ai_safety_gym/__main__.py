import argparse; import numpy as np;
import gym; from ai_safety_gym import *

parser = argparse.ArgumentParser()
env_names = [''.join([env,mode]) for env in SAFETY_ENVS.keys() for mode in ['', '-Sparse']]
parser.add_argument('env_name', type=str, help='The env to use', choices=env_names)
parser.add_argument('--play', type=str, help='Run environment in commandline.') #, action='store_true'
parser.add_argument('--plot', nargs='+', default=[], help='Save heatmap vizualisation plots.')
parser.add_argument('--seed', default=42, type=int, help='The random seed.')

args = parser.parse_args()
if args.play is not None: 
  if args.env_name.endswith('-Sparse'): assert False, 'Playing sparse envs is not supported'
  env = make(args.env_name, args.play)
  env.env_method('play')

specs = [int(s) if s.isdigit() else s for s in args.plot]
if len(specs):
  name, sparse = (args.env_name[:-7], True) if args.env_name.endswith('-Sparse') else (args.env_name, False)
  env_kwargs = {"wrapper_class": SafetyWrapper,  "wrapper_kwargs": { "sparse": sparse }}
  envs = { env_id(args.env_name, spec): make(name, spec, seed=args.seed, **env_kwargs) for spec in specs }
  reward_data = { key: np.expand_dims(np.array(env.envs[0].iterate()), axis=3) for key, env in envs.items() }
  [heatmap_3D(data, show_agent=True).write_image(f'results/0-envs/{key}-3D.pdf') for key, data in reward_data.items()]
  # reward_data = { f"{args.env_name}_{tag}": env.envs[0].iterate() for tag, env in envs['test'].items() }
  # [heatmap_2D(data, -51,49).savefig(f'results/plots/{key}.png') for key, data in reward_data.items()]
