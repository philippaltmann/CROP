""" Registering 8 AI Safety Problems from https://arxiv.org/pdf/1711.09883.pdf 
from environment classes from ai_safety_gridworlds.environments.{env} to gym"""
from gym.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env

from .gym.env import SafetyEnv
from .gym.wrapper import SafetyWrapper
from .gym.plotting import *

SAFETY_ENVS = {
  # 6. Distributional shift: distributional_shift.py
  #    How can we detect and adapt to a data distribution that is different from the training distribution?
  "DistributionalShift": { # The agent should navigate to the goal, while avoiding the lava fields.
    "register": {
      'Train': ['Obstacle','Target'], 'Obstacle': ['Train','Target'], 'Target': ['Train','Obstacle'], 
      'Mazes7': ['Maze7'], 'Mazes9': ['Maze9'], 'Mazes11': ['Maze11'], 'Mazes13': ['Maze13'], 'Mazes15': ['Maze15'], # Mazes -> Generated nondeterministic
      'Maze7': ['Mazes7'], 'Maze9': ['Mazes9'], 'Maze11': ['Mazes11'], 'Maze13': ['Mazes13'], 'Maze15': ['Mazes15']}, # Maze -> Single deterministic maze
    "template": lambda level, steps=100: { 
      "nondeterministic": 'Mazes' in level, "max_episode_steps": steps, "kwargs": {
        "env_name": 'distributional_shift', 'level_choice': level, "max_iterations": steps 
      }
    }, 
  },
}

# Env Creation Helpers 
env_spec = lambda env: env.get_attr('env')[0].spec

env_id = lambda name, key: "{}-v{}".format(name, key) if isinstance(key, int) else "{}{}-v0".format(name, str(key).capitalize())
call = lambda f, x: {k: f(v) for k,v in x.items()} if isinstance(x, dict) else f(x) 
make = lambda name, config, generator=make_vec_env, **args: call(lambda id: generator(id, **args), call(lambda k: env_id(name, k), config)) #wrapper_kwargs
def factory(name, spec, n_train=4, wrapper=SafetyWrapper):
  if name.endswith('-Sparse'): name = name[:-7]; sparse = True 
  else: sparse = False
  assert name in SAFETY_ENVS.keys(), f'NAME Needs to be ∈ {list(SAFETY_ENVS.keys())}'
  evaluation = {f'evaluation-{i}': name for i,name in enumerate(SAFETY_ENVS[name]['register'][spec])}
  config = {'train': spec, 'test': {'validation': spec, **evaluation}}
  n_train = int(n_train); assert n_train > 0, "Please specify a number of training environments > 0"
  BASE_STAGE = {"wrapper_class": wrapper, "wrapper_kwargs": { "sparse": sparse }} 
  STAGES = { "train": { "n_envs": n_train, **BASE_STAGE}, "test": {**BASE_STAGE} }
  return { stage: make(name, config[stage], **args) for stage, args in STAGES.items() }, sparse 

# Env Registration
[register(env_id(name, spec), entry_point=SafetyEnv, **detail["template"](spec)) for name, detail in SAFETY_ENVS.items() for spec in detail["register"]]
