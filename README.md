# DIRECT

Discriminative Reward Co-Training

## Setup

### Requirements

- python 3.8.6
- cuda 11.3 or 10.2

### Installation

```sh
pip install -r requirements.txt
```

## Training

Example for training DIRECT in safety environments:

```python
from direct import DIRECT 
from util import TrainableAlgorithm
from ai_safety_gym import factory

envs = factory(seed=42, name='DistributionalShift')
model:TrainableAlgorithm = DIRECT(envs=envs, seed=42)

# Evaluation is done within the training loop
model.learn(total_timesteps=10e5, stop_on_reward=40)
model.save(base_path+"models/trained")

reload = algorithm.load(envs=envs, path=base_path+"models/trained")
reload.learn(total_timesteps=512, reset_num_timesteps=False)
```

## Running Experiments

```sh
# Train DIRECT and baselines 
python -m run DIRECT --env DistributionalShift 0 --path experiments/1-Train
python -m run [A2C|DQN|PPO|SIL] --env DistributionalShift 0 --path experiments/1-Train

# Reloading in Shifted Obstacle and Shifted Target Envs 
python -m run [DIRECT|A2C|DQN|PPO|SIL] --env DistributionalShift 1 --path experiments/2-Adapt --load <Training Path> -s <Seed to load> 
python -m run [DIRECT|A2C|DQN|PPO|SIL] --env DistributionalShift 2 --path experiments/2-Adapt --load <Training Path> -s <Seed to load> 

# Display help for command line arguments 
$ python -m run -h
```

## Plotting

```sh
# New 
# Evalutaition 
python -m plot results/DistributionalShift -m Evaluation -g env -e Train

python -m plot results/DistributionalShift -e Train -a PPO CROP Action --heatmap Obstacle

# TODO: fix
python -m plot eresults/DistributionalShift -e Train --mergeon algorithm --eval 0 1



# Old
# Generate Env plots: 
python -m ai_safety_gym DistributionalShift --plot 0 1 3

# Generate Training plots:
python -m plot results/Train -m Return -g env 

python -m plot experiments/1-Train -m Return -g env 
python -m plot experiments/1-Train --mergeon algorithm --eval 0 1
python -m plot experiments/1-Train -e TrainingDense -a DIRECT --heatmap 0

# Generate Adaptation plots:
python -m plot experiments/2-Adapt -b experiments/1-Train -m Return -g env 
python -m plot experiments/2-Adapt --mergeon algorithm --eval 1 -e TargetShift
python -m plot experiments/2-Adapt --mergeon algorithm --eval 3 -e GoalShift
```


## Envs

Run
python -m ai_safety_gym.environments.distributional_shift --level Mazes9