# CROP

Compact Reshaped Observation Processing
## Setup

### Requirements

- python 3.10
- cuda 11.6 or 11.7

### Installation

```sh
pip install -r requirements.txt
```

## Training

Example for training CROP in safety environments:

```python
from crop import * 
from util import TrainableAlgorithm
from ai_safety_gym import factory

# Init environment, wrapped with Radius CROP 
envs = factory(seed=42, name='DistributionalShift', spec='Train', wrapper=CROP('Radius'))
model:TrainableAlgorithm = PPO(envs=envs, path='results')

# Evaluation is done within the training loop
model.learn(total_timesteps=10e5, stop_on_reward=40)
model.save(base_path+"models/trained")
```

## Running Experiments

```sh
# Train CROP and baselines 
python -m run PPO CROP Object --env DistributionalShift Train
python -m run [PPO|A2C] [|CROP Radius|CROP Action|CROP Object|RAD] --env [Train|Maze7|Mazes7|Maze11|Mazes11]

# Use flag --test to run without writing out
# Use --path experiments/1-Train to write to path other than /results
# Display help for command line arguments 
$ python -m run -h
```

## Plotting

```sh
# Evaluation Train, Test & Heatmaps
python -m plot results/1-evaluation -m Validation Evaluation -e Train -a PPO -g env algorithm 
python -m plot results/1-evaluation --heatmap Obstacle 

# Benchmark Train & Test 
python -m plot results/2-benchmark -m Validation Evaluation -g env
