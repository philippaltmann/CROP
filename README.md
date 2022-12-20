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

envs = factory(seed=42, name='DistributionalShift')
model:TrainableAlgorithm = # TODO

# TODO: init crop wrapper ... 
# Evaluation is done within the training loop
model.learn(total_timesteps=10e5, stop_on_reward=40)
model.save(base_path+"models/trained")
```

## Running Experiments

```sh
# Train CROP and baselines 
python -m run PPO CROP Object --env DistributionalShift Train
python -m run [PPO|SAC] --env DistributionalShift Train

# Run without writing out
--test 

# Write to path other than /results
--path experiments/1-Train

# Display help for command line arguments 
$ python -m run -h
```

## Plotting

```sh
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
python -m plot experiments/1-Train -e TrainingDense -a PPO --heatmap 0

# Generate Adaptation plots:
python -m plot experiments/2-Adapt -b experiments/1-Train -m Return -g env 
python -m plot experiments/2-Adapt --mergeon algorithm --eval 1 -e TargetShift
python -m plot experiments/2-Adapt --mergeon algorithm --eval 3 -e GoalShift
```


## Envs

Run
python -m ai_safety_gym.environments.distributional_shift --level Mazes9