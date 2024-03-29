import numpy as np; import torch as th; from typing import Any, Dict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard.writer import SummaryWriter
from algorithm.logging import write_hyperparameters
from ai_safety_gym import env_spec

class EvaluationCallback(BaseCallback):
  """ Callback for evaluating an agent.
  :param model: The model to be evaluated
  :param eval_envs: A dict containing environments for testing the current model.
  :param stop_on_reward: Whether to use early stopping. Defaults to True
  :param reward_threshold: The reward threshold to stop at."""
  def __init__(self, model: BaseAlgorithm, eval_envs: dict, stop_on_reward:float=None):
    super(EvaluationCallback, self).__init__()
    self.model = model; self.eval_envs = eval_envs; self.writer: SummaryWriter = self.model.writer
    self.stop_on_reward = lambda r: (stop_on_reward and r >= stop_on_reward) or not self.model.continue_training

  def _on_training_start(self):  self.evaluate()

  def _on_rollout_end(self) -> None:
    if self.writer == None: return 
    # Uncomment for early stopping based on 100-mean training return
    # mean_return = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
    # if self.stop_on_reward(mean_return): self.continue_training = False
    if self.model.should_eval(): self.evaluate()

  def _on_step(self) -> bool: 
    """ Write timesteps to info & stop on reward threshold"""
    [info['episode'].update({'t': self.model.num_timesteps}) for info in self.locals['infos'] if info.get('episode')]
    return self.model.continue_training

  def _on_training_end(self) -> None: # No Early Stopping->Unkown, not reached (continue=True)->Failure, reached (stopped)->Success
    if self.writer == None: return 
    status = 'STATUS_UNKNOWN' if not self.stop_on_reward else 'STATUS_FAILURE' if self.model.continue_training else 'STATUS_SUCCESS'
    metrics = self.evaluate(); write_hyperparameters(self.model, list(metrics.keys()), status)

  def evaluate(self):
    """Run evaluation & write hyperparameters, results & video to tensorboard. Args:
        write_hp: Bool flag to use basic method for writing hyperparams for current evaluation, defaults to False
    Returns: metrics: A dict of evaluation metrics, can be used to write custom hparams """ 
    step = self.model.num_timesteps
    if not self.writer: return []
    metrics = {k:v for label, env in self.eval_envs.items() for k, v in self.run_eval(env, label, step).items()}
    [self.writer.add_scalar(key, value, step) for key, value in metrics.items()]; self.writer.flush()  
    return metrics

  def run_eval(self, env, label: str, step: int, eval_kwargs: Dict[str, Any]={}, write_video:bool=True):
    video_buffer, FPS, metrics = [], 10, {} # Move video frames from buffer to tensor, unsqueeze & clear buffer
    def retreive(buffer): entries = buffer.copy(); buffer.clear(); return th.tensor(np.array(entries)).unsqueeze(0)
    record_video = lambda locals, _: video_buffer.append(locals['env'].render(mode='rgb_array'))
    if 'Maze' in env_spec(env)._kwargs['level_choice']: eval_kwargs = {'n_eval_episodes': 100,'deterministic': False} | eval_kwargs
    else: eval_kwargs = {'n_eval_episodes': 1,'deterministic': True} | eval_kwargs

    r,_ = evaluate_policy(self.model, env, callback=record_video, **eval_kwargs)
    metrics[f"rewards/{label}"] = r # metrics[f"metrics/{label}_reward"] = r
    # Early stopping based on evaluation return
    if self.stop_on_reward(r) and label == 'validation': self.model.continue_training = False
    if write_video: self.writer.add_video(label, retreive(video_buffer), step, FPS) 
    # Create & write tringle heatmap plots
    [self.writer.add_figure(f'{k}_heatmap/{label}', env.envs[0].heatmap(*i), step) for k,i in self.model.heatmap_iterations.items()]  
    self.writer.flush()
    return metrics
