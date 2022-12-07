import os

from wrapper.wrapper_3x3 import Wrapper_3x3
from wrapper.wrapper_5x5 import Wrapper_5x5
from wrapper.wrapper_augment_test import Wrapper_augment_test
from wrapper.wrapper_augment_training import Wrapper_augment_training
from wrapper.wrapper_lrou import Wrapper_lrou
from wrapper.wrapper_relative_coordinates import Wrapper_relative_coordinates
from wrapper.wrapper_reward import Wrapper_reward
from wrapper.wrapper_standard import Wrapper_standard
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import load_results, Monitor
from stable_baselines3 import A2C, DDPG, DQN, HER, SAC, TD3, PPO
import ai_safety_gridworlds.helpers.factory
from typing import Callable
from scripts.Enums import *
from scripts.visualize import *
from scripts.GridMap import *


def get_top_models(logDirs, percentage):
    """
    get top performing agents in Training
    :param logDirs: folder to choose agents from
    :param percentage: top percentage from 0.0 to 1
    """
    data_frames = []
    for folder in logDirs:
        data_frame = load_results(folder)
        y_var = data_frame.r.values
        data_frames.append((y_var[-100:].mean(), folder))
    data_frames.sort(key=lambda tup: tup[0])

    data_remove = (1 - percentage) * len(data_frames)

    return [i[1] for i in data_frames[round(data_remove):]]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """

        return max(progress_remaining - 0.1, 0) * initial_value

    return func


environment_name = 'distributional_shift'

LogDirs = []


def Train_PPO(log_paths):
    """
    Starts PPO training in Environment 0 for 'number_agents' agents for every path in 'log_paths'
	
    :param log_paths: [(path,wrapper),(path,wrapper),...] Paths and wrapper 
    """
    number_agents = 1
    for log_path_index in range(len(log_paths)):
        env = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name, is_testing=False)
        env.reset()

        wrapped_env = get_wrapped_env(env, log_paths[log_path_index][1])

        for i in range(0, number_agents):
            #create a path/folder for every agent 
            path = log_paths[log_path_index][0] + str(i)

            os.mkdir(path)
            monitor = Monitor(wrapped_env, path)

            if log_paths[log_path_index][1] == Wrapper.relative:
                model = PPO('MultiInputPolicy', monitor, verbose=1, learning_rate=linear_schedule(0.0005), max_grad_norm=25,
                    ent_coef=0.1, gamma=1) 
            else:
                model = PPO('MlpPolicy', monitor, verbose=1, learning_rate=linear_schedule(0.0005), max_grad_norm=25,
                    ent_coef=0.1, gamma=1)  
            model.learn(total_timesteps=100_000, eval_log_path=path)
            model.save(path=path + "/PPO")

			
def Train_PPO_early_stop(log_paths):
    """
    Starts PPO training in Environment 0 for number_agents agents for every path in log_paths
    with early stopping
	
    :param log_paths: [(path,wrapper),(path,wrapper),...] Paths and wrapper 
    """
    number_agents = 10
    early_stop_threshold = 25
    for log_path_index in range(len(log_paths)):
        env = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name, is_testing=False)
        env.reset()

        wrapped_env = get_wrapped_env(env, log_paths[log_path_index][1])

        for i in range(0, number_agents):
            #create a path/folder for every agent 
            path = log_paths[log_path_index][0] + str(i)

            os.mkdir(path)
            monitor = Monitor(wrapped_env, path)
            
            if log_paths[log_path_index][1] == Wrapper.relative:
                model = PPO('MultiInputPolicy', monitor, verbose=1, learning_rate=linear_schedule(0.0005), max_grad_norm=25,
                    ent_coef=0.1, gamma=1) 
            else:
                model = PPO('MlpPolicy', monitor, verbose=1, learning_rate=linear_schedule(0.0005), max_grad_norm=25,
                    ent_coef=0.1, gamma=1)  
				
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=early_stop_threshold, verbose=1)
            eval_callback1 = EvalCallback(monitor, callback_on_new_best=callback_on_best, verbose=1,n_eval_episodes=25,eval_freq=5000,deterministic=False)
            model.learn(total_timesteps=100_000,eval_log_path=path,callback=eval_callback1)
            model.save(path=path + "/PPO")

def Train_DQN(log_paths):
    """
    Starts DQN training in Environment 0 for number_agents agents for every path in log_paths
    
    :param log_paths: [(path,wrapper),(path,wrapper),...] Paths and wrapper 
    """
    number_agents = 10
    for log_path_index in range(len(log_paths)):
        env = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name, is_testing=False)
        env.reset()

        wrapped_env = get_wrapped_env(env, log_paths[log_path_index][1])

        for i in range(0, number_agents):
            path = log_paths[log_path_index][0] + str(i)

            os.mkdir(path)
            monitor = Monitor(wrapped_env, path)

            if log_paths[log_path_index][1] == Wrapper.relative:
                model = DQN('MultiInputPolicy', monitor, verbose=1, learning_rate=5e-4, batch_size=64,
                        exploration_initial_eps=1.0,
                        exploration_final_eps=0.01, exploration_fraction=0.8, learning_starts=10000, buffer_size=10000,
                        target_update_interval=10000, gamma=1,
                        max_grad_norm=10)
            else:
                model = DQN('MlpPolicy', monitor, verbose=1, learning_rate=5e-4, batch_size=64,
                        exploration_initial_eps=1.0,
                        exploration_final_eps=0.01, exploration_fraction=0.8, learning_starts=10000, buffer_size=10000,
                        target_update_interval=10000, gamma=1,
                        max_grad_norm=10)

            model.learn(total_timesteps=500_000, eval_log_path=path)
            model.save(path=path + "/DQN")


def Test_PPO(log_paths,environments_to_test_in):
    """
    Run PPO Tests
    """
    return Test_in_env(log_paths, RLAlgo.PPO,environments_to_test_in)


def Test_DQN(log_paths,environments_to_test_in):
    """
    Run DQN Tests
    """
    return Test_in_env(log_paths, RLAlgo.DQN,environments_to_test_in)


def Test_in_env(log_paths, algo_type,environments_to_test_in=[0,1,2]):
    """
    Runs agents in test environments, returns episode rewards, also prints episode end in (Goal/Lava/Time)
    
    :param log_paths: [(path,wrapper),(path,wrapper),...] Paths and wrapper 
    :param algo_type:  (RLAlgo.Type) learning algorithmen used
    :return:  episode rewards
    """
	
	#number of episodes to run for each agent
    number_of_episodes = 100
    number_of_agents = 10

    #Get a path to every Agent in the algorithm-group folder
    for log_path_index in range(len(log_paths)):
        LogDir = []
        for i in range(0, number_of_agents):
            path = log_paths[log_path_index][0] + str(i)
            LogDir.append(path)
        LogDirs.append(LogDir)

    allData = []
	
    #Run test for every environment
    for environment_number in environments_to_test_in:
        TestEnv = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name, is_testing=True,
                                                                           level_choice=environment_number)
        TestEnv.reset()

        enviroment_data = []

        # For each algorithm-group
        for logDir in range(len(LogDirs)):
            wrappedTestEnv = get_wrapped_env(TestEnv, log_paths[logDir][1])
            algorithm_data = []
			
			
            #log the finsih Type
            finish_type = dict()
            finish_type["Goal"] = 0
            finish_type["Lava"] = 0
            finish_type["Time"] = 0
			
			#For each Agent in algorithm-group
            for log in get_top_models(LogDirs[logDir], 1):
                print(log)

                #load agent model
                if algo_type == RLAlgo.PPO:
                    path = log + "/PPO"
                    model = PPO.load(path, env=wrappedTestEnv)
                if algo_type == RLAlgo.DQN:
                    path = log + "/DQN"
                    model = DQN.load(path, env=wrappedTestEnv)

                obs = wrappedTestEnv.reset()
                j = 0
                sum_reward = 0
				
                #run 'number_of_episodes' episodes
                while j < number_of_episodes:
				
                    #determinist has to be True for DQN-'relative' agents because of their multiInputPolicy
                    #Otherwise predict the best action not deterministic
                    action, _states = model.predict(obs, deterministic=False)
                    #Execute the best predicted action
                    obs, rewards, done, info = wrappedTestEnv.step(action)
                
                    sum_reward += rewards
                    if done:
                        #log finish Type
                        if(rewards == -1):
                            finish_type["Time"] = finish_type["Time"]+1
                        if(rewards == 49):
                            finish_type["Goal"] = finish_type["Goal"]+1
                        if(rewards == -51):
                            finish_type["Lava"] = finish_type["Lava"]+1
						
                        #save result and reset
                        algorithm_data.append(sum_reward)
                        sum_reward = 0
                        obs = wrappedTestEnv.reset()
                        j += 1
						
            print('Agents from {} in Env. {} results: {}'.format(log_paths[logDir][0], environment_number,finish_type))
            enviroment_data.append(algorithm_data)
        allData.append(enviroment_data)


    return allData
	
def show_info_maps(log_paths, algo_type,environments_to_test_in=[0,1,2]):
    """
    Runs agents in test environments, shows heat-map and quiver-map, also prints episode end in (Goal/Lava/Time)
    
    :param log_paths: [(path,wrapper),(path,wrapper),...] Paths and wrapper 
    :param algo_type:  (RLAlgo.Type) learning algorithmen used
    """
	
	#number of episodes to run for each agent
    number_of_episodes = 100
    number_of_agents = 10

    for log_path_index in range(len(log_paths)):
        LogDir = []
        for i in range(0, number_of_agents):
            path = log_paths[log_path_index][0] + str(i)
            LogDir.append(path)
        LogDirs.append(LogDir)

    allData = []
    for environment_number in environments_to_test_in:
        TestEnv = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name, is_testing=True,
                                                                           level_choice=environment_number)
        TestEnv.reset()

        enviroment_data = []

        # For each algorithm-group
        for logDir in range(len(LogDirs)):
            print(logDir)
            wrappedTestEnv = get_wrapped_env(TestEnv, log_paths[logDir][1])
            algorithm_data = []
			
            #grid-maps
            gridMap = GridMap()
            gridMap.__int__(9, 7)
            pathData = []
			
            #log the finsih Type
            finish_type = dict()
            finish_type["Goal"] = 0
            finish_type["Lava"] = 0
            finish_type["Time"] = 0
			
            for log in get_top_models(LogDirs[logDir], 1):
                print(log)

                if algo_type == RLAlgo.PPO:
                    path = log + "/PPO"
                    model = PPO.load(path, env=wrappedTestEnv)
                if algo_type == RLAlgo.DQN:
                    path = log + "/DQN"
                    model = DQN.load(path, env=wrappedTestEnv)

                obs = wrappedTestEnv.reset()
                j = 0
                sum_reward = 0
                while j < number_of_episodes:
                    action, _states = model.predict(obs, deterministic=False)
                    obs, rewards, done, info = wrappedTestEnv.step(action)
                
                    gridMap.add_obs(info["state"],info["action"])  #grid
                    sum_reward += rewards
                    if done:
                        gridMap.add_finished_obs(info["finished_state"])  # grid
                        if(rewards == -1):
                            finish_type["Time"] = finish_type["Time"]+1
                        if(rewards == 49):
                            finish_type["Goal"] = finish_type["Goal"]+1
                        if(rewards == -51):
                            finish_type["Lava"] = finish_type["Lava"]+1
						
                        algorithm_data.append(sum_reward)
                        sum_reward = 0
                        obs = wrappedTestEnv.reset()
                        j += 1
						
                pathData.append(gridMap.get_normalized_dict()) #grid	
				
            #show Heatmap and quiver
            plotter = Visulizer() #grid
            plotter.show_heat_map(gridMap,'Environment:' + str(environment_number)) #grid
            normalized_map = plotter.normalize_maps(pathData) #grid
            plotter.show_map(normalized_map,'Environment:' + str(environment_number)) #grid
				
            print('Agents from {} in Env. {} results: {}'.format(log_paths[logDir][0], environment_number,finish_type))

def get_wrapped_env(env, wrapper_type):
    """
    return wrapped Environment
    :param env:  original environment
    :param wrapper_type:  wrapper to use
    :return:  wrapped environment
    """
    if wrapper_type == Wrapper.threexthree:
        return Wrapper_3x3(env)
		
    if wrapper_type == Wrapper.fivexfive:
        return Wrapper_5x5(env)
		
    if wrapper_type == Wrapper.augment_testing:
        return Wrapper_augment_test(env)
		
    if wrapper_type == Wrapper.augment_training:
        return Wrapper_augment_training(env,5)
		
    if wrapper_type == Wrapper.lrou:
        return Wrapper_lrou(env)
		
    if wrapper_type == Wrapper.relative:
        return Wrapper_relative_coordinates(env)
		
    if wrapper_type == Wrapper.reward_metrik:
        return Wrapper_reward(env)
		
    if wrapper_type == Wrapper.basic:
        return Wrapper_standard(env)
		
    else:
        return
