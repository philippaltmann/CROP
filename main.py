from scripts.train_test import *
from scripts.Enums import *
from scripts.Utility import *

import os
cwd = os.getcwd()


path_agents_training_PPO = [(cwd +'/Agents/Test_Training/PPO/',Wrapper.relative)]
path_agents_training_DQN = [(cwd +'/Agents/Test_Training/DQN/',Wrapper.fivexfive)]
path_agents_standard = [(cwd +'/Agents/PPO/Ohne_Modifikation/',Wrapper.basic)]
path_agents_early = [(cwd +'/Agents/PPO/Early_Stop_25/',Wrapper.basic)]
path_agents_augmented = [(cwd +'/Agents/PPO/Augment_100k/',Wrapper.augment_testing),(cwd +'/Agents/PPO/Augment_1M/',Wrapper.augment_testing)]




if __name__ == '__main__':
	
    #Train Agents with Data Augmentation
    Train_PPO(path_agents_training_PPO)
    #Train_DQN(path_agents_training_DQN)
	
    utility_plotter = Utility()
	
	#plot training with multiple lines
    #utility_plotter.plot_results_lines(algorithm_data=[path_agents_early[0]],num_timesteps=100_000,labels=["PPO stopp bei 25"]) 
	
	#plot training with filled learning curve (takes long time)
    #utility_plotter.plot_results(path_agents_augmented,num_timesteps=[100_000,1_000_000],labels=["100k","1M"])
	
	
	#run agents in test environments [A,B,C], get episode rewards, also prints (Goal/Lava/Time)
    result = Test_PPO(path_agents_augmented,[0,1,2])
    print(result)
	
    #plot results as boxplot
    utility_plotter.boxplot_two(result,["PPO - 100k","PPO - 1M"]) 
    #plot results as violineplot
    utility_plotter.violinplot_two(result,["PPO - 100k","PPO - 1M"]) 
	
	#show heat-map and quiver-map for early stopped Approach
    show_info_maps(path_agents_early,RLAlgo.PPO,[0,1,2])
    

	

    
