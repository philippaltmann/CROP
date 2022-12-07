import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
import pandas as pd

class Utility:
    def boxplot_four(self,data,labels):
        """
        Boxplot environment results for four different algorithms for three environments
        """
        # best_x = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5]]
        # best_y = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        best_x = [[0, 4], [5, 9], [10, 14]]
        best_y = [[42, 42], [44, 44], [40, 40]]
        ax = plt.axes()
        position = 0.5
        for env_data in data:
            bp = plt.boxplot(env_data, showfliers=True, positions=[position, position + 1,position + 2,position + 3], widths=0.6, showmeans=True)
            self.set_box_colors(bp,4)
            position += 5

        # ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5])
        # ax.set_xticklabels(['A', 'B', 'C','D','E','F','G'])
        ax.set_xticks([ 2, 7, 12])
        ax.set_xticklabels(['A', 'B', 'C'])

        # legend
        hB, = plt.plot([1, 1], color="blue")
        hG, = plt.plot([1, 1], color="green")
        hP, = plt.plot([1, 1], color="#8407ad")
        hBr, = plt.plot([1, 1], color="#3b9c93")
        labels.append("Optimaler Reward")
        hR, = plt.plot(best_x[0], best_y[0], color='red', linestyle=':',label="Optimaler Reward")
        plt.plot(best_x[1], best_y[1], color='red', linestyle=':')
        plt.plot(best_x[2], best_y[2], color='red', linestyle=':')

        plt.legend((hB, hG,hP,hBr,hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)
        hP.set_visible(False)
        hBr.set_visible(False)


        # plt.plot(best_x[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(best_x[4],best_y[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(best_x[5],best_y[5],color='red',linestyle=':')
        # plt.plot(best_x[6],best_y[6],color='red',linestyle=':')

        plt.xlabel("Environment")
        plt.ylabel("Reward")
        plt.show()

    def boxplot_three(self,data,labels):
        """
        Boxplot environment results for three different algorithms for three environments
        """
        best_x = [[0.5,3.5],[4.5,7.5],[8.5,11.5]]
        # best_y = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        plt.rcParams.update({'font.size': 28})
        #best_x = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        best_y = [[42, 42], [44, 44], [40, 40]]
        ax = plt.axes()
        position = 1
        for env_data in data:
            bp = plt.boxplot(env_data, showfliers=True, positions=[position, position + 1,position + 2], widths=0.6, showmeans=True)
            self.set_box_colors(bp,3)
            position += 4

        #ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5])
        ax.set_xticks([2, 6, 10])
        ax.set_xticklabels(['A', 'B', 'C'])

        #ax.set_xticklabels(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])

        # legend
        hB, = plt.plot([1, 1], color="blue")
        hG, = plt.plot([1, 1], color="green")
        hP, = plt.plot([1, 1], color="#8407ad")
        hBr, = plt.plot([1, 1], color="#3b9c93")
        hR, =  plt.plot(best_x[0], best_y[0], color='red', linestyle=':')
        labels.append("Optimaler Reward")
        plt.legend((hB, hG,hP,hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)
        hP.set_visible(False)
        hBr.set_visible(False)


        plt.plot(best_x[1], best_y[1], color='red', linestyle=':')
        plt.plot(best_x[2], best_y[2], color='red', linestyle=':')
        # plt.plot(best_x[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(best_x[4],best_y[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(best_x[5],best_y[5],color='red',linestyle=':')
        # plt.plot(best_x[6],best_y[6],color='red',linestyle=':')

        plt.xlabel("Environment")
        plt.ylabel("Reward")
        plt.show()

    def boxplot_two(self,data,labels):
        """
        Boxplot environment results for two different algorithms for ten environments
        """
        best_x = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5],[21.5,23.5],[24.5,26.5],[27.5,29.5],[30.5,32.5]]
        # best_y = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        plt.rcParams.update({'font.size': 32})
        #best_x = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        best_y = [[42, 42], [44, 44], [40, 40],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42]]
        ax = plt.axes()
        position = 1
        for env_data in data:
            bp = plt.boxplot(env_data, showfliers=True, positions=[position, position + 1], widths=0.6, showmeans=True)
            self.set_box_colors(bp,2)
            position += 3

        ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5])
        # ax.set_xticklabels(['A', 'B', 'C','D','E','F','G'])
        #ax.set_xticks([1.5, 4.5, 7.5])
        ax.set_xticklabels(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])

        # legend
        hB, = plt.plot([1, 1], color="blue")
        hG, = plt.plot([1, 1], color="green")
        hP, = plt.plot([1, 1], color="#8407ad")
        hBr, = plt.plot([1, 1], color="#3b9c93")
        hR, =  plt.plot(best_x[0], best_y[0], color='red', linestyle=':')
        labels.append("Optimaler Reward")
        plt.legend((hB, hG,hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)
        hP.set_visible(False)
        hBr.set_visible(False)


        plt.plot(best_x[1], best_y[1], color='red', linestyle=':')
        plt.plot(best_x[2], best_y[2], color='red', linestyle=':')
        # plt.plot(best_x[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(best_x[4],best_y[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(best_x[5],best_y[5],color='red',linestyle=':')
        # plt.plot(best_x[6],best_y[6],color='red',linestyle=':')

        plt.xlabel("Environment")
        plt.ylabel("Reward")
        plt.show()

    def boxplot_one(self,data,labels):
        """
        Boxplot environment results for one algorithms for ten environments
        """
        best_x = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5],[21.5,23.5],[24.5,26.5],[27.5,29.5],[30.5,32.5]]
        # best_y = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        plt.rcParams.update({'font.size': 32})
        #best_x = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        best_y = [[42, 42], [42, 42], [42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42]]
        ax = plt.axes()
        position = 1

        #for env_data in data:
        bp = plt.boxplot(data, showfliers=True, positions=[1.5, 4.5, 7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5], widths=1.5, showmeans=True)
        self.set_box_colors(bp, 1)
        position += 2


        ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5])
        # ax.set_xticklabels(['A', 'B', 'C','D','E','F','G'])
        #ax.set_xticks([1.5, 4.5, 7.5])
        ax.set_xticklabels(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])

        # legend

        labels.append("Optimaler Reward")
        hR, = plt.plot([0,33], [40,40], color='red', linestyle=':')
        plt.legend([hR], labels)



        #plt.plot(best_x[0], best_y[0], color='red', linestyle=':')
        #plt.plot(best_x[1], best_y[1], color='red', linestyle=':')
        #plt.plot(best_x[2], best_y[2], color='red', linestyle=':')
        # plt.plot(best_x[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(best_x[4],best_y[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(best_x[5],best_y[5],color='red',linestyle=':')
        # plt.plot(best_x[6],best_y[6],color='red',linestyle=':')

        plt.xlabel("P_Augment")
        plt.ylabel("Reward")
        plt.show()

    def set_box_colors(self,bp,count):
        """
        Define boxplot colors
        """
        if(count >=1 ):
            plt.setp(bp['boxes'][0], color='blue')
            plt.setp(bp['caps'][0], color='blue')
            plt.setp(bp['caps'][1], color='blue')
            plt.setp(bp['whiskers'][0], color='blue')
            plt.setp(bp['whiskers'][1], color='blue')
            plt.setp(bp['fliers'][0], markeredgecolor='blue')
            plt.setp(bp['means'][0], markerfacecolor='#ff9d00',markeredgecolor='black')
            #plt.setp(bp['medians'][0], color='blue')

        if (count >= 2):
            plt.setp(bp['boxes'][1], color='green')
            plt.setp(bp['caps'][2], color='green')
            plt.setp(bp['caps'][3], color='green')
            plt.setp(bp['whiskers'][2], color='green')
            plt.setp(bp['whiskers'][3], color='green')
            plt.setp(bp['fliers'][1], markeredgecolor='green')
            plt.setp(bp['means'][1], markerfacecolor='#ff9d00',markeredgecolor='black')
            #plt.setp(bp['medians'][1], color='green')

        if (count >= 3):
            plt.setp(bp['boxes'][2], color='#8407ad')
            plt.setp(bp['caps'][4], color='#8407ad')
            plt.setp(bp['caps'][5], color='#8407ad')
            plt.setp(bp['whiskers'][4], color='#8407ad')
            plt.setp(bp['whiskers'][5], color='#8407ad')
            plt.setp(bp['fliers'][2], markeredgecolor='#8407ad')
            plt.setp(bp['means'][2], markerfacecolor='#ff9d00',markeredgecolor='black')
            #plt.setp(bp['medians'][2], color='#8407ad')

        if (count >= 4):
            plt.setp(bp['boxes'][3], color='#3b9c93')
            plt.setp(bp['caps'][6], color='#3b9c93')
            plt.setp(bp['caps'][7], color='#3b9c93')
            plt.setp(bp['whiskers'][6], color='#3b9c93')
            plt.setp(bp['whiskers'][7], color='#3b9c93')
            plt.setp(bp['fliers'][3], markeredgecolor='#3b9c93')
            plt.setp(bp['means'][3], markerfacecolor='#ff9d00',markeredgecolor='black')
            #plt.setp(bp['medians'][3], color='#452306')

    def violinplot_two(self,datas, labels):
        """
        Violinplot two different algorithms in three environments
        """
        # best_x = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5]]
        # best_y = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        best_x = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        best_y = [[42, 42], [44, 44], [40, 40]]
        ax = plt.axes()
        position = 1

        for env_data in datas:
            vp = plt.violinplot(env_data[0], positions=[position], widths=0.6, showmedians=True, showmeans=True)
            vp['bodies'][0].set_color('blue')
            vp['cbars'].set_color('blue')
            vp['cmins'].set_color('blue')
            vp['cmaxes'].set_color('blue')
            vp['cmedians'].set_color('orange')

            xy = [[l.vertices[:, 0].mean(), l.vertices[0, 1]] for l in vp['cmeans'].get_paths()]
            xy = np.array(xy)
            ax.scatter(xy[:, 0], xy[:, 1], s=50, marker="^", zorder=3,c='#ff9d00',edgecolors='black')
            vp['cmeans'].set_visible(False)

            vp = plt.violinplot(env_data[1], positions=[position + 1], widths=0.6, showmedians=True, showmeans=True)
            vp['bodies'][0].set_color('green')
            vp['cbars'].set_color('green')
            vp['cmins'].set_color('green')
            vp['cmaxes'].set_color('green')
            vp['cmedians'].set_color('orange')
            xy = [[l.vertices[:, 0].mean(), l.vertices[0, 1]] for l in vp['cmeans'].get_paths()]
            xy = np.array(xy)
            ax.scatter(xy[:, 0], xy[:, 1], s=50, marker="^", zorder=3,c='#ff9d00',edgecolors='black')
            vp['cmeans'].set_visible(False)
            position += 3

        # ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5])
        # ax.set_xticklabels(['A', 'B', 'C','D','E','F','G'])
        ax.set_xticks([1.5, 4.5, 7.5])
        ax.set_xticklabels(['A', 'B', 'C'])

        # legend
        hB, = plt.plot([1, 1], color="blue")
        hG, = plt.plot([1, 1], color="green")

        plt.xlabel("Environment")
        plt.ylabel('Reward')

        hR, = plt.plot(best_x[0], best_y[0], color='red', linestyle=':',label="Optimaler Reward")
        labels.append("Optimaler Reward")
        plt.plot(best_x[1], best_y[1], color='red', linestyle=':')
        plt.plot(best_x[2], best_y[2], color='red', linestyle=':')

        plt.legend((hB, hG, hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)


        # plt.plot(best_x[3],best_y[3],color='red',linestyle=':')
        # plt.plot(best_x[4],best_y[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(best_x[5],best_y[5],color='red',linestyle=':')
        # plt.plot(best_x[6],best_y[6],color='red',linestyle=':')



        plt.show()

    def violinplot_one(self,datas, labels):
        """
        Violinplot one algorithms in three environments
        """
        # best_x = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5]]
        # best_y = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        best_x = [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]
        best_y = [[42, 42], [44, 44], [40, 40]]
        ax = plt.axes()
        position = 1
        for env_data in datas:
            vp = plt.violinplot(env_data[0], positions=[position], widths=0.6, showmedians=True, showmeans=True)
            vp['bodies'][0].set_color('blue')
            vp['cbars'].set_color('blue')
            vp['cmins'].set_color('blue')
            vp['cmaxes'].set_color('blue')
            vp['cmedians'].set_color('orange')

            xy = [[l.vertices[:, 0].mean(), l.vertices[0, 1]] for l in vp['cmeans'].get_paths()]
            xy = np.array(xy)
            ax.scatter(xy[:, 0], xy[:, 1], s=50, marker="^", zorder=3,c='#ff9d00',edgecolors='black')
            vp['cmeans'].set_visible(False)

            position += 2

        # ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5])
        # ax.set_xticklabels(['A', 'B', 'C','D','E','F','G'])
        ax.set_xticks([1, 3, 5])
        ax.set_xticklabels(['A', 'B', 'C'])

        # legend
        hB, = plt.plot([1, 1], color="blue")
        hG, = plt.plot([1, 1], color="green")
        plt.legend((hB, hG), labels)
        hB.set_visible(False)
        hG.set_visible(False)

        plt.plot(best_x[0], best_y[0], color='red', linestyle=':',label="Optimaler Reward")
        plt.plot(best_x[1], best_y[1], color='red', linestyle=':')
        plt.plot(best_x[2], best_y[2], color='red', linestyle=':')
        # plt.plot(best_x[3],best_y[3],color='red',linestyle=':')
        # plt.plot(best_x[4],best_y[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(best_x[5],best_y[5],color='red',linestyle=':')
        # plt.plot(best_x[6],best_y[6],color='red',linestyle=':')

        plt.xlabel("Test Enviroment Nr.")
        plt.ylabel("Reward")
        plt.show()
		
    def moving_average(self,values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')
	
    def plot_results_lines(self,algorithm_data, num_timesteps,labels,  x_axis="timesteps"):
        """
        plot the results with multiple lines
        
        :param algorithm_data: the save location of the results to plot
        :param num_timesteps:  numbe of timesteps
		:param labels:  labels
		:param x_axis:  x_axis label
        """
        color_list=["blue","green","#8407ad","#3b9c93"]
        color_it = 0
        plt.rcParams.update({'font.size': 32})
        number_of_agents = 10

        for (log_folder,wrapper) in algorithm_data:
            data_frames = []
            for folder in range(0,number_of_agents):
                data_frame = load_results(log_folder+str(folder))
                if num_timesteps is not None:
                    data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
                data_frames.append(data_frame)
            xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
        
        
            y_lines=[]
        
            for (x,y) in xy_list:
        
                y = self.moving_average(y, window=50)
                # Truncate x
                x = x[len(x) - len(y):]
                i= 0
                for x_value in x:
                    y_lines.append([x_value,y[i]])
                    i += 1
        
                plt.plot(x,y,color=color_list[color_it],alpha=0.1)
        
            df = pd.DataFrame(np.array(y_lines)[:, 0:], columns=['x', 'y'])
            df_mean = df.groupby('x')['y'].mean()
        
        
            #[49:] for moving average
            plt.plot(df_mean.index[49:], self.moving_average(df_mean, window=50), '-r',color=color_list[color_it],label=labels[color_it])
        
            color_it +=1
        
        plt.plot([0,num_timesteps],[42,42],linestyle=':',color='red', label="Optimaler Reward")
        
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.legend()
        plt.show()

		
    def plot_results(self,algorithm_data, num_timesteps,labels, x_axis="timesteps"):
        """
        plot the results with multiple lines
        
        :param algorithm_data: the save location of the results to plot
        :param num_timesteps:  numbe of timesteps
        :param labels:  labels
        :param x_axis:  x_axis label
        """
        plt.rcParams.update({'font.size': 32})
        color_list = ["blue", "green", "#8407ad", "#3b9c93","#000000"]
        Counter = 0
        number_of_agents = 10
		
        for (log_folder,wrapper) in algorithm_data:
            data_frames = []
            for folder in range(0,number_of_agents):
                data_frame = load_results(log_folder+str(folder))
                if num_timesteps[Counter] is not None:
                    data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps[Counter]]
                data_frames.append(data_frame)
            xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    
            y_lines = []
            min_list = []
            max_list = []
    
            for (x_list, y_list) in xy_list:
    
                y_list = self.moving_average(y_list, window=50)
                # Truncate x
                x_list = x_list[len(x_list) - len(y_list):]
                i = 0
                for x_value in x_list:
                    y_lines.append([x_value, y_list[i]])
                    i += 1
    
                # plt.plot(x_list,y_list,color=color_list[color_it],alpha=0.1)
    
            df = pd.DataFrame(np.array(y_lines)[:, 0:], columns=['x', 'y'])
            df_mean = df.groupby('x')['y'].mean()
    
            arranged_x = np.arange(start=2500, stop=num_timesteps[Counter] + 2500, step=2500)
            arrangedY_lists = []
            for (x_list, y_list) in xy_list:
    
                arrangedY_List = []
                for x_value in arranged_x:
                    value = min(x_list, key=lambda x: abs(x - x_value))
                    index = x_list.tolist().index(value)
                    if index - 20 > 0:
                        corrY_value = np.mean(y_list[index - 20:index])
                    else:
                        corrY_value = y_list[index]
                    arrangedY_List.append(corrY_value)
                arrangedY_lists.append(arrangedY_List)
    
            meanList = []
            min_list = []
            max_list = []
    
            for i in range(0, len(arrangedY_lists[0])):
                average = 0
                minV = 100
                max = -100
                for arrangedY_List in arrangedY_lists:
                    average += arrangedY_List[i]
                    if arrangedY_List[i] < minV:
                        minV = arrangedY_List[i]
                    if arrangedY_List[i] > max:
                        max = arrangedY_List[i]
                min_list.append(minV)
                max_list.append(max)
                meanList.append(average / len(arrangedY_lists))
    
            plt.fill_between(arranged_x[2:],
                             self.moving_average(min_list, window=3), self.moving_average(max_list, window=3), color=color_list[Counter],
                             alpha=0.1)
    
            # [49:] for moving average
            plt.plot(arranged_x[2:], self.moving_average(meanList, window=3), '-r',
                     color=color_list[Counter],
                     label=labels[Counter])
    
            Counter += 1
    
    
        plt.plot([0,num_timesteps[0]],[42,42],linestyle=':',color='red', label="Optimaler Reward")
    
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
