import matplotlib.pyplot as plt
import numpy as np

class Utility:

    def boxplotFour(self,data,labels):
        # bestx = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5]]
        # besty = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        bestx = [[0, 4], [5, 9], [10, 14]]
        besty = [[42, 42], [44, 44], [40, 40]]
        ax = plt.axes()
        position = 0.5
        for env_data in data:
            bp = plt.boxplot(env_data, showfliers=True, positions=[position, position + 1,position + 2,position + 3], widths=0.6, showmeans=True)
            self.setBoxColors(bp,4)
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
        hR, = plt.plot(bestx[0], besty[0], color='red', linestyle=':',label="Optimaler Reward")
        plt.plot(bestx[1], besty[1], color='red', linestyle=':')
        plt.plot(bestx[2], besty[2], color='red', linestyle=':')

        plt.legend((hB, hG,hP,hBr,hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)
        hP.set_visible(False)
        hBr.set_visible(False)


        # plt.plot(bestx[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(bestx[4],besty[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(bestx[5],besty[5],color='red',linestyle=':')
        # plt.plot(bestx[6],besty[6],color='red',linestyle=':')

        plt.xlabel("Environment")
        plt.ylabel("Reward")
        plt.show()

    def boxplotThree(self,data,labels):
        bestx = [[0.5,3.5],[4.5,7.5],[8.5,11.5]]
        # besty = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        plt.rcParams.update({'font.size': 28})
        #bestx = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        besty = [[42, 42], [44, 44], [40, 40]]
        ax = plt.axes()
        position = 1
        for env_data in data:
            bp = plt.boxplot(env_data, showfliers=True, positions=[position, position + 1,position + 2], widths=0.6, showmeans=True)
            self.setBoxColors(bp,3)
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
        hR, =  plt.plot(bestx[0], besty[0], color='red', linestyle=':')
        labels.append("Optimaler Reward")
        plt.legend((hB, hG,hP,hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)
        hP.set_visible(False)
        hBr.set_visible(False)


        plt.plot(bestx[1], besty[1], color='red', linestyle=':')
        plt.plot(bestx[2], besty[2], color='red', linestyle=':')
        # plt.plot(bestx[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(bestx[4],besty[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(bestx[5],besty[5],color='red',linestyle=':')
        # plt.plot(bestx[6],besty[6],color='red',linestyle=':')

        plt.xlabel("Environment")
        plt.ylabel("Reward")
        plt.show()

    def boxplotTwo(self,data,labels):
        bestx = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5],[21.5,23.5],[24.5,26.5],[27.5,29.5],[30.5,32.5]]
        # besty = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        plt.rcParams.update({'font.size': 32})
        #bestx = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        besty = [[42, 42], [44, 44], [40, 40],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42]]
        ax = plt.axes()
        position = 1
        for env_data in data:
            bp = plt.boxplot(env_data, showfliers=True, positions=[position, position + 1], widths=0.6, showmeans=True)
            self.setBoxColors(bp,2)
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
        hR, =  plt.plot(bestx[0], besty[0], color='red', linestyle=':')
        labels.append("Optimaler Reward")
        plt.legend((hB, hG,hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)
        hP.set_visible(False)
        hBr.set_visible(False)


        plt.plot(bestx[1], besty[1], color='red', linestyle=':')
        plt.plot(bestx[2], besty[2], color='red', linestyle=':')
        # plt.plot(bestx[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(bestx[4],besty[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(bestx[5],besty[5],color='red',linestyle=':')
        # plt.plot(bestx[6],besty[6],color='red',linestyle=':')

        plt.xlabel("Environment")
        plt.ylabel("Reward")
        plt.show()

    def boxplotOne(self,data,labels):
        bestx = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5],[21.5,23.5],[24.5,26.5],[27.5,29.5],[30.5,32.5]]
        # besty = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        plt.rcParams.update({'font.size': 32})
        #bestx = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        besty = [[42, 42], [42, 42], [42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42],[42, 42]]
        ax = plt.axes()
        position = 1

        #for env_data in data:
        bp = plt.boxplot(data, showfliers=True, positions=[1.5, 4.5, 7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5], widths=1.5, showmeans=True)
        self.setBoxColors(bp, 1)
        position += 2


        ax.set_xticks([1.5, 4.5, 7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5])
        # ax.set_xticklabels(['A', 'B', 'C','D','E','F','G'])
        #ax.set_xticks([1.5, 4.5, 7.5])
        ax.set_xticklabels(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])

        # legend

        labels.append("Optimaler Reward")
        hR, = plt.plot([0,33], [40,40], color='red', linestyle=':')
        plt.legend([hR], labels)



        #plt.plot(bestx[0], besty[0], color='red', linestyle=':')
        #plt.plot(bestx[1], besty[1], color='red', linestyle=':')
        #plt.plot(bestx[2], besty[2], color='red', linestyle=':')
        # plt.plot(bestx[3],bes grty[3],color='red',linestyle=':')
        # plt.plot(bestx[4],besty[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(bestx[5],besty[5],color='red',linestyle=':')
        # plt.plot(bestx[6],besty[6],color='red',linestyle=':')

        plt.xlabel("P_Augment")
        plt.ylabel("Reward")
        plt.show()

    def setBoxColors(self,bp,count):
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

    def violinPlotTwo(self,datas, labels):

        # bestx = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5]]
        # besty = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        bestx = [[0.5, 2.5], [3.5, 5.5], [6.5, 8.5]]
        besty = [[42, 42], [44, 44], [40, 40]]
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

        hR, = plt.plot(bestx[0], besty[0], color='red', linestyle=':',label="Optimaler Reward")
        labels.append("Optimaler Reward")
        plt.plot(bestx[1], besty[1], color='red', linestyle=':')
        plt.plot(bestx[2], besty[2], color='red', linestyle=':')

        plt.legend((hB, hG, hR), labels)
        hB.set_visible(False)
        hG.set_visible(False)


        # plt.plot(bestx[3],besty[3],color='red',linestyle=':')
        # plt.plot(bestx[4],besty[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(bestx[5],besty[5],color='red',linestyle=':')
        # plt.plot(bestx[6],besty[6],color='red',linestyle=':')



        plt.show()

    def violinPlotOne(self,datas, labels):

        # bestx = [[0.5,2.5],[3.5,5.5],[6.5,8.5],[9.5,11.5],[12.5,14.5],[15.5,17.5],[18.5,20.5]]
        # besty = [[42,42],[44,44],[40,40],[44,44],[36,36],[40,40],[42,42]]
        bestx = [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]
        besty = [[42, 42], [44, 44], [40, 40]]
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

        plt.plot(bestx[0], besty[0], color='red', linestyle=':',label="Optimaler Reward")
        plt.plot(bestx[1], besty[1], color='red', linestyle=':')
        plt.plot(bestx[2], besty[2], color='red', linestyle=':')
        # plt.plot(bestx[3],besty[3],color='red',linestyle=':')
        # plt.plot(bestx[4],besty[4],color='red',linestyle=':',label='best possible reward')
        # plt.plot(bestx[5],besty[5],color='red',linestyle=':')
        # plt.plot(bestx[6],besty[6],color='red',linestyle=':')

        plt.xlabel("Test Enviroment Nr.")
        plt.ylabel("Reward")
        plt.show()
