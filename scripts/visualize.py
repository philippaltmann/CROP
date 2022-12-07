import numpy as np
import matplotlib.pyplot as plt
from scripts.GridMap import *

class Visulizer:

    def linearMap(x, a, b, A=0, B=1):
        """

        This function takes scalar ``x`` in range [a,b] and linearly maps it to
        the range [A,B].
        ``x`` is truncated to lie in possible boundaries.
        """
        if a == b:
            res = B
        else:
            res = (x - a) / (1. * (b - a)) * (B - A) + A
        if res < A:
            res = A
        if res > B:
            res = B
        return res

    def show_map(self, representation,title):
        """
        Plot the Quiver-Map
        :param representation: GridMap dictionary
        :param title: title
        """
        representation = representation.get_normalized_dict()
        fig,ax = plt.subplots(figsize=(9,7))

        x_pos = []
        y_pos = []
        x_direct = []
        y_direct = []
        for state in representation.keys():
            x_pos.append(state[0])
            y_pos.append(-state[1])
            x_pos.append(state[0])
            y_pos.append(-state[1])
            x_pos.append(state[0])
            y_pos.append(-state[1])
            x_pos.append(state[0])
            y_pos.append(-state[1])

            x_direct.append(representation[state]["r"])
            x_direct.append(0)
            x_direct.append(-representation[state]["l"])
            x_direct.append(0)
            y_direct.append(0)
            y_direct.append(-representation[state]["d"])
            y_direct.append(0)
            y_direct.append(representation[state]["u"])
            ax.xaxis.set_ticks(np.arange(9))
            ax.yaxis.set_ticks(np.arange(-7))

        ax.quiver(x_pos, y_pos, x_direct, y_direct, scale=20)
        plt.title(title)
        plt.show()

    def show_heat_map(self,representation,title):
        """
        Plot the Heat-Map
        :param representation: GridMap dictionary
        :param title: title
        """
        m = np.zeros((7,9))

        for state in representation.get_map().keys():
            m[state[1],state[0]] = (representation.get_map()[state]["u"]+representation.get_map()[state]["d"]+representation.get_map()[state]["l"]+representation.get_map()[state]["r"])
        plt.imshow(m, cmap='YlGn')
        plt.title(title)
        plt.show()

    def normalize_maps(self,maps):
        return_map = GridMap()
        for map in maps:
            for state in map.keys():
                return_map.state_dict[state]["u"] = map[state]["u"]
                return_map.state_dict[state]["r"] = map[state]["r"]
                return_map.state_dict[state]["d"] = map[state]["d"]
                return_map.state_dict[state]["l"] = map[state]["l"]
        return return_map

