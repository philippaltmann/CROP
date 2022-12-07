import numpy as np
import matplotlib.pyplot as plt

class Visulizer:

    def linearMap(x, a, b, A=0, B=1):
        """
        .. warning::
            ``x`` *MUST* be a scalar for truth values to make sense.
        This function takes scalar ``x`` in range [a,b] and linearly maps it to
        the range [A,B].
        Note that ``x`` is truncated to lie in possible boundaries.
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

    def show_map(self, representation):
        representation = representation.getNormalizedDict()
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
        plt.show()

    def show_heat_map(self,representation):
        #print(representation.get_Map())
        m = np.zeros((7,9))

        for state in representation.get_Map().keys():
            m[state[1],state[0]] = (representation.get_Map()[state]["u"]+representation.get_Map()[state]["d"]+representation.get_Map()[state]["l"]+representation.get_Map()[state]["r"])
        #print(m)
        plt.imshow(m, cmap='YlGn')
        plt.show()

