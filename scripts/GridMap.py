class GridMap:
    """
    class to save agent actions and states from a gridworld
    """
    state_dict = dict()
    def __int__(self,x,y):
        for i in range(x):
            for j in range(y):
                self.state_dict[(i,j)] = {"u":0,"d":0,"l":0,"r":0}

    def add_obs(self,state,action):
        """
        save state and action in the dictionary 
        """
        self.state_dict[state][action] = self.state_dict[state][action]+1

    def add_finished_obs(self,state):
        """
        If Environment is finished, save the state with placeholder action "u"
        """
        self.state_dict[state]["u"] = self.state_dict[state]["u"]+1

    def get_map(self):
        return self.state_dict

    def get_normalized_dict(self):
        """
        normalize the state-action dictionary
        """
        n_dict = dict()
        for state in self.state_dict.keys():
            n_dict[state] = dict()
            sum = self.state_dict[state]["u"] + self.state_dict[state]["d"] +self.state_dict[state]["l"] +self.state_dict[state]["r"]

            if(sum != 0):
                n_dict[state]["u"] = self.state_dict[state]["u"]/sum
                n_dict[state]["d"] = self.state_dict[state]["d"] / sum
                n_dict[state]["l"] = self.state_dict[state]["l"] / sum
                n_dict[state]["r"] = self.state_dict[state]["r"] / sum
            else:
                n_dict[state]["u"] = 0
                n_dict[state]["d"] = 0
                n_dict[state]["l"] = 0
                n_dict[state]["r"] = 0

        return n_dict

