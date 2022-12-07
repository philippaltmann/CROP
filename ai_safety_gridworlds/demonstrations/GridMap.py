class GridMap:
    StateDict = dict()
    def __int__(self,x,y):
        for i in range(x):
            for j in range(y):
                self.StateDict[(i,j)] = {"u":0,"d":0,"l":0,"r":0}

    def addObs(self,state,action):
        #print(state,action)
        self.StateDict[state][action] = self.StateDict[state][action]+1

    def addFinishedObs(self,state):

        self.StateDict[state]["u"] = self.StateDict[state]["u"]+1

    def get_Map(self):
        return self.StateDict

    def getNormalizedDict(self):
        nDict = dict()
        for state in self.StateDict.keys():
            nDict[state] = dict()
            sum = self.StateDict[state]["u"] + self.StateDict[state]["d"] +self.StateDict[state]["l"] +self.StateDict[state]["r"]

            if(sum != 0):
                nDict[state]["u"] = self.StateDict[state]["u"]/sum
                nDict[state]["d"] = self.StateDict[state]["d"] / sum
                nDict[state]["l"] = self.StateDict[state]["l"] / sum
                nDict[state]["r"] = self.StateDict[state]["r"] / sum
            else:
                nDict[state]["u"] = 0
                nDict[state]["d"] = 0
                nDict[state]["l"] = 0
                nDict[state]["r"] = 0

        return nDict

