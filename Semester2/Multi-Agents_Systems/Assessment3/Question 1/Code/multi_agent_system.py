from random import randint
import scipy.stats as ss
import numpy as np
import pandas as pd

class task:
    def __init__(self, id):
        self.id = id
        self.done = 0

    def finishTask(self):
        self.done = True

    def resetTaskDone(self):
        self.done = False

class proposalBundle:
    def __init__(self, partId, state, proposal):
        self.partId = partId
        self.state = state
        self.proposal = proposal

class prefBundle:
    def __init__(self, partId, prefList):
        self.partId = partId
        self.prefList = prefList

class proposal:
    taskId = 0
    time = 0
    cost = 0

    def __init__(self, taskId,time,cost):
        self.taskId = taskId
        self.time = time
        self.cost = cost

class requestReplyMsg:
    def __init__(self, type,data,sender):
        self.type = type
        self.data = data
        self.sender = sender


class result:
    def __init__(self, partID,winLose):
        self.partID = partID
        self.winLose = winLose



class participant:

    def __init__(self, id, proposals,agentType,weightCost,weightTime):
        self.id = id
        self.proposals = proposals
        self.agentType = agentType
        self.weightCost = weightCost
        self.weightTime = weightTime

    def computePref(self,time,cost):
        if(self.agentType == 'CostOverTime'):
            return cost/time
        if (self.agentType == 'TimeOverCost'):
            return time/cost
        if (self.agentType == 'random'):
            return randint(0,100)
        if (self.agentType == 'weighted'):
            return time*self.weightTime + cost*self.weightCost


    # reply helper Functions
    def getProposal(self,taskId):

        proposals_list = []
        for i in self.proposals:
            if taskId == i.taskId:
                proposals_list.append(i)

        if(len(proposals_list) > 0 ):
            return "accept" , proposals_list[randint(0,len(proposals_list) -1 )]
        else:
            return "Reject", None

    def getPreferences(self,acceptedProposals):
        prefrencesValue = []
        for prop in acceptedProposals:
            prefVal = self.computePref(prop.proposal.time,prop.proposal.cost)
            prefrencesValue.append(prefVal)

        ranks = ss.rankdata(prefrencesValue)
        return ranks



    # receive Function
    def receiveComm(self,request):
        if(request.type == "CallForProposal"):
            reply= requestReplyMsg("proposalReply",None,self)
            mProposalBundle = proposalBundle(self.id, None, None)

            mProposalBundle.state , mProposalBundle.proposal = self.getProposal(request.data)
            reply.data = mProposalBundle
            request.sender.receiveComm(reply)



        if (request.type == "AcceptProposals"):
            reply = requestReplyMsg("prefrences", None, self)
            mPrefBundle = prefBundle(self.id, None)

            mPrefBundle.prefList = self.getPreferences(request.data)
            reply.data = mPrefBundle
            request.sender.receiveComm(reply)


        if (request.type == "informResults"):
            f = 0


    ###############################################################################


class initiator:

    def __init__(self,tasks):
        self.id = 1
        self.tasks = tasks
        self.receiveProposals = []
        self.prefLists = []
        self.resultsList = []
        self.partResultsList = []
        self.allAcceptedProposals = []
        self.winner = -1

    def getTaskToDo(self):
        for task in self.tasks:
            if (task.done == 0):
                return task

    def getAllAcceptedProposals(self):
        self.allAcceptedProposals = []
        for prop in self.receiveProposals:
            if(prop.state == "accept"):
                self.allAcceptedProposals.append(prop)


    def votingProcess(self):
        sumPref = self.prefLists[0].prefList
        for i in range(1,len(self.prefLists)):
            sumPref = sumPref + self.prefLists[i].prefList

        rankPref = ss.rankdata(sumPref)
        self.resultsList = []
        self.partResultsList = []
        i = 0
        for rank in rankPref:
            if(rank > 1):
                self.resultsList.append("lose")
            else:
                self.resultsList.append("winner")
                self.winner = i
            i = i + 1

        self.partResultsList = []
        for i in range(len(self.prefLists)):
            if(self.allAcceptedProposals[self.winner].partId == i ):
                self.partResultsList.append("winner")
            else:
                self.partResultsList.append("loser")


    def resetLists(self):
        self.receiveProposals = []
        self.prefLists = []
        self.resultsList = []
        self.allAcceptedProposals = []
        self.winner = -1

# Send Functions
    def callForProposals(self,participants,taskId):
        request = requestReplyMsg("CallForProposal", taskId, self)
        for p in participants:
            p.receiveComm(request)

    def acceptProposals(self,participants,allAcceptedProposals):
        request = requestReplyMsg("AcceptProposals", allAcceptedProposals, self)
        for p in participants:
            p.receiveComm(request)



    def informResults(self,participants,allResults):
        request = requestReplyMsg("informResults", allResults, self)
        for p in participants:
            p.receiveComm(request)

    # receive Functions
    def receiveComm(self,request):
        if (request.type == "proposalReply"):
            self.receiveProposals.append(request.data)

        if (request.type == "prefrences"):
            self.prefLists.append(request.data)


    #Run-Off Voting
    def table(n):
        table = np.arange(1, n + 1)
        np.random.shuffle(table)
        return table

    n = 12

    votes = pd.DataFrame([table(n), table(n), table(n),
                          table(n), table(n), table(n),
                          table(n), table(n), table(n),
                          table(n), table(n), table(n),
                          table(n), table(n), table(n),
                          table(n), table(n), table(n)], columns=["participant0", "participant1", "participant2",
                                                                  "participant3", "participant4",
                                                                  "participant5", "participant6", "participant7",
                                                                  "participant8", "participant9",
                                                                  "participant10", "participant11"])

    dataframe_table = votes.copy()
    length = votes.columns.shape[0]
    iterations = np.arange(1, length - 1)

    for i in iterations:
        min_votes = dataframe_table[dataframe_table == dataframe_table.min(axis=1).min()].count().idxmax()
        dataframe_table = dataframe_table.drop([min_votes], axis=1)
        dataframe_table = dataframe_table.rank(axis=1, method='first') + i

    winner = dataframe_table[votes == length].count().idxmax()
    print("The Winner is: ", winner)

