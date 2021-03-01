from multi_agent_system import *
import random


def generateprop(taskID):
    cost = randint(1, 10)
    days = randint(15, 20)

    return proposal(taskID, days, cost)


# Printing Functions
def printParticipents(allParts):
    for part in allParts :
        print("Part. with ID = ",part.id , " type: ", part.agentType)
        if (part.agentType == 'weighted'):
            print("     weights :",part.weightCost , " ",part.weightTime )
        print("     Proposals :")
        for prop in part.proposals:
            print("     ",prop.taskId," ",prop.time," ",prop.cost)

# Print Received Proposals
def PrintReceivedReplied(props):
    print("Received Replies : ")
    for prop in props:
        print("     Part: ",prop.partId,"State :", prop.state)
# Print Accepted Proposals
def PrintReceivedProposals(props):
    print("Proposals (Accept) : ")
    for prop in props:
        print("     Part: ",prop.partId,
              " State:", prop.state ,
              " Task: ",prop.proposal.taskId,
              " time: ",prop.proposal.time,
              " cost:",prop.proposal.cost)


# Print Preferences
def PrintPreferences(Prefs):
    print("Prefrences (Accept) : ")
    for pref in Prefs:
        print("     Part: ",pref.partId, " ranks:",pref.prefList)

# Print Resulted Proposals
def PrintResults(initiator):
    print("Results : ")
    print("    ", initiator.partResultsList)
    print("    Part ID :", initiator.allAcceptedProposals[initiator.winner].partId)
    print("     Proposal : ",
          "         Task: ", initiator.allAcceptedProposals[initiator.winner].proposal.taskId,
          "         time: ", initiator.allAcceptedProposals[initiator.winner].proposal.time,
          "         cost:", initiator.allAcceptedProposals[initiator.winner].proposal.cost)

# Participents Generation

def getType(partId):
    if(partId < 3):
        return 'CostOverTime'
    elif(partId < 6):
        return 'TimeOverCost'
    elif(partId < 9):
        return 'random'
    else:
        return 'weighted'


def getCostTimeWeight(agentType):
    if(agentType == 'weighted'):
        return random.uniform(0, 1)
    return 0


def generateTasks(taskSize):
    tasks = []
    for i in range(taskSize):
        tasks.append(task(i))

    return tasks
#########################################################################

task_size=10
tasks = generateTasks(task_size)

mParticipants = []
for i in range(12):
    mProposals = []
    for j in range(task_size):
        if randint(0,1):
            mProposals.append(generateprop(j))

    mParticipants.append(participant(i,
                                     mProposals,
                                     getType(i),
                                     getCostTimeWeight(getType(i)),
                                     getCostTimeWeight(getType(i))))

printParticipents(mParticipants)


mInitiator = initiator(tasks)

for i in range(task_size):

    mInitiator.callForProposals(mParticipants,i)
    PrintReceivedReplied(mInitiator.receiveProposals)

    mInitiator.getAllAcceptedProposals()
    PrintReceivedProposals(mInitiator.allAcceptedProposals)

    mInitiator.acceptProposals(mParticipants,mInitiator.allAcceptedProposals)
    PrintPreferences(mInitiator.prefLists)

    mInitiator.votingProcess()
    PrintResults(mInitiator)

    mInitiator.informResults(mParticipants,mInitiator.partResultsList)
    mInitiator.resetLists()

