import re           #Regular Expression Module

examplesDirectory = "/Users/yassindinana/Desktop/University/Masters/Semester 2/Multi-Agents Systems/Assessments/Assessment 3/Question 2 " \
                    "- Coding/ASMNTIII-Part2-files"
testsDirectory = "/Users/yassindinana/Desktop/University/Masters/Semester 2/Multi-Agents Systems/Assessments/Assessment 3/Question 2 - " \
                 "Coding/ASMNTIII-Part2-files/tests/graphs"

nameExample_file = "/example.apx"
nameTest_file = "/test1.apx"
combiningDirectories = examplesDirectory + nameExample_file

def read_file(examplesDirectory):
    dummy = open(examplesDirectory, "r")
    Aspartix = dummy.read()
    linesOfContent = Aspartix.split("\n")
    extractedArguments = []
    extractedAttacks = []
    for line in linesOfContent:
        if len(line) == 7:
            extractedArguments.append(re.search(r"(?<=\().*?(?=\))", line).group(0))
        elif len(line) > 7:
            dummy = re.search(r"(?<=\().*?(?=\))", line).group(0)
            dummy = dummy.split(",")
            extractedAttacks.append(dummy)
        else:
                dummy2=0


    extractedArguments  = set(extractedArguments)
    extractedAttacks    = [tuple(x) for x in extractedAttacks]
    extractedAttacks    = set(extractedAttacks)
    output = (extractedArguments, extractedAttacks)
    print(output)
    print(f"There are {len(extractedArguments)} arguments and {len(extractedAttacks)} attacks. ")
    return output


output = read_file(combiningDirectories)


def conflictFree(output):
    arguments = output[0]
    attacks = output[1]
    conflict_Free = []
    notConflict_Free = []
    for argumentX in arguments:
        for argumentY in arguments:
            temp = (argumentX, argumentY)
            temp1 = (argumentY, argumentX)
            if (temp in attacks) or (temp1 in attacks):
                notConflict_Free.append(set(temp))
            elif ((temp[0], temp[0]) in attacks) or ((temp[1], temp[1]) in attacks):
                dummy = 0
            else:
                conflict_Free.append(temp)

    conflict_Free = [list(element) for element in conflict_Free]
    conflictFreeFunc = []
    Remove = []

    for extension in conflict_Free:
        if extension[0] == extension[1]:
            conflictFreeFunc.append(extension[0])
        else:
            conflictFreeFunc.append(extension)
    for extension in conflictFreeFunc:
        for extensionR in conflictFreeFunc:
            if len(extension) > 1 and len(extensionR) > 1:
                if extension[0] == extensionR[1] and extension[1] == extensionR[0]:
                    if extension not in Remove:
                        Remove.append(extensionR)




    for extensionR in conflictFreeFunc:
        if len(extensionR) == 1:
            conflictFreeFunc.remove(extensionR)
    for extensionR in Remove:
        conflictFreeFunc.remove(extensionR)
    return conflictFreeFunc


