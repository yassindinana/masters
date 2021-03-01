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


output= read_file(combiningDirectories)




