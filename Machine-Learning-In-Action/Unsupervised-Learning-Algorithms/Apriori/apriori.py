from numpy import *
from importlib import reload

# Create a simple dataset for testing
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# Create a candidate itemset of size 1
def createC1(dataSet):
    # create an empty list C1 to store all unique values
    C1 = []
    # iterate over all the transactions in our dataset
    for transaction in dataSet:
        # iterate over all items for each transaction
        for item in transaction:
            # if an item is not in C1, add it to C1
            if not [item] in C1:
                # add a list of single-item lists, which will be used in set operations later
                C1.append([item])

    C1.sort() # sort the list C1
    return map(frozenset, C1)# use frozen set so we can use it as a key in a dict

# Scan the dataset for any itemsets that meet our minimum support requirements (L1).
# L1 then gets combined to become C2 and C2 will get filtered to become C2.
# There are 3 arguments: a list of candidate sets Ck, a dataset D, and the minimum support minSupport
def scanD(D, Ck, minSupport):
    # Create an empty dictonary
    ssCnt = {}
    # Go over all transactions in the dataset and all the candidate sets Ck
    for tid in D:
        for can in Ck:
            # if the sets of C1 are part of the dataset
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1 # increment the count in dictonary
    numItems = float(len(D))
    # create an empty list that will hold the sets that do meet the minimum support
    retList = []
    supportData = {}
    # Go over every element in the dictonary
    for key in ssCnt:
        # meausure the support
        support = ssCnt[key]/numItems
        # if the support meets the minimum support requirements
        if support >= minSupport:
            # add it to the beginning of retList
            retList.insert(0,key)
        supportData[key] = support
    # return retList and supportData (which holds the support values for the frequent itemsets)
    return retList, supportData

# Creates candidate itemsets Ck. It takes a list of frequent itemsets Lk and the size of the itemsets k
def aprioriGen(Lk, k):
    # create an empty list
    retList = []
    # the number of elements in Lk
    lenLk = len(Lk)
    # 2 for loops to compare each item in Lk with all of the other items in Lk
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # take 2 sets in our list and compare them
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2: #if first k - 2 elements are equal
                retList.append(Lk[i] | Lk[j]) # set union to combine the 2 sets
    return retList

# Generate a list of candidate itemsets. It takes in a dataset and a support number
def apriori(dataSet, minSupport = 0.5):
    # create C1
    C1 = createC1(dataSet)
    # take the dataset and turn that into D (a list of sets)
    D = map(set, dataSet)
    # use scanD to create L1
    L1, supportData = scanD(D, C1, minSupport)
    # place L1 inside a list L (contains L1, L2, L3...)
    L = [L1]
    k = 2
    # create larger lists of larger itemsets until the next-largest itemset is empty
    while (len(L[k-2]) > 0):
        # use aprioriGen to create Ck
        Ck = aprioriGen(L[k-2], k)
        # use scanD to create Lk from Ck
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        # append Lk to L
        L.append(Lk)
        # increment k
        k += 1
    # return L when Lk is empty
    return L, supportData

# Function to generate a list of rules with confidence values that we can sort through later
# It takes in 3 inputs: a list of frequent itemsets, a dictionary of support data for those itemsets, and a minimum confidence threshold
def generateRules(L, supportData, minConf=0.7):  # supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):# only get the sets with two or more items
        # loop over every frequent itemset in L
        for freqSet in L[i]:
            # create a list of single-item sets: H1 for each frequent itemset
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                # if the fre itemset has more than 2 items it, merge them with rulesFromConseq()
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # if the itemset only has 2 items in it, calculate the confidence with calcConf()
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# Function to evaluate the candidate rules
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] # create an empty list to hold the rules that meet the minimum requirements
    # iterate over all the itemsets in H
    for conseq in H:
        # calculate the confidence with support values in supportData
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        # if a rule does not meet the minimum confidence
        if conf >= minConf:
            # print the rule to the screeen
            print freqSet-conseq,'-->',conseq,'conf:',conf
            # fill in the list brl
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# Function to generate more association rules from initial itemset
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # meausure m (the size of the itemsets in H)
    m = len(H[0])
    # see if the frequent itemset is large enough to have subsets of size m removed, if so, proceed
    if (len(freqSet) > (m + 1)):
        # use aprioriGen() to generate combinations of the items in H without repeating
        Hmp1 = aprioriGen(H, m+1)
        # test the confidence of all possible rules in Hmp1
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # if more than one rule remains, recursively call rulesFromConseq() with Hmp1 to see if such rules can be combined further
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
