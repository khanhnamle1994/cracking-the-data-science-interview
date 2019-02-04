class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue # name of the node
        self.count = numOccur # count of the node
        self.nodeLink = None    # used to link similar items
        self.parent = parentNode      # parent of the node in the tree
        self.children = {}  # empty dictionary for the children of the node

    # increment the count variable by a given amount
    def inc(self, numOccur):
        self.count += numOccur

    # display the tree in text
    def disp(self, ind=1):
        print '  '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)

# Function that takes the dataset and the minimum support as arguments and builds the FP-tree
def createTree(dataSet, minSup=1):
    headerTable = {}
    # go over dataSet twice
    for trans in dataSet: # first pass counts frequency of occurance
        for item in trans:
            # store such frequencies in headerTable
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    for k in headerTable.keys():  # second pass removes items not meeting minSup
        if headerTable[k] < minSup:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None, None  # if no items meet min support, do no further processing

    for k in headerTable:
        # reformat headerTable to hold a count and pointer to the 1st item of each type
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None) # create the base node containing a null set

    for tranSet, count in dataSet.items():  # go through dataset 2nd time and only use frequent items
        localD = {}
        # sort transactions by global frequency
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count) # call updateTree() method

    return retTree, headerTable # return tree and header table

# Function to populate tree with ordered freq itemset
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children: # check if the first item in the transaction exists as a child node
        inTree.children[items[0]].inc(count) # if so, incrament count
    else:   # if the item doesn't exist
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # create a new treeNode
        if headerTable[items[0]][1] == None: # update header table by adding new treeNode as a child
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) # call updateHeader() method
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

# Function to make sure the node links point to every instance of this item in the tree
def updateHeader(nodeToTest, targetNode):
    # start with the first nodeLink in the header table and then follow the nodeLinks until the end
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

# Function to ascend from leaf node to root
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name) # collect the names of items encountered
        ascendTree(leafNode.parent, prefixPath)

# Function to iterate through the linked list until it hits the end
def findPrefixPath(basePat, treeNode): # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

# Function to recursively find frequent itemsets
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # sort the items in the headerTable by frequency of occurrence
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    for basePat in bigL:  # start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # add each frequent item to the list of frequent itemsets
        freqItemList.append(newFreqSet)
        # recursively call findPrefixPath() to create a conditional base
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # construct cond FP-tree from conditional pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None: # if the tree has any items in it
            # recursively call mineTree()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
