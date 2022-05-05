import time

def getKey(i):
    return str(i)

def get_Degree(T, key): #Get Number of Children of a certain Node.
    i = 0 
    keys = [key]
    while key+'.'+getKey(i) in T: 
        i=i+1
        keys.append(key+'.'+getKey(i-1))   
    return i, keys
      

#Return the depth; number of children and grandchildren ...
def get_depth(nodes): 
	minDepth = min(nodes, key=len).count('.')
	maxDepth = max(nodes, key=len).count('.')
	return (maxDepth - minDepth)

#Get a list containing the keys of the nodes included in the subtree
def get_Subtree(T, key):
    ret = [key] 
      #loop through the keys
    for node in T.keys(): 
        if node.startswith(key+'.'):
              ret.append(node)
    return ret

#Returns a list of values of the subtree
def get_Values(T, subtree):
    ret = []
    for k in subtree:
        ret.append(T[k]['value'])
    return ret

def Contains(T1, T2, rootKey):
    subtree = get_Subtree(T1, rootKey)
    relativeDepth = get_depth(subtree)
      
      #Check if leaf node
    if relativeDepth==0:
        return 1

      #Check if value of rootKey is not between the values of the second tree
    elif not T1[rootKey] in T2.values():
        return len(subtree)

      #for all the keys in T2; 
    for k in T2.keys():
            #check if the value of the key k is == to rootKey
        if T2[k] == T1[rootKey]: 
            subtree2 = get_Subtree(T2, k)
            relativeDepth2 = get_depth(subtree2) 

              #If subtree1 has more nodes OR a bigger depth than subtree2, then it is NOT in subtree2
            if len(subtree)<=len(subtree2) and relativeDepth <= relativeDepth2:  
                val1 = get_Values(T1, subtree)
                val2 = get_Values(T2, subtree2)

                    #if one of the values of subtree is not in subtree2 (regardless of ordering now)
                if set(val1).issubset(set(val2)): 

                      #Check if all the nodes of subtree are in order in subtree2 (regardless of 
                      # useless nodes); and checks the children of each child/node
                    i = 1  
                    for j in range(1, len(val2)):
                        if val1[i]==val2[j]:
                            s1 = get_Subtree(T1, subtree[i])
                            s2 = get_Subtree(T2, subtree2[j])
                            if get_depth(s1) == get_depth(s2):
                                i = i+1
                                if i==len(val1):
                                    return 1   
    return len(subtree)

def TED(Tree1, Tree2, x, y, cost_del, cost_ins, Matrices): #Recursive TED Part... to be merged soon?
    Matrices.update({x+','+y: []})

    #Degree of the branch     
    degreeA, keyA = get_Degree(Tree1, x)
    degreeB, keyB = get_Degree(Tree2, y)

    #Instantiate the 2D Array 
    Dist = [ [0] * (degreeB + 1) for e in range(degreeA + 1) ]

    #Check if Roots are the Same

    # print(Tree1[x].values())
    # print(Tree2[y].values())

    if Tree1[x]['value'] == Tree2[y]['value']:
        Dist[0][0] = 0
    else:
        Dist[0][0] = 1 

    #Compute first row and column  
    for i in range(1, degreeA+1): #First Column --> Delete
        Dist[i][0] = Dist[i-1][0] + cost_del[keyA[i]] 
    for j in range(1, degreeB+1): #First Row --> Insert
        Dist[0][j] = Dist[0][j-1] + cost_ins[keyB[j]]

    #Compute rest of the table 
    for i in range(1, degreeA+1):
        for j in range(1, degreeB+1):
            costs = [] 
            #delete
            costs.append(Dist[i-1][j] + cost_del[keyA[i]])
            #insert
            costs.append(Dist[i][j-1] + cost_ins[keyB[j]])
            #update
            upd, dummies = TED(Tree1, Tree2, keyA[i], keyB[j], cost_del, cost_ins, Matrices)
            costs.append(Dist[i-1][j-1] + upd)
            
            Dist[i][j] = min(costs)
            
    Matrices[x+','+y] = Dist
    return Dist[degreeA][degreeB], Matrices


def Begin(Tree1, Tree2): 

    cost_dels = {}
    cost_inss = {}

    for i in Tree1:
        cost_dels.update({i: Contains(Tree1, Tree2, i)})
    for i in Tree2:
        cost_inss.update({i : Contains(Tree2, Tree1, i)}) 

    distance, Trees = TED(Tree1, Tree2, '0', '0', cost_dels, cost_inss, {})
    
    similarity = 1/(1+distance) 
    return similarity  