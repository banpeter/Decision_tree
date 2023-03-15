#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[202]:


#Ház ár legenerálása
#Egy mérethez egy árat számítunk


#size: square meter math.sqrt(i+j/10)*7+ 0.2*i)*1000000
#distance from the center 1000/distance +10   dinstance 100-5000 meter
#interest rate 1-10% = -(x**2)/7 +30
#government support yes = 0-10 million
#local average salaray(HUF) 2-8


X_training = []
y_training = []
#header = ["size"]
header = ["size","distance","support","salary"]

for i in range(10000):
    
    price = 0
    
    size = random.randint(0,99)+random.randint(0,10)/10
    distance = random.randint(30,100)
    #interest_rate = random.randint(1,10)
    support = random.randint(0,10)
    salary = random.randint(3,8)

    price += ((math.sqrt(size)*10+0.3*(size/10)))
    price += (1000/distance + 5)
    #price += (-(interest_rate**2)/4 +30)
    price += support
    price += salary**3
    

    X_training.append([size,distance,support,salary])
    y_training.append(price)
    
X_train = []
y_train = []

X_test = []
y_test = []

#Felosztjuk az adatokat 
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.2) # 80% training and 20% testing data
print(X_train[0])
print(y_train[0])


# In[418]:


batch_x = []
batch_y = []

a = len(X_train)
step = 1000

for i in range(0, a, step):


    for j in range(i, i + step):
        batch_x.append(X_train[i:i + step])
        batch_y.append(y_train[i:i + step])

        


# In[34]:


#Sorba rendezzük az adathalmazt a méret alapján növekvően
def sort_f(X,y,column):
    
    for i in range(len(X)):
        temp = i
        for j in range(i,len(X)):
            
            #print(X[j])
            #print(X[temp][column])
            if(X[j][column]<X[temp][column]):
                temp = j
        if(temp != i):
            tmp = X[i]
            X[i] = X[temp]
            X[temp] = tmp

            tmp = y[i]
            y[i] = y[temp]
            y[temp] = tmp
    return X,y


# In[35]:


# visualize data set
plt.rcParams["figure.figsize"] = (15,7)
plt.scatter([ X_training[i][0] for i in range(len(X_training)) ],y_training,s=1)

plt.xlabel("Square meter")
plt.ylabel("Price")
plt.title("All Data")

plt.show()


# In[36]:


plt.rcParams["figure.figsize"] = (15,7)
plt.scatter([ X_training[i][1] for i in range(len(X_training)) ],y_training,s=1)

plt.xlabel("Square meter")
plt.ylabel("Price")
plt.title("All Data")

plt.show()


# In[37]:


# visualize data set
plt.rcParams["figure.figsize"] = (15,7)
plt.scatter([ X_test[i][0] for i in range(len(X_test)) ],y_test,s=1)

plt.xlabel("Square meter")
plt.ylabel("Price")
plt.title("Test Data")

plt.show()


# In[167]:


class Tree:
    def __init__(self,X,feature_names,labels):
        
        self.X = X
        self.num_of_nodes = 0
        self.currentsplit = 0
        self.split_result = 0
        self.feature_names = feature_names #coloum names
        self.labels = labels#y
        self.catagories = set(labels)
        self.nodes = []
        self.split = 0
        self.leaf = 0
        self.steps = [0,0,0]
        


# In[168]:


class Node:
    def __init__(self,labels,X,feature_ids):
        
        self.split_result = 0
        self.split = 0 #which column / which feature id
        self.feature_ids = feature_ids
        self.labels = labels
        self.X = X
        self.nodes = []
        self.regr = 0
        self.depth = 0
        self.top = 0
        self.bottom = 0
        self.steps = [0,0,0]
        self.prev_node = 0
        
        self.leaf = 0 #true or false


# In[169]:


def get_features(X,labels,column,bottom,top):#return splited data
    

    
    sub_label = [labels[x] for x in range(len(labels)) if X[x][column]<top and X[x][column]>=bottom]
    sub_X = [X[x] for x in range(len(X)) if X[x][column]<top and X[x][column]>=bottom]


        
    return [sub_X,sub_label]


# In[170]:


def regression(regr_type,X,y,column,dgr):#whaT TYPE OF REGRESSION,data
    
    
    #reshape data
    
    X = [ [X[i][column]] for i in range(len(X)) ]

    
    regr = LinearRegression()
    regr.fit(X,y)
    
    
    return regr


# In[171]:


def calculate_error(regr,X,y,column,regr_type,dgr):
    
    
    #reshape data
    X = [ [X[i][column]] for i in range(len(X)) ]
    if(regr_type == 1):
        X = PolynomialFeatures(degree=dgr).fit_transform(X)
    predict = regr.predict(X)

    difference = []
    
    for i in range(len(predict)):
        difference.append( (predict[i]-y[i])**2 )
    error = sum(difference)/len(difference)

  
    return error


# In[172]:


def calculate_distance(i,j,column):
    

 
    n1 = [ i[0][k][column] for k in range(len(i[0])) ]
    n2 = [ j[0][k][column] for k in range(len(j[0])) ]
    
    
    
    Ni = len(n1)
    Nj = len(n2)
    
    s1i = sum(n1)/Ni
    s1j = sum(n2)/Nj
    
    s2i = sum([x**2 for x in n1])/Ni
    s2j = sum([x**2 for x in n2])/Nj
    
    s3i = sum([x*y for x,y in zip(n1,i[1])])/Ni
    s3j = sum([x*y for x,y in zip(n2,j[1])])/Nj
    
    s4i = sum(i[1])/Ni
    s4j = sum(j[1])/Nj
    
    D = (s1i-s1j)**2 + (s2i-s2j)**2 + (s3i-s3j)**2 +(s4i-s4j)**2
    
    return D


# In[173]:


def find_min_distance(sliced_data,column):
    
    dist = 0
    min_dist = -1
    index1 = -1
    index2 = -1
    #print(sliced_data[1])
    
    for i in range(len(sliced_data)-1):

            dist = calculate_distance(sliced_data[i],sliced_data[i+1],column)
            
            if(min_dist == -1):
                min_dist = dist
                index1 = i
                index2 = i+1
            if(dist<min_dist):
                min_dist = dist
                index1 = i
                index2 = i+1
    
    return index1,index2
    


# In[174]:


def merge(sliced_data,regressions,column):
    
   
    index1,index2 = find_min_distance(sliced_data,column)
    


    sliced_data[index1][0] = sliced_data[index1][0]+sliced_data[index2][0]
    sliced_data[index1][1] = sliced_data[index1][1]+sliced_data[index2][1]

    regressions[index1] = regression(0,sliced_data[index1][0],sliced_data[index1][1],column,0)
    #print("###############################################################################")
    sliced_data.pop(index2)
    regressions.pop(index2)
    
    return sliced_data,regressions


# In[346]:


def find_best_split(X,labels,feature_ids,steps):
    
    step = -1
    #print(steps)
    features = 0 
    svalue = 0
    
    min_error = -1
    min_sliced_data = []
    min_regressions = []
    min_steps = -1
    min_column = -1

    for column in feature_ids:
           
            
            X,labels = sort_f(X,labels,column)
            features = [ X[j][column] for j in range(len(X)) ] 
            
            sliced_data = []
            regressions = []
           
            
            dist = abs(features[0]-features[-1])
            
            
            step = int(dist/10)
            
            if(step < 2):
                step = 1
                
                if(steps[column] > 2):
                    continue
           
            for k in range(int(features[0]),int(features[-1])+1,step): 
                
                sliced_data.append(get_features(X,labels,column,k,k+step))
                
                if( len(sliced_data[-1][0]) == 0):
                    
                    
                    sliced_data.pop(-1)
                    continue
                regressions.append( regression( 0,sliced_data[-1][0],sliced_data[-1][1],column,0 ) )
                
            while(len(sliced_data)>3):
               
                sliced_data,regressions = merge(sliced_data,regressions,column)
                
            
            error = 0
            for i in range(len(sliced_data)):
                
                error += calculate_error(regressions[i],sliced_data[i][0],sliced_data[i][1],column,0,0)
            #sum the error after the split
            #choose min
            
            if(min_error == -1):
                min_sliced_data = sliced_data
                min_regressions = regressions
                min_steps = step
                min_column = column
                min_error = error
                continue
                
            if(error<min_error):
                min_sliced_data = sliced_data
                min_regressions = regressions
                min_steps = step
                min_column = column
    

    if(min_steps < 2 and min_steps>=0):
            steps[min_column] += 1
    
    return min_sliced_data,min_regressions,min_column,steps


# In[347]:


def build_tree(X,feature_ids,labels,leaf_size,var,depth,steps):
    
    #Két felé bontjuk az adatokat úgy hogy a legkisseb mse kapjuk

    boundaries = [] # last element of each bach
    
    
    
    split = 0 
    sub_nodes = []
    
    st = steps.copy()
    
    
    split_data,split_regression,column,st = find_best_split(X,labels,feature_ids,st)
 
    
    for i,j in zip(split_data,split_regression):


        node = Node(i[1],i[0],feature_ids)#no need for feature_ids -> object inheritance   
        node.split = column   #column

        #split point

        node.split_result = i[0][-1][column]
        node.depth = depth
        node.regr = j

        node.top = i[0][-1][column]
        node.bottom = i[0][0][column]
        node.steps = st
        print(node.steps)


        #Ha egy bizonyos error érték alá megyünk vagy elértünk egy bizonyos elemszámot akkor a node-ot leaf-nek nyilvánítjuk
        if(len(node.labels) <= leaf_size or calculate_error(node.regr,node.X,node.labels,split,0,3)<1):#############################
            node.leaf = 1


        else:
            node.leaf = 0
        sub_nodes.append(node)
    
    
    
    if(depth == 2000):
        for i in sub_nodes:
            i.leaf = 1
            print("leaf2")
        return sub_nodes
    
    depth +=1

    
    leaf =  0
    for i in sub_nodes:
        print(i.steps)
        if(i.leaf == 1):
            leaf +=1
    #if we reach limit return nodes 
            
            
    if(leaf == len(sub_nodes) and leaf != 0):

        return sub_nodes
    
    else:
        for node in sub_nodes:
            if(node.leaf == 0):
                    node.nodes = build_tree(node.X,node.feature_ids,node.labels,leaf_size,var,depth,node.steps)
   

               

    return sub_nodes   


# In[388]:




def rebuild_tree(X,feature_ids,labels,leaf_size,var,depth,steps,prev_node):
    
    #Két felé bontjuk az adatokat úgy hogy a legkisseb mse kapjuk
    #print(prev_node.depth)
    boundaries = [] # last element of each bach
    
    
    
    split = 0 
    sub_nodes = []
    
    st = steps.copy()
    
    
    split_data,split_regression,column,st = find_best_split(X,labels,feature_ids,st)
    
    
    rebuild = 0
    
    if(len(prev_node.nodes)>0):
            err_old = 0
            for i in range(len(prev_node.nodes)):
                err_old += calculate_error(prev_node.regr,prev_node.nodes[i].X,prev_node.nodes[i].labels,
                                                prev_node.nodes[i].split,0,3)
                err_old += calculate_error(prev_node.nodes[i].regr, split_data[i][0], split_data[i][1],
                                                column, 0, 3)

            err_new = 0
            for i,j in zip(split_data,split_regression):
                err_new += calculate_error(j, i[0], i[1],
                                                column, 0, 3)
                err_new += calculate_error(j, i[0], i[1],
                                                column, 0, 3)

            diff = err_new/err_old
            min_err = min(err_new,err_old)
            print(diff)

            if(diff>0.1 and diff<1 and min_err == err_new):
                rebuild = 1
                #rebuild
    
    
    if(rebuild == 0):
        for node in prev_node.nodes:
                if(node.leaf == 0):
                    node.nodes = rebuild_tree(node.X,node.feature_ids,node.labels,leaf_size,var,depth,node.steps,node)
        
                    
    else:
        for node in prev_node.nodes:
                if(node.leaf == 0):
                    node.nodes = build_tree(prev_node.X,prev_node.feature_ids,prev_node.labels,leaf_size,var,depth,prev_node.steps)
         
    return prev_node.nodes


# In[389]:


def inicialize(X,feature_names,labels,leaf_size,var,depth,prev_nodes,rebuild):
    

    feature_ids = [x for x in range(len(feature_names))]
    tree = Tree(X,feature_names,labels)
    steps = []
    for i in feature_ids:
        steps.append(0)
    if(rebuild == 1):
        
        for i in prev_nodes.nodes:
            i.nodes = rebuild_tree(X,feature_ids,labels,leaf_size,var,depth,steps,i)
        
    else:
        tree.nodes = build_tree(X,feature_ids,labels,leaf_size,var,depth,steps)
   
    return tree


# In[ ]:


#build_tree(training_data,header,labels)
forest = []
bottom = 1
top = 2

            

print(type(y_train))
print(type(batch_y[1]))

print("-------------------------------------------------------")
print("Iteration: %d"%(1))

tree = inicialize(batch_x[0],header,batch_y[0],10,1,0,0,0)
forest.append(tree)


print("-------------------------------------------------------")
print("Iteration: %d"%(1))
for i,j in zip(batch_x[1:],batch_y[1:]):   
        tree = inicialize(i,header,j,10,1,0,tree,1)

    


# In[404]:




#print(forest[0].nodes[0].labels)


# In[405]:


def print_tree(node,num):
    for i in node.nodes:
        if(i.leaf != 1):
            print(len(i.labels))
            print_tree(i,num+1)
        else:
            print(i.labels)
   


# In[406]:


def print_tree(node,num):
    for i in node.nodes:
        if(i.leaf != 1):
            print(i.steps)
            print_tree(i,num+1)
        else:
            print(i.steps)
           
   


# In[417]:


#print_tree(forest[0],0)


# In[408]:


def print_tree(node,num):
    for i in node.nodes:
        if(i.leaf != 1):
            
            for j in range(num):
                print(' ',end='')
            print(i.split,i.bottom,i.top,end = '   ')
            print(i.steps)
            if(i.split == -1):
                for j in range(num):
                    print(' ',end='')
                    print(i.labels)
            
            print_tree(i,num+1)
        else:
            for j in range(num):
                print(' ',end='')
            print(i.split,i.bottom,i.top)


# In[409]:


#print_tree(tree,0)
#Egy fa leveleinek elemszáma
#Nem egyenlő az előre megadott határtól így tudjuk hogy az MSE szerint döntött így


# In[410]:



def predict(node,value):
    #split_value
    #melyik érték alapján történt a split
    if(node.leaf == 1 ):#regression hasznalata
        p = node.regr.predict(np.array(value[node.split]).reshape((1,-1)))

        
        return p
    else:

        c=0
        for i in node.nodes:
           
            if (i.top>=value[i.split] and i.bottom<=value[i.split]):
                #print(i.split)
                c+=1
                p=predict(i,value)
                return p
        if(c == 0):
            p = node.regr.predict(np.array(value[node.split]).reshape((1,-1)))
            return p
            print("nope")
            print(node.split)
            print(p)


# In[411]:


#Kiszámítjuk a teszt és a training data alapján az error mértékét
#Valamint a test adat alapján összehasonlítjuk az eredménnyel

diff = []
difference = []




for i in forest:
    
    d = []
    error = 0
    for j in range(len(X_test)): 
        #print(X_test[j])
        p = predict(i,X_test[j])
        #print(p)
        if(p == None):
            d.append(0)
            continue
        #A few extreme situations appear, so I do not take them into consideration, in order to get a clear view during visualization
        if(abs(y_test[j]-p[0])<1000 and p>0):
            error += abs(y_test[j]-p)
            d.append(p)
        else:
            d.append(0)
        
        #print(y_test[j],p)
        
    
    diff.append(d)
    difference.append(error)
    break
print(max(difference))
print(X_test[0])


# In[412]:


plt.rcParams["figure.figsize"] = (15,7)
print(len(diff[0]))
print(len(y_test))
plt.scatter([ X_test[i][0] for i in range(len(X_test)) ],y_test,s=5)
plt.scatter([ X_test[i][0] for i in range(len(X_test)) ],diff,c="r",s=5)
plt.legend(["Testing data" , "Predicted value"])
print(sum(y_test))
plt.xlabel("Square meter")
plt.ylabel("Price")
plt.title("Test data")

plt.show()


# In[413]:


plt.rcParams["figure.figsize"] = (15,7)
print(len(X_test))
print(len(y_test))
plt.scatter([ X_test[i][1] for i in range(len(X_test)) ],y_test,s=5)
plt.scatter([ X_test[i][1] for i in range(len(X_test)) ],diff,c="r",s=5)
plt.legend(["Testing data" , "Predicted value"])
print(sum(y_test))
plt.xlabel("Distance from center meter")
plt.ylabel("Price")
plt.title("Test data")

plt.show()


# In[414]:


plt.rcParams["figure.figsize"] = (15,7)
print(len(X_test))
print(len(y_test))
plt.scatter([ X_test[i][2] for i in range(len(X_test)) ],y_test,s=5)
plt.scatter([ X_test[i][2] for i in range(len(X_test)) ],diff,c="r",s=5)
plt.legend(["Testing data" , "Predicted value"])
print(sum(y_test))
plt.xlabel("Salary")
plt.ylabel("Price")
plt.title("Test data")

plt.show()


# In[415]:


plt.rcParams["figure.figsize"] = (15,7)
print(len(X_test))
print(len(y_test))
plt.scatter([ X_test[i][3] for i in range(len(X_test)) ],y_test,s=5)
plt.scatter([ X_test[i][3] for i in range(len(X_test)) ],diff,c="r",s=5)
plt.legend(["Testing data" , "Predicted value"])
print(sum(y_test))
plt.xlabel("AVG salary")
plt.ylabel("Price")
plt.title("Test data")

plt.show()

