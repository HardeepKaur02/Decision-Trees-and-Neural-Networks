from collections import Counter,defaultdict,deque
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
import sys
import math
import time
import heapq
import csv
NUMERIC = 0
CATEGORICAL = 1
N = 17

def hotEncode(i,val):   
    dic= [[],['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'],[ 'divorced','married','single','unknown'],['basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'],['no','yes','unknown'],[],['no','yes','unknown'],['no','yes','unknown'],['cellular','telephone'],[],[ 'jan', 'feb', 'mar','apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'],[],[],[],[],['failure','nonexistent','success']]
    ans = []
    if dic[i] == []:
        ans.append(val)
        return ans
    for j in range(len(dic[i])):
        if dic[i][j] == val:
            ans.append(True)
        else:
            ans.append(False)
    return ans

def HotEncode(file_name):
    file_in = open(file_name)
    csvReader = csv.reader(file_in,delimiter = ';')
    attr_types = [0,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1]
    encoded_data,y = [],[]
    bool_map = {"yes": True, "no": False}    
    fields = next(csvReader)
    for row in csvReader:
        if str(row[-1]) == 'yes':
            y.append(True)
        if str(row[-1]) == 'no':
            y.append(False)
        x_row = []
        for j in range(16):
            if attr_types[j] == 0:
                entry = hotEncode(j,float(row[j]))
            else:
                entry = hotEncode(j,str(row[j]))
            x_row.extend(entry)
        encoded_data.append(x_row)
        #print(x_row,len(x_row))       
    return encoded_data,y
def choose_attribute(x, attr_types):
    j_best, split_values, min_entropy = -1, [] , float('inf')
    y = x[:, -1]
    num_attr = len(attr_types)
    for j in range(num_attr):
        w = x[:,j]
        med = 0
        if attr_types[j] == NUMERIC:
            w = w.astype('float32')
            med = np.median(w)
            y_split =  [y[w<=med], y[w>med]]
        else: # multi split
            attr_vals = np.unique(w)
            for uq_attr in attr_vals:
                y_split.append(y[w==uq_attr].tolist())
        entropy = 0
        for y_ in y_split: #for each value possible for attribute xj in data
            Hy_xj = 0
            counts = Counter(y_)
            counts = list(counts.values())
            counts = np.array(counts).astype('float32')
            probs = counts/len(y_)
            Hy_xj = -1 * np.sum(probs * np.log(probs))
            entropy += len(y_) * Hy_xj / len(y)
        if entropy < min_entropy:
            min_entropy = entropy
            j_best = j
            if attr_types[j] == NUMERIC:
                split_values = [med]
                split_on_num = True
            else:
                split_values = attr_vals
                split_on_num = False
    if j_best == -1:
        return -1, [], False
    
    return j_best, split_values, split_on_num

class Node:
    def __init__(self, x, x_test=None, x_valid=None, par=None,def_cl = "yes",level = 0):
        
        self.parent = par
        self.num_children =0
        self.children = []
        self.is_leaf = True
        
        self.x = x
        self.x_test = x_test
        self.x_valid = x_valid
        
        self.attr_split = -1 # if it is an internal node
        self.level = level # BFS level of the node in the tree
        
        y_temp = x[:,-1]
        
        self.class_freq = Counter(y_temp)
        if not bool(self.class_freq):
            self.pred = def_cl
        else:
            self.pred = max(self.class_freq.items(), key=itemgetter(1))[0]
        self.split_values = []   #single value for numerical, all possible values for categorical
        self.split_on_num = False
        
        # correct_1: accuracy if this were a leaf node 
        self.correct = 0
        self.correct_1 = 0
        self.correct_test = 0
        self.correct_1_test = 0
        self.correct_train = 0
        self.correct_1_train = 0
        self.is_deleted = False

    def __lt__(self, node):
        return node.correct < self.correct
     
class DecisionTree:

    def __init__(self,train_data=None,test_data=None,val_data=None,attr_types=None,threshold=1.0,pruning=False,max_nodes=float('inf')):
        self.train_acc = []
        self.test_acc = []
        self.valid_acc = []
        self.valid_acc_2 = []
        self.train_acc_2 = []
        self.test_acc_2 = []
        self.pruned_tree_sizes = []
        if train_data is not None:
            self.grow_tree(train_data=train_data,test_data=test_data,val_data=val_data,attr_types=attr_types,threshold=threshold,pruning=pruning,max_nodes=max_nodes)
        else:
            self.root = None

    def predict(self, test_data):
        predicted = []
        for x in test_data:
            node = self.root
            while not node.is_leaf:
                if node.split_on_num:
                    split_value = node.split_values[0]
                    if x[node.attr_split] <= split_value:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if x[node.attr_split] in node.split_values:
                        k = node.split_values.index(x[node.attr_split])
                    else:
                        k=-1
                    node = node.children[k]
            predicted.append(node.pred)
        return np.array(predicted)

    def grow_tree(self,train_data,test_data,val_data,attr_types,threshold,pruning,max_nodes):

        self.root = Node(x=train_data, x_test=test_data, x_valid=val_data,level = 0)
        q = deque()   # grow tree using BFS
        q.appendleft(self.root)
        ### keep a track of all the nodes, to create heap later for pruning ###
        nodes = [self.root]

        train_accuracy, test_accuracy, valid_accuracy = 0, 0, 0
        y_train, y_test, y_valid = train_data[:, -1], test_data[:, -1], val_data[:, -1]

        m_valid = val_data.shape[0]
        m_test = test_data.shape[0]
        m_train = train_data.shape[0]

        def num_correct(node, i=0):
            # i= 0 for train data, 1 for test data, 2 for validation data
            correct_val = node.pred
            arr = []
            if i==0:
                arr = node.x[:, -1]
            elif i ==1:
                arr = node.x_test[:, -1]
            else:
                arr = node.x_valid[:, -1]
            return np.count_nonzero(arr == correct_val)

        correct_train = num_correct(self.root)
        correct_test = num_correct(self.root,1)
        correct_valid = num_correct(self.root,2)

        while train_accuracy < threshold and q and len(nodes) < max_nodes:
        
            node = q.pop()
            # if node is pure
            total_freq = 0
            if len(node.class_freq.values()) > 0:
                total_freq = sum(node.class_freq.values())
            # max_freq = max(node.class_freq.values())
            #print(node.class_freq,len(node.x))
            if len(node.class_freq) == 1:
                node.x = None
            else:
                node.attr_split, node.split_values, node.split_on_num = choose_attribute(node.x, attr_types)
                j = node.attr_split
                if j == -1:
                    node.x = None
                    continue
                if node.split_on_num == True:
                    split_value = node.split_values[0]
                    left_x = node.x[node.x[:, j].astype('float32') <= split_value].reshape(-1,N)
                    right_x = node.x[node.x[:, j].astype('float32') > split_value].reshape(-1,N)
                    left_x_test = node.x_test[node.x_test[:, j].astype('float32') <= split_value].reshape(-1,N)
                    right_x_test = node.x_test[node.x_test[:, j].astype('float32') > split_value].reshape(-1,N)
                    left_x_valid = node.x_valid[node.x_valid[:, j].astype('float32') <= split_value].reshape(-1,N)                    
                    right_x_valid = node.x_valid[node.x_valid[:, j].astype('float32') > split_value].reshape(-1,N)
                    node.num_children+=1
                    node.children.append(Node(x=left_x, x_test=left_x_test, x_valid=left_x_valid, par=node,level = node.level +1))
                    q.appendleft(node.children[0])
                    nodes.append(node.children[0])
                    
                    node.is_leaf = False
                    left_y = node.children[0].pred
                    def_cl = "yes"
                    if left_y == "yes":
                        def_cl = "no"
                    node.children.append(Node(x=right_x, x_test=right_x_test, x_valid=right_x_valid, par=node,def_cl = def_cl,level = node.level +1 ))
                    if right_x.shape[0] != 0:
                        q.appendleft(node.children[1])
                    nodes.append(node.children[1])
                    node.num_children +=1 
                    
                    
                else:
                    for kids in range(len(node.split_values)):
                        kid_x = node.x[node.x[:, j] == node.split_values[kids]]
                        kid_x_test = node.x_test[node.x_test[:, j] == node.split_values[kids]]
                        kid_x_valid = node.x_valid[node.x_valid[:, j] == node.split_values[kids]]
                        node.children.append(Node(x = kid_x, x_test = kid_x_test, x_valid = kid_x_valid, par = node,level = node.level +1))
                        node.num_children += 1
                        q.appendleft(node.children[kids])
                        nodes.append(node.children[kids])
                    
                correct_train -= num_correct(node)
                correct_test  -=  num_correct(node,1)
                correct_valid -= num_correct(node,2)
                for kids in range(node.num_children):
                    correct_train += num_correct(node.children[kids])
                    correct_test  += num_correct(node.children[kids],1)
                    correct_valid += num_correct(node.children[kids],2)                    
                train_accuracy = 100 * correct_train / m_train
                test_accuracy = 100 * correct_test / m_test
                valid_accuracy = 100 * correct_valid / m_valid
                self.train_acc.append(train_accuracy)
                self.test_acc.append(test_accuracy)
                self.valid_acc.append(valid_accuracy)

                node.x, node.class_freq = None, None
                node.x_test, node.x_valid = None, None
        # no need of data now
        max_level = 0
        for node in nodes:
            node.x = None
            node.x_valid = None
            node.x_test = None
            max_level = max(max_level,node.level)
        print(max_level)
        # print("Waheguru Ji")


        if not pruning:
            return
        # print(self.root.attr_split)
        # print(self.root.split_values[0])
        # print(self.root.children[0].is_leaf)
        # print(self.root.children[0].split_values[0])
        # print(self.root.children[1].is_leaf)
        # print(self.root.children[1].split_values[0])

        print("W")

        ################ HELPER FUNCTIONS ###############
        def num_accurate(n, data, option=3):
            accurate = np.count_nonzero( data[:, -1] == n.pred)  #if this were a leaf node
            if option == 3:
                n.correct_1 = accurate
            elif option == 2:
                n.correct_1_test = accurate
            else:
                n.correct_1_train = accurate
            if not n.is_leaf:
                if n.split_on_num:
                    split_value = float(n.split_values[0])
                    data_left = data[data[:, n.attr_split].astype('float32') <= split_value]
                    data_right = data[data[:, n.attr_split].astype('float32') > split_value]
                    accurate = num_accurate(n.children[0], data_left, option)
                    accurate += num_accurate(n.children[1], data_right, option)
                else:
                    accurate = 0
                    for kids in range(num_children):
                        data_kid = data[data[:,n.attr_split] == n.split_values[kids]]
                        accurate += num_accurate(n.children[kids],data_kid,option)
            if option == 3:
                n.correct = accurate
            elif option == 2:
                n.correct_test = accurate
            else:
                n.correct_train = accurate
            return accurate

        datas = [train_data,test_data,val_data]
        for i in range(3):
            num_accurate(self.root,datas[i],i+1)

        def num_accurate_root(n,heap):
            while n.parent is not None:
                n.parent.correct = 0
                n.parent.correct_test = 0
                n.parent.correct_train = 0
                for kids in range(n.parent.num_children):
                    n.parent.correct += n.parent.children[kids].correct
                    n.parent.correct_test += n.parent.children[kids].correct_test
                    n.parent.correct_train += n.parent.children[kids].correct_train
                heapq.heappush(heap, (n.parent.correct - n.parent.correct_1, n.parent))
                n = n.parent        

        def del_tree(n):
            n.is_deleted = True
            if n.is_leaf:
                return 1
            else:
                res = 1
                for kids in range(n.num_children):
                    res += del_tree(n.children[kids])
                return res
        #################################################
                     ### PRUNING STARTS ###
        #################################################

        num_nodes = len(nodes)
        heap = []
        for node in nodes:
            if not node.is_leaf and node.level > 0:
                heapq.heappush(heap, (node.correct - node.correct_1, node))

        while heap:
            diff, n = heapq.heappop(heap)
            if n.is_deleted or (n.correct - n.correct_1 != diff):
                continue
            if diff >= 0:
                break
            if n.level > max_level//2:
                num_nodes -= del_tree(n)
                n.correct = n.correct_1
                n.correct_test = n.correct_1_test
                n.correct_train = n.correct_1_train
                n.is_leaf = True
                n.num_children = 0
                n.split_values = []
                n.children = []
                num_accurate_root(n, heap)
                self.valid_acc_2.append(100 * self.root.correct / val_data.shape[0])
                self.train_acc_2.append(100 * self.root.correct_train / train_data.shape[0])
                self.test_acc_2.append(100 * self.root.correct_test / test_data.shape[0])
                self.pruned_tree_sizes.append(num_nodes)
        return

def getScore(n_estimators, max_features, min_samples_split):
    classifier = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,bootstrap=True,criterion='entropy',oob_score=True,n_jobs=4)
    classifier.fit(train,y_train)
    oob_score = classifier.oob_score_
    y_pred_test = classifier.predict(test)
    y_pred_valid = classifier.predict(valid)
    test_acc = (y_pred_test == y_test).sum() / len(y_test)
    valid_acc = (y_pred_valid == y_valid).sum() / len(y_valid)
    print(n_estimators, max_features, min_samples_split, ':', oob_score, test_acc, valid_acc)
    return (oob_score*100, test_acc*100, valid_acc*100)


def main():
    part = sys.argv[1]
    if part == "a":
        train = np.loadtxt('bank_train.csv',dtype={'names': ('age', 'job', 'marital', 'education', 'default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'),
                'formats': ('f4','S15','S10','S25','S10', 'f4', 'S10','S10','S10', 'f4','S10', 'f4','f4','f4','f4', 'S15','S3')}, delimiter=';', skiprows=1)
        test = np.loadtxt('bank_test.csv', dtype={'names': ('age', 'job', 'marital', 'education', 'default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'),
                'formats': ('f4','S15','S10','S25','S10', 'f4', 'S10','S10','S10', 'f4','S10', 'f4','f4','f4','f4', 'S15','S3')}, delimiter=';', skiprows=1)
        valid = np.loadtxt('bank_val.csv',dtype={'names': ('age', 'job', 'marital', 'education', 'default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'),
                'formats': ('f4','S15','S10','S25','S10', 'f4', 'S10','S10','S10', 'f4','S10', 'f4','f4','f4','f4', 'S15','S3')}, delimiter=';', skiprows=1)

        train = train.tolist()
        test = test.tolist()
        valid = valid.tolist()
        train = np.array(train).reshape(-1,N)
        test = np.array(test).reshape(-1,N)
        valid = np.array(valid).reshape(-1,N)

        attr_types = [0,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1] # 0 for numeric attributes, 1 for categorical
        
        decision_tree = DecisionTree(train_data=train,test_data=test,val_data=valid,attr_types=attr_types,threshold=99.0,pruning=False,max_nodes = 5500)
        x = list(range(1, 2 * len(decision_tree.train_acc) + 1, 2))
        plt.xlabel('Number of nodes')
        plt.ylabel('Accuracy (in %)')
        plt.plot(x, decision_tree.train_acc, label='Training accuracy')
        plt.plot(x, decision_tree.test_acc, label='Test accuracy')
        plt.plot(x, decision_tree.valid_acc, label='Validation accuracy')
        print('final train accuracy:', decision_tree.train_acc[-1])
        print('final test accuracy:', decision_tree.test_acc[-1])
        print('final validation accuracy:', decision_tree.valid_acc[-1])
        plt.legend()
        plt.savefig('accuracies_pruning1.png')
        plt.close()
    
    if part == "b":
        train = np.loadtxt('bank_train.csv',dtype={'names': ('age', 'job', 'marital', 'education', 'default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'),
                'formats': ('f4','S15','S10','S25','S10', 'f4', 'S10','S10','S10', 'f4','S10', 'f4','f4','f4','f4', 'S15','S3')}, delimiter=';', skiprows=1)
        test = np.loadtxt('bank_test.csv', dtype={'names': ('age', 'job', 'marital', 'education', 'default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'),
                'formats': ('f4','S15','S10','S25','S10', 'f4', 'S10','S10','S10', 'f4','S10', 'f4','f4','f4','f4', 'S15','S3')}, delimiter=';', skiprows=1)
        valid = np.loadtxt('bank_val.csv',dtype={'names': ('age', 'job', 'marital', 'education', 'default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'),
                'formats': ('f4','S15','S10','S25','S10', 'f4', 'S10','S10','S10', 'f4','S10', 'f4','f4','f4','f4', 'S15','S3')}, delimiter=';', skiprows=1)

        train = train.tolist()
        test = test.tolist()
        valid = valid.tolist()
        train = np.array(train).reshape(-1,N)
        test = np.array(test).reshape(-1,N)
        valid = np.array(valid).reshape(-1,N)

        attr_types = [0,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1] # 0 for numeric attributes, 1 for categorical
        

        decision_tree = DecisionTree(train_data=train,test_data=test,val_data=valid,attr_types=attr_types,threshold=99.0,pruning=True)

        x = list(range(1, 2 * len(decision_tree.train_acc) + 1, 2))

        print('initial train accuracy:', decision_tree.train_acc[-1])
        print('initial test accuracy:', decision_tree.test_acc[-1])
        print('initial validation accuracy:', decision_tree.valid_acc[-1])

        print('post pruning train accuracy:', decision_tree.train_acc_2[-1])
        print('post pruning test accuracy:', decision_tree.test_acc_2[-1])
        print('post pruning validation accuracy:', decision_tree.valid_acc_2[-1])
        y_pt = decision_tree.valid_acc[100]
        x_pt = x[100]
        print(x_pt,y_pt)
        plt.xlabel('Number of nodes')
        plt.ylabel('Accuracy (in %)')
        plt.ylim([84,100])
        plt.vlines(x_pt,84,y_pt,linestyle = "dashed",colors = "r")
        plt.plot(x, decision_tree.train_acc, label='Training accuracy')
        plt.plot(x, decision_tree.test_acc, label='Test accuracy')
        plt.plot(x, decision_tree.valid_acc, label='Validation accuracy')
        plt.legend()
        plt.savefig('decision_tree_accuracies.png')
        plt.show()
        plt.close()
        plt.xlabel('Number of nodes')
        plt.ylabel('Accuracy (in %)')
        plt.plot(decision_tree.pruned_tree_sizes, decision_tree.valid_acc_2, label='Validation accuracy')
        plt.plot(decision_tree.pruned_tree_sizes, decision_tree.train_acc_2, label='Training accuracy')
        plt.plot(decision_tree.pruned_tree_sizes, decision_tree.test_acc_2, label='Test accuracy')
        plt.legend()
        plt.savefig('decision_tree_post_pruning.png')
        plt.show()
    
    if part == "c":
        train,y_train = HotEncode('bank_train.csv')
        test,y_test = HotEncode('bank_test.csv')
        valid,y_valid = HotEncode('bank_val.csv')

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        scores = []

        domain_n_estimators = [50, 150, 250, 350, 450] # 50 to 450
        domain_max_features = [0.1, 0.3, 0.5, 0.7, 0.9] # 0.1 to 1.0
        domain_min_samples_split = [2, 4, 6, 8, 10] # 2 to 10
        ##### Using GridSearchCV
        # parameter_grid = {'n_estimators': n_estimators, 'max_features': max_features,'min_samples_split': min_samples_split,'bootstrap': [True],'oob_score': [True],'criterion': ["entropy"]}
        # clf = RandomForestClassifier()
        # best_model = GridSearchCV(estimator = clf,param_grid = parameter_grid,n_jobs= 4)
        # best_model.fit(train,y_train)
        #print(classifier.cv_results_)
        best_oob_score = -1
        best_n_estimators, best_min_samples_split, best_max_features = -1, -1, -1
        best_model = None

        for n_estimators in domain_n_estimators:
            for max_features in domain_max_features:
                for min_samples_split in domain_min_samples_split:
                    t = time.time()
                    classifier = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,bootstrap=True,oob_score=True,n_jobs=4,criterion = "entropy")
                    classifier.fit(train, y_train)
                    oob_score = classifier.oob_score_
                    if oob_score > best_oob_score:
                        best_oob_score = oob_score
                        best_n_estimators = n_estimators
                        best_max_features = max_features
                        best_min_samples_split = min_samples_split
                        best_model = classifier

        print(best_n_estimators, best_max_features, best_min_samples_split)
        print('oob score:', best_oob_score)
        y_pred_test = best_model.predict(test)
        y_pred_valid = best_model.predict(valid)
        y_pred_train = best_model.predict(train)
        print('Training:', (y_pred_train == y_train).sum() / len(y_train))
        print('Validation:', (y_pred_valid == y_valid).sum() / len(y_valid))
        print('Test: ', (y_pred_test == y_test).sum() / len(y_test))

    if part == "d":
        train,y_train = HotEncode('bank_train.csv')
        test,y_test = HotEncode('bank_test.csv')
        valid,y_valid = HotEncode('bank_val.csv')

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        scores = []

        domain_n_estimators = [50, 150, 250, 350, 450] # 50 to 450
        domain_max_features = [0.1, 0.3, 0.5, 0.7, 0.9] # 0.1 to 1.0
        domain_min_samples_split = [2, 4, 6, 8, 10] # 2 to 10

        n_test, n_val, n_oob = [], [], []
        # Varying n_estimators
        for n in domain_n_estimators:
            oob, test_acc, val_acc = getScore(n,0.5,4)
            n_oob.append(oob)
            n_test.append(test_acc)
            n_val.append(val_acc)

        f_test, f_val, f_oob = [], [], []
        # Varying max_features
        for f in domain_max_features:
            oob, test_acc, val_acc = getScore(250,f,4)
            f_oob.append(oob)
            f_test.append(test_acc)
            f_val.append(val_acc)

        s_test, s_val, s_oob = [], [], []
        # Varying min_samples_split
        for s in domain_min_samples_split:
            oob, test_acc, val_acc = getScore(250,0.5,s)
            s_oob.append(oob)
            s_test.append(test_acc)
            s_val.append(val_acc)

        plt.xlabel('Number of estimators')
        plt.ylabel('Accuracy (in %)')
        # plt.yticks(0.1)
        # plt.ylim(85,95)
        x_pt = 250
        plt.ylim([85.6,91.8])
        plt.vlines(x_pt,85.6,91.8,linestyle = "dashed",colors = "r")
        plt.plot(domain_n_estimators, n_oob, label='Out of bag')
        plt.plot(domain_n_estimators, n_test, label='Test')
        plt.plot(domain_n_estimators, n_val, label='Validation')
        plt.legend()
        plt.savefig('Estimator_sensitivity.png')
        plt.show()
        plt.close()
        plt.xlabel('Fraction of features used')
        plt.ylabel('Accuracy (in %)')
        x_pt = 0.5
        plt.ylim([85.8,91.8])
        plt.vlines(x_pt,85.8,91.8,linestyle = "dashed",colors = "r")
        plt.plot(domain_max_features, f_oob, label='Out of bag')
        plt.plot(domain_max_features, f_test, label='Test')
        plt.plot(domain_max_features, f_val, label='Validation')
        plt.legend()
        plt.savefig('Feature_sensitivity.png')
        plt.show()
        plt.close()
        plt.xlabel('Minimum samples needed for split')
        plt.ylabel('Accuracy (in %)')
        x_pt = 4
        plt.ylim([85.6,92.0])
        plt.vlines(x_pt,85.6,92.0,linestyle = "dashed",colors = "r")
        plt.plot(domain_min_samples_split, s_oob, label='Out of bag')
        plt.plot(domain_min_samples_split, s_test, label='Test')
        plt.plot(domain_min_samples_split, s_val, label='Validation')
        plt.legend()
        plt.savefig('Min_samples_split_sensitivity.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()