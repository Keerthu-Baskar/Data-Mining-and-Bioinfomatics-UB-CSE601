import numpy as np
from collections import defaultdict
from random import sample

file = input("Enter the name of the file! ")

with open(file) as textFile:
    lines=[line.split() for line in textFile]


input_data=np.array(lines)

attributes = input_data[:,0:-1]
category_count = 0

category_mapping = {}
category_list = []
for i in range(attributes.shape[1]):
    if attributes[0][i].isalpha():
        current_col = attributes[:,i]
        for ind , ele in enumerate(current_col):
            if ele not in category_mapping:
                category_mapping[ele] = category_count
                category_count+=1
            current_col[ind] = float(category_mapping[ele])
        category_list.append(i)

ground_truth_labels = np.array(input_data[:,-1].reshape((len(input_data),1)),dtype=int)
attributes=np.array(attributes, dtype=float)
data=np.concatenate((attributes,ground_truth_labels),axis=1)



def calculate_split_point(dataset , no_of_features):

    def dataset_split(dataset , value , column):
        left = []
        right = []

        for row in dataset:
            if row[column] < value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left) , np.array(right)


    def calculate_gini_index(left_node , right_node):
        total_gini = 0.0
        total_group_size = float(len(left_node)+ len(right_node))
        if len(left_node) > 0:
            left_score = 0.0
            left_size = float(len(left_node))
            left_zero_class = list(left_node[:,-1]).count(0)/left_size
            left_one_class = list(left_node[:,-1]).count(1) / left_size
            left_score += left_zero_class*left_zero_class
            left_score+= left_one_class * left_one_class
            total_gini+=(1.0 - left_score)*(left_size / total_group_size)

        if len(right_node) > 0:

            right_score = 0.0
            right_size = float(len(right_node))
            right_zero_class = list(right_node[:,-1]).count(0)/right_size
            right_one_class = list(right_node[:,-1]).count(1) / right_size
            right_score+= right_zero_class*right_zero_class
            right_score+= right_one_class * right_one_class
            total_gini+= (1.0 - right_score)*(right_size/total_group_size)
        return total_gini


    min_error = float('inf')
    split_point = None
    train_cols = [i for i in range(dataset.shape[1]-1)]
    columns = sample(train_cols , no_of_features )
    # columns.append(sample(train_cols , no_of_features ))


    for cols in columns:
        for row in dataset:
            left_node , right_node = dataset_split(dataset , row[cols] , cols)
            current_gini_score = calculate_gini_index(left_node , right_node)
            if current_gini_score < min_error:
                min_error = current_gini_score
                split_point = {'attr': cols , 'value': row[cols] , 'left': left_node , 'right': right_node}
    return split_point

def decision_tree_build(node):

    def terminal_nodes(left_node , right_node):
        total_zeroes = total_ones = 0

        if len(left_node) > 0:
            total_zeroes += list(left_node[:,-1]).count(0)
            total_ones+= list(left_node[:,-1]).count(1)
        if len(right_node) > 0:
            total_zeroes+= list(right_node[:,-1]).count(0)
            total_ones+= list(right_node[:,-1]).count(1)
        return 1 if total_ones > total_zeroes else 0

    left = node['left']
    node.pop('left',None)
    right = node['right']
    node.pop('right',None)
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = terminal_nodes(left, right)
        return node 
    if len(set(left[:,-1])) == 1:
        node['left'] = terminal_nodes(left, [])
    else:
        node['left'] = decision_tree_build(calculate_split_point(left , no_of_rf))
    if len(set(right[:,-1])) == 1:
        node['right'] = terminal_nodes([], right)
    else:
        node['right'] = decision_tree_build(calculate_split_point(right , no_of_rf))
    return node

def make_prediction(node, row):
    if row[node['attr']] < node['value']:
        if type(node['left']) is not dict:
            return node['left']
        else:
            return make_prediction(node['left'] , row)    

    if type(node['right']) is not dict:
        return node['right']
    else:
        return make_prediction(node['right'] , row)


data_chunks = np.array_split(data,10)
data_chunks = np.array(data_chunks)
accuracy = precision = recall = f1_score = 0.0
no_of_rf = int((data.shape[1])*0.2)

no_of_trees = int(input("Enter the number of tree: "))

def random_forest(train_data , test_data):
    training_data_length = len(train_data)
    forest_prediction = defaultdict(list)
    for i in range((no_of_trees)):
        select_cols = np.random.choice(training_data_length, training_data_length, replace=True)
        train_subset = train_data[select_cols,:]
        root = {}
        root = calculate_split_point(train_subset,no_of_rf)
        root = decision_tree_build(root)

        for ind  in range(len(test_data)):
            forest_prediction[ind].append(make_prediction(root , test_data[ind]))
    final_prediction = []
    for k , v in forest_prediction.items():
        final_prediction.append(max(set(v) , key=v.count))
    return final_prediction




for iteration in range(10):
    predicted_values = []
    train_dataset = np.array(np.concatenate([y for (x,y) in enumerate(data_chunks,0) if x != iteration],axis=0))
    test_dataset = np.array(data_chunks[iteration])
    predicted_values = random_forest(train_dataset , test_dataset)
    actual_class = list(test_dataset[:,-1])
    true_positive = true_negative = false_positive = false_negative = 0 
    for i in range(len(actual_class)):
        if actual_class[i] == 1 and  predicted_values[i] == 1:
            true_positive+=1
        elif actual_class[i] == 0 and  predicted_values[i] == 0:
            true_negative+=1
        elif actual_class[i] == 0 and predicted_values[i] == 1:
            false_positive+=1
        else:
            false_negative+=1

    accuracy += (float((true_positive + true_negative)/(true_positive + false_negative + false_positive + true_negative)))*10
    if (true_positive+false_positive)!=0:
        precision += (float((true_positive)/(true_positive + false_positive)))*10
    if (true_positive+false_negative)!=0:
        recall += (float((true_positive)/(true_positive + false_negative)))*10
f1_score = 0.01*2*(precision*recall)/(precision+recall)
print("Accuracy: "+str(accuracy))
print("Precision: "+str(precision))
print("Recall: "+str(recall))
print("F1 Measure: "+str(f1_score))




