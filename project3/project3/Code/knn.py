import numpy as np

#getting the file

file=input("Enter the filename: ")
k = int(input("Enter the value for k: "))


with open(file) as textFile:
    lines=[line.split("\t") for line in textFile]

data=np.asarray(lines)

sample_data = data[0]

categ_data_cols=[]
for col in range(sample_data.size):
    feature = sample_data[col]
    if feature.isalpha():
        ft_array,indices = np.unique(data[:,col],return_inverse = True)
        data[:,col] = indices.astype(np.float)

        categ_data_cols.append(col)

#print(categ_data_cols)

#print(data[0])


points= np.matrix(data[:,:-1],dtype=float,copy=False)
#print("points")

#print(points)



#print(points)
#getting ground truth
ground_truth_labels = np.asarray(data[:,-1],dtype=int)

#print("length")
#print(len(points[0]))

#Normalsiing the data by taking min and max of each columns..


for col in range(points.shape[1]):
    if col not in categ_data_cols:

        v = points[:,col]
        #print("col")
        #print(col)
        #print(points[:,col])
        points[:,col] = (v - v.min()) / (v.max() - v.min())
        #print(points[:,col])




#print(ground_truth_labels)
# print(points)

#print(points)


#normalising data
'''
points_mean = np.mean(points , axis =0)
points_std = np.std(points , axis=0)
points = points - points_mean
points = points/points_std
'''

#print("notmalising points")
#print(points)




#splitting pounts into 10 groups
split_points = np.array_split(points,10)
# doing the same for labels
ground_truth_labels_split = np.array_split(ground_truth_labels,10)
#print("SPlit")
#print(ground_truth_labels_split)



def knn_implement(train_data,train_label,test_data,k):
    test_label=[]
    #print(len(train_data))
    #print(len(train_label))
    for index in range(len(test_data)):
        dist_list=[]
        for i in range(len(train_data)):
            dist = np.linalg.norm(train_data[i] - test_data[index])
            #dist=eucl_calc(train_data[i],test_data[index])
            dist_list.append([i,dist])

        #sort the distance list based on dist
        dist_list.sort(key = lambda x: x[1])

        neigh_list=[]

        label_list =[]
        for i in range(k):
            neighbour = dist_list[i]
            #neighbour=heappop(dist_list)
            neigh_list.append(neighbour)
            #print("NN")
            #print(neighbour)
            neigh_point=neighbour[0]

            label_list.append(train_label[neigh_point])
        #print("NN")
        #print(neigh_list)
        #similar to vstack purpose
        #print(label_list)
        test_label.append(max(set(label_list),key=label_list.count))

    return test_label

def performance_metric(test_actual_version, test_predicted_version):

    #samples classification
    true_pos = 0
    false_neg =0
    false_pos=0
    true_neg = 0

    # to identify the performance metrics
    acc = 0
    prec = 0
    recall = 0
    f1_measure = 0

    for i in range(len(test_predicted_version)):
        if test_actual_version[i] == 1 and test_predicted_version[i] == 1:
            true_pos += 1
        elif test_actual_version[i] == 1 and test_predicted_version[i] == 0:
            false_neg += 1
        elif test_actual_version[i] == 0 and test_predicted_version[i] == 1:
            false_pos += 1
        elif test_actual_version[i] == 0 and test_predicted_version[i] == 0:
            true_neg += 1

    acc += (float(true_pos+true_neg)/(true_pos+false_neg+false_pos+true_neg))

    if(true_pos+false_pos != 0):
        prec += (float(true_pos)/(true_pos+false_pos))

    if(true_pos+false_neg != 0):
        recall += (float(true_pos)/(true_pos+false_neg))

    f1_measure += (float(2*true_pos)/((2*true_pos)+false_neg+false_pos))

    return acc, prec, recall, f1_measure

acc_average=0
precision_average=0
recall_average=0
f1_Measure_average = 0

#the actual implementation starts.

for index in range(10):
    #9 groups for training data are put into one group

    train_data=np.asarray(np.vstack([x for i,x in enumerate(split_points) if i != index]))
    train_label=np.asarray(np.concatenate([x for i,x in enumerate(ground_truth_labels_split) if i != index]))
    #print(len(train_label))
    test_data=np.asarray(split_points[index])
    test_label=np.asarray(ground_truth_labels_split[index])

    test_predicted_label=knn_implement(train_data,train_label,test_data,k)

    print("Printing test Predicted labels")
    print(test_predicted_label)
    print("Printing test actual labels")
    print(test_label)
    acc, prec, recall, f1_measure = performance_metric(test_label, test_predicted_label)

    print("Accuracy:",acc*10)
    acc_average += acc*10

    print("Precision:",prec*10)
    precision_average += prec*10

    print("Recall:",recall*10)
    recall_average += recall*10

    print("F1_measure:",f1_measure)
    f1_Measure_average += f1_measure*0.1

print()
print("Average Accuracy Obtained:",acc_average)
print("Average Precision Obtained:",precision_average)
print("Average Recall Obtained:",recall_average)
print("Average F_measure Obtained:",f1_Measure_average)




