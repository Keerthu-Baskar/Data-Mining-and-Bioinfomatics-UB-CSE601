import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from pandas import DataFrame
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier

import csv


#getting the file

file_train=input("Enter the Train filename: ")
file_test = input("Enter the Test filename: ")

#k = int(input("Enter the value for k: "))


with open(file_train) as textFile:
    lines_train=[line.split("\t") for line in textFile]

data_train=np.asarray(lines_train)



with open(file_test) as textFile:
    lines_test=[line.split("\t") for line in textFile]

data_test=np.asarray(lines_test)


# get the training features and groudn truth
points_train= np.matrix(data_train[:,:-1],dtype=float,copy=False)

#print('points_train')
#print(points_train)
#print(len(points_train))

#getting ground truth
ground_truth_labels_train = np.asarray(data_train[:,-1],dtype=int)
train_label = ground_truth_labels_train

#print("Train labels")
#print(ground_truth_labels_train)
#print(len(ground_truth_labels_train))

# get the Testing  features and groudn truth of Test
points_test= np.matrix(data_test[:,:],dtype=float,copy=False)
test_data = points_test

#print("points_test")
#print(points_test)
#print(len(points_test))

'''
for col in range(points_train.shape[1]):

        v = points_train[:,col]
        u_test = points_test[:,col]
        #print(col)
        #print(points[:,col])
        points_train[:,col] = (v - v.min()) / (v.max() - v.min())

        points_test[:,col] = (u_test - v.min()) / (v.max() - v.min())

        #print(points[:,col])
'''


points_mean = np.mean(points_train , axis =0)
points_std = np.std(points_train , axis=0)
points_train = points_train - points_mean
points_train = points_train/points_std


train_data = points_train
#print("Normalised data Train")
#print(train_data)

points_test = points_test - points_mean
points_test = points_test/points_std

test_data=points_test
#print("Normalised data Test")
#print(test_data)

'''
ada  = AdaBoostRegressor(n_estimators =1000,loss='exponential',learning_rate=1)
ada.fit(train_data,train_label)
ada_predict = ada.predict(test_data)
'''
'''
ada  = AdaBoostRegressor(n_estimators =1000,loss='linear',learning_rate=0.8)
ada.fit(train_data,train_label)
ada_predict = ada.predict(test_data)
'''
'''
ada  = AdaBoostRegressor(n_estimators =1300,loss='square',learning_rate=1)
ada.fit(train_data,train_label)
ada_predict_1 = ada.predict(test_data)


ada  = AdaBoostRegressor(n_estimators =1300,loss='exponential',learning_rate=1)
ada.fit(train_data,train_label)
ada_predict_2 = ada.predict(test_data)


ada  = AdaBoostRegressor(n_estimators =1000,loss='linear',learning_rate=0.8)
ada.fit(train_data,train_label)
ada_predict_3 = ada.predict(test_data)
'''

ada  = AdaBoostRegressor(n_estimators =1300,loss='exponential',learning_rate=1)
ada.fit(train_data,train_label)
ada_predict_1 = ada.predict(test_data)

ada  = AdaBoostRegressor(n_estimators =1300,loss='square',learning_rate=1)
ada.fit(train_data,train_label)
ada_predict_2 = ada.predict(test_data)

ada  = AdaBoostRegressor(n_estimators =1300,loss='exponential',learning_rate=1)
ada.fit(train_data,train_label)
ada_predict_3 = ada.predict(test_data)

'''
res = []
for ele in ada_predict:
    if ele < 0.5:
        res.append(0)
    else:
        res.append(1)

ada_new = res
'''

res1 = []
for ele in ada_predict_1:
    if ele < 0.5:
        res1.append(0)
    else:
        res1.append(1)

ada_new_1 = res1

res2 = []
for ele in ada_predict_2:
    if ele < 0.5:
        res2.append(0)
    else:
        res2.append(1)

ada_new_2 = res2

res3 = []
for ele in ada_predict_3:
    if ele < 0.5:
        res3.append(0)
    else:
        res3.append(1)

ada_new_3 = res3





final_list=[]

for i in range(len(ada_predict_1)):
    count_of_zero = 0
    count_of_one = 0

    if(ada_new_1[i]==0):
        count_of_zero=count_of_zero+1
    else:
        count_of_one=count_of_one+1



    if(ada_new_2[i]==0):
        count_of_zero=count_of_zero+1
    else:
        count_of_one=count_of_one+1
    if(ada_new_3[i]==0):
        count_of_zero=count_of_zero+1
    else:
        count_of_one=count_of_one+1



    if(count_of_zero>count_of_one):
        final_list.append(0)
    else:
        final_list.append(1)



print("Final Predictions")
print(final_list)



id = 418

id_list=[]

for i in range(378):
    id_list.append(id)
    id=id+1

#print(id_list)



test_pred_csv = {'id': id_list ,
        'label': final_list
        }

df = DataFrame(test_pred_csv, columns= ['id', 'label'])

#print (df)

df.to_csv('ada_final_5.csv',index=False)



'''
wtr = csv.writer(open ('out_new.csv', 'w'), delimiter=',', lineterminator='\n')

#wtr.writerow("id")


#wtr.writerow("label")
for x in final_list : wtr.writerow ([x])
'''




