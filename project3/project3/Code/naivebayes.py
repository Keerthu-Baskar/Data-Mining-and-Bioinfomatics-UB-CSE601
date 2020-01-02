__author__ = 'Sriram Wall'
import numpy as np
import math
from scipy.stats import norm



#def naive_bayes(train_records_cat,test_categoric_data):

categ_data_cols=[]

file=input("Enter the filename: ")


with open(file) as textFile:
    lines=[line.split("\t") for line in textFile]

data=np.asarray(lines)

sample_data = data[0]



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



for col in range(points.shape[1]):
    if col not in categ_data_cols:
        v = points[:,col]
        #print("col")
        # print(col)
        #print(points[:,col])
        points[:,col] = (v - v.min()) / (v.max() - v.min())
        #print(points[:,col])
#print(points)
#getting ground truth
ground_truth_labels = np.asarray(data[:,-1],dtype=int)


#print(ground_truth_labels)

#splitting pounts into 10 groups
split_points = np.array_split(points,10)
# doing the same for labels
ground_truth_labels_split = np.array_split(ground_truth_labels,10)
#print("SPlit")
#print(ground_truth_labels_split)

#for calculating the metrics
acc_average=0
precision_average=0
recall_average=0
f1_Measure_average = 0


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





for index in range(10):
    #9 groups for training data are put into one group

    train_data=np.asarray(np.vstack([x for i,x in enumerate(split_points) if i != index]))
    train_label=np.asarray(np.concatenate([x for i,x in enumerate(ground_truth_labels_split) if i != index]))
    #print(len(train_label))
    test_data=np.asarray(split_points[index])
    test_label=np.asarray(ground_truth_labels_split[index])

    train_cont= np.delete(train_data,categ_data_cols,axis=1)

    test_cont = np.delete(test_data,categ_data_cols,axis=1)



    #GET THE NUMBER OF 0'S AND 1'S DATA SEP

    train_zero_records = train_cont[train_label == 0.0]

    train_one_records = train_cont[train_label == 1.0]


    #to implement gaussian formula
    mean = np.mean(train_zero_records,axis=0)
    std_deviation = np.std(train_zero_records, axis=0)

    cont_zero_result=norm.pdf(test_cont,mean,std_deviation)

    mean_for_one = np.mean(train_one_records,axis=0)
    std_for_one = np.std(train_one_records, axis=0)

    cont_one_result =norm.pdf(test_cont,mean_for_one,std_for_one)

    cont_one_result_convert=list()
    cont_zero_result_convert=list()

    for i in range(len(test_data)):
        cont_zero_result_convert.append(np.prod(cont_zero_result[i]))
        cont_one_result_convert.append(np.prod(cont_one_result[i]))



    #cont_prob_for_zero = gaussdist(train_zero_records,test_cont)

    #print(cont_prob_for_zero)
   # print(len(cont_zero_result))
    #print(len(cont_one_result))

    #


    zero_cat_list =  np.ones((len(test_data)))
    one_cat_list = np.ones((len(test_data)))

    for index_cat in range(len(categ_data_cols)):

        train_categoric_data = train_data[:,categ_data_cols[index_cat]]
        test_categoric_data = test_data[:,categ_data_cols[index_cat]]
        #(test_categoric_data)

        train_zero_records_cat = train_categoric_data[train_label == 0.0]
        train_one_records_cat = train_categoric_data[train_label == 1.0]

        #print(train_zero_records_cat)

        prob_of_zero = len(train_zero_records_cat)/len(train_categoric_data)

        prob_of_one = len(train_one_records_cat)/len(train_categoric_data)



        #()
        for ind in range(len(test_categoric_data)):

            # for zero finding three terms, one already found above
            count_of_x = len(train_categoric_data[train_categoric_data==test_categoric_data[ind]])

            #same for both 0 and 1
            denom_p_of_x =  count_of_x/len(train_categoric_data)

            #finding number of absent where output label is zero
            #print(np.where((train_zero_records_cat ==test_categoric_data[ind])))



            prob_of_x_given_zero = len(train_zero_records_cat[np.where((train_zero_records_cat ==test_categoric_data[ind]))])/len(train_zero_records_cat)

            #for one finding three terms
            prob_of_x_given_one = len(train_one_records_cat[np.where((train_one_records_cat ==test_categoric_data[ind]))])/len(train_one_records_cat)


            #ft_array,indices = np.unique(train_categoric_data,return_inverse = True)

            #lenofcat = len(indices)

            lenofcat = len(set(train_categoric_data))




            #lenofcat = lenofcat.size
            if prob_of_x_given_zero == 0 or prob_of_x_given_zero == 1:
                prob_of_x_given_zero = (len(train_zero_records_cat[np.where((train_zero_records_cat ==test_categoric_data[ind]))]  ) + 1) /(len(train_zero_records_cat) + lenofcat)


            if prob_of_x_given_one == 0 or prob_of_x_given_one == 1:
                prob_of_x_given_one = (len(train_one_records_cat[np.where((train_one_records_cat ==test_categoric_data[ind]))]) + 1) /(len(train_one_records_cat) + lenofcat)

            prob_of_zero_geiven_x = (prob_of_x_given_zero * prob_of_zero)  / denom_p_of_x

            prob_of_one_geiven_x = (prob_of_x_given_one * prob_of_one) / denom_p_of_x

            zero_cat_list[ind]=zero_cat_list[ind]*(prob_of_zero_geiven_x)

            one_cat_list[ind]=one_cat_list[ind]*(prob_of_one_geiven_x)

    #now multiply zero and one prob of continuous and cat data


    test_final_predict=[]

    for i in range(len(test_data)):

        final_prob_zero = zero_cat_list[i]*cont_zero_result_convert[i]
        final_prob_one= one_cat_list[i]*cont_one_result_convert[i]
        #print("cat")
        #print(zero_cat_list[i])
        #print("cont")
        #print(cont_zero_result_convert[i])
        if final_prob_zero > final_prob_one:
            test_final_predict.append(0)
        else:
            test_final_predict.append(1)

    #print("Final test predict labels")

    #print(len(test_final_predict))

    print(test_final_predict)

    #determint the metrics by calling the function
    acc, prec, recall, f1_measure = performance_metric(test_label, test_final_predict)

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


