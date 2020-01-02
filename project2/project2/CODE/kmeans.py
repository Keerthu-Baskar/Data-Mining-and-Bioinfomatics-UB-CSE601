import sys
import numpy as np
from random import randint
from scipy.spatial import distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict



file_name = input("Enter the name of the file: ")
file = open(file_name, "r")
lines = file.readlines()



data_matrix = []
centroids = []

for line in lines:
    data = line.strip().split("\t")
    data_matrix.append(data)

data_array = np.asarray(data_matrix, dtype = float)
gene_id = data_array[:,0]
ground_truth = data_array[:,1]
attributes = np.delete(data_array,np.s_[0:2],axis = 1)




k = int(input("Enter the cluster value: "))
print("cluster is set to be: "+str(k))

rows = attributes.shape[0]
cols = attributes.shape[1]

choice = input("Do you want to enter centroids 1.Yes 2.No :")
reverse_index = defaultdict(int)
if choice == '1':
    for i in range(k):
        index = int(input("Enter the "+str(i+1)+" ID: "))
        centroids.append(attributes[index-1])
else:
    selected_centroids = np.random.choice(rows, k, replace=False)
    centroids = attributes[selected_centroids, :]

no_of_iterations = int(input("Enter the max number of iterations: "))




for i in range(no_of_iterations):
    clusters = defaultdict(list)
    cluster_temp = []
    for j in range(rows):
        current_ans = float('inf')
        centroid_choice = None
        centroid_dist = []
        for l in range(len(centroids)):
            current_distance = distance.euclidean(attributes[j],centroids[l])
            if current_distance < current_ans:
                current_ans = current_distance
                centroid_choice = l
        clusters[centroid_choice].append(j)
    centroids_new =[]
    for l in range(len(centroids)):
        relevant_attributes = attributes[clusters[l],:]
        if len(relevant_attributes) == 0:
            centroids_new.append(centroids[l])
        else:
            centroids_new.append(np.mean(relevant_attributes , axis= 0))
    if np.array_equal(centroids , centroids_new):
        break
    centroids = centroids_new

reverse_mapping = [0]*(len(data_array))
for key , v in clusters.items():
    for elements in v:
        reverse_mapping[elements] = key+1

cluster_assignment = [ele for ele in reverse_mapping]

cluster_assignment=np.asarray(cluster_assignment,dtype=int)

#plotting using PCA
pca_plot_matrix = PCA(n_components=2).fit_transform(attributes)
plot_unique_labels = list(set(cluster_assignment))
unique_naming_list_1=[]

colours_unique_vector = cm.Set1(np.linspace(0, 1, len(plot_unique_labels)))

for i in range(len(plot_unique_labels)):
    dis_rows_index = np.where(cluster_assignment==plot_unique_labels[i])
    dis_rows = pca_plot_matrix[dis_rows_index]
    x_plot =[dis_rows[:,0]]
    y_plot = [dis_rows[:,1]]
    unique_naming_list_1.append(plt.scatter(x_plot, y_plot, c=colours_unique_vector[i]))

plot_unique_labels=[-1.0 if x==0 else x for x in plot_unique_labels]
plot_unique_labels=np.array(plot_unique_labels,dtype=int)

plt.legend(unique_naming_list_1,plot_unique_labels,loc="best",ncol=1,markerfirst=True,shadow=True)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("K means clustering using PCA for visualisation "+file_name,fontweight="bold")
plt.show()


def calculateJacRand(ground_truth,cluster_assignment):

    true_pos = 0
    true_neg = 0
    false_pos=0
    false_neg=0
    for i in range(len(data_array)):
        for j in range(len(data_array)):
            if ground_truth[i]==ground_truth[j]:
                if cluster_assignment[i]==cluster_assignment[j]:
                    true_pos=true_pos+1
                else:
                    false_neg=false_neg+1
            elif ground_truth[i]!=ground_truth[j]:
                if cluster_assignment[i]==cluster_assignment[j]:
                    false_pos=false_pos+1
                else:
                    true_neg=true_neg+1
    jaccard_value=(true_pos)/(true_pos+false_pos+false_neg)
    rand_index_value=(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    return jaccard_value,rand_index_value

jaccard_value,rand_index_value=calculateJacRand(ground_truth,cluster_assignment)

print("Jaccard Coefficient = ",jaccard_value)
print("Rand Index = ",rand_index_value)