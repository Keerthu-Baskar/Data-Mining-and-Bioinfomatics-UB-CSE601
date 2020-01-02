__author__ = 'Sriram Wall'

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.cm as cm

file=input("Enter the filename: ")
with open(file) as textFile:
    lines=[line.split() for line in textFile]

cluster_count=int(input("Enter the number of Clusters: "))

data=np.asarray(lines)

#getting the attributes
points= np.matrix(data[:,2:],dtype=float,copy=False)

#print(len(points))
#getting ground truth
ground_truth = data[:,1]


#finding the distance of every point from a given point
#dist_mat = np.linalg.norm(points - points[:,None], axis=-1)
# Compute distance matrix from the attributes
dist_matrix=distance_matrix(points,points)

total_points=len(points)

check = total_points-cluster_count


cluster_label = [[x] for x in range(total_points)]

#print(cluster_label)

#given an element in cluster, return the index of that cluster eg [[5,1,2,7],[6],] index : 0
def find_cluster(cluster_list,point):

    for index,cluster in enumerate(cluster_list):

        if point in cluster:

            return index


def update_cluster():
    for i in range(check):
        min_value=np.min(dist_matrix[np.nonzero(dist_matrix)])
        coord=np.where(dist_matrix == min_value)

        x_cord=coord[0][0]
        y_cord=coord[1][0]

        x_cluster = find_cluster(cluster_label,x_cord)
        y_cluster= find_cluster(cluster_label,y_cord)

        cluster_label[x_cluster]=cluster_label[x_cluster] + cluster_label[y_cluster]
        cluster_label.pop(y_cluster)


        for point in range(total_points):

            dist_matrix[x_cord][point] = min(dist_matrix[x_cord][point],dist_matrix[y_cord][point])
            dist_matrix[point][x_cord] = min(dist_matrix[point][x_cord],dist_matrix[point][y_cord])

            dist_matrix[y_cord][point] = np.inf
            dist_matrix[point][y_cord] = np.inf

            dist_matrix[point][point] =0




    final_point_cluster_name =np.zeros(len(points),dtype=int)
    cluster_name=1;

    for cluster in cluster_label:
        for point in cluster:
            final_point_cluster_name[point]=cluster_name
        cluster_name=cluster_name+1

    return final_point_cluster_name

final_point_cluster_name = update_cluster()
#print("Final cluster")
#print(final_point_cluster_name)

#Plotting using PCA

pca_plot_matrix = PCA(n_components=2).fit_transform(points)
plot_unique_labels = list(set(final_point_cluster_name))
unique_naming_list_1=[]

colours_unique_vector = cm.Set1(np.linspace(0, 1, len(plot_unique_labels)))

for i in range(len(plot_unique_labels)):
    dis_rows_index = np.where(final_point_cluster_name==plot_unique_labels[i])
    dis_rows = pca_plot_matrix[dis_rows_index]
    x_plot =[dis_rows[:,0]]
    y_plot = [dis_rows[:,1]]
    unique_naming_list_1.append(plt.scatter(x_plot, y_plot, c=colours_unique_vector[i]))

        #plt.scatter(x_plot,y_plot,c=colours_unique_vector[i])
plot_unique_labels=[-1.0 if x==0 else x for x in plot_unique_labels]
plot_unique_labels=np.array(plot_unique_labels,dtype=int)

plt.legend(unique_naming_list_1,plot_unique_labels,loc="best",ncol=1,markerfirst=True,shadow=True)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Hierarchical Agglomerative clustering using PCA for visualisation "+file,fontweight="bold")
plt.show()


#calculating J and R coeff

#print("length")
#print(len(points))
#getting ground truth
ground_truth = data[:,1]




def calculateJacRand(points,ground_truth,final_point_cluster_name):
    #compute jaccard coefficient and rand index

    true_pos = 0
    true_neg = 0
    false_pos=0
    false_neg=0
    for i in range(len(data)):
        for j in range(len(data)):
            if ground_truth[i]==ground_truth[j]:
                if final_point_cluster_name[i]==final_point_cluster_name[j]:
                    true_pos=true_pos+1
                else:
                    false_neg=false_neg+1
            elif ground_truth[i]!=ground_truth[j]:
                if final_point_cluster_name[i]==final_point_cluster_name[j]:
                    false_pos=false_pos+1
                else:
                    true_neg=true_neg+1
    jaccard_value=(true_pos)/(true_pos+false_pos+false_neg)
    rand_index_value=(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    return jaccard_value,rand_index_value

jaccard_value,rand_index_value=calculateJacRand(points,ground_truth,final_point_cluster_name)

print("Jaccard Coefficient = ",jaccard_value)
print("Rand Index = ",rand_index_value)
