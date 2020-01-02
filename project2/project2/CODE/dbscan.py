__author__ = 'Sriram Wall'

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.cm as cm

import matplotlib.cm as CM

#getting file
file=input("Enter the filename: ")
with open(file) as textFile:
    lines=[line.split() for line in textFile]

data=np.asarray(lines)
# Input eps value
epsilon = float(input("Enter the eps value: "))
# Input minimum points
min_points=int(input("Enter the minimum points: "))
#getting atte
points= np.matrix(data[:,2:],dtype=float,copy=False)

#getting ground truth
ground_truth = data[:,1]

#finding the distance of every point from a given point
#dist_mat = np.linalg.norm(points - points[:,None], axis=-1)
# Compute distance matrix from the attributes
dist_matrix=distance_matrix(points,points)

#final results
cluster_final =np.zeros(len(points),dtype=int)

#visited array
#visited_points=np.full(len(points), False, dtype=bool)

visited_points=np.zeros(len(points),dtype=bool)

def regionQuery(query_point_index,points,epsilon,dist_matrix):
    neighbour_list = []

    for i in range(len(points)):
        if dist_matrix[query_point_index][i] <= epsilon:
            neighbour_list.append(i)

    return neighbour_list


def expandCluster(corepoint_index,points, neighbour_pts,cluster,epsilon, min_points,cluster_final,visited_points,dist_matrix):
     i=0
     #print("com expand cluster")
     while i < len(neighbour_pts):
         #print(i)
         
         if(not visited_points[neighbour_pts[i]]):
             visited_points[neighbour_pts[i]]=True
             new_neighbours = regionQuery(neighbour_pts[i],points,epsilon,dist_matrix)
             if(len(new_neighbours) >= min_points):
                 neighbour_pts=neighbour_pts+new_neighbours
             if(cluster_final[neighbour_pts[i]]==0):
                 cluster_final[neighbour_pts[i]]=cluster;
         i=i+1


def dbscan(points,epsilon,min_points,visited_points,cluster_final,dist_matrix):
    cluster=0

    for i in range(len(points)):

        if(not visited_points[i]):
            visited_points[i]=True
            neighbour_pts = regionQuery(i,points,epsilon,dist_matrix)
            if(len(neighbour_pts)<min_points):
                cluster_final[i]=0
            else:
                cluster=cluster+1;
                cluster_final[i]=cluster;
                expandCluster(i,points, neighbour_pts,cluster,epsilon, min_points,cluster_final,visited_points,dist_matrix)


dbscan(points,epsilon,min_points,visited_points,cluster_final,dist_matrix)


#print(cluster_final)

#Plotting using PCA

pca_plot_matrix = PCA(n_components=2).fit_transform(points)
plot_unique_labels = list(set(cluster_final))
unique_naming_list_1=[]

colours_unique_vector = cm.Set1(np.linspace(0, 1, len(plot_unique_labels)))

for i in range(len(plot_unique_labels)):
    dis_rows_index = np.where(cluster_final==plot_unique_labels[i])
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
plt.title("DBSCAN using PCA to visualize "+file,fontweight="bold")
plt.show()



#print(plot_unique_labels)


def calculateJacRand(points,ground_truth,cluster_final):
    #compute jaccard coefficient and rand index

    true_pos = 0
    true_neg = 0
    false_pos=0
    false_neg=0
    for i in range(len(data)):
        for j in range(len(data)):
            if ground_truth[i]==ground_truth[j]:
                if cluster_final[i]==cluster_final[j]:
                    true_pos=true_pos+1
                else:
                    false_neg=false_neg+1
            elif ground_truth[i]!=ground_truth[j]:
                if cluster_final[i]==cluster_final[j]:
                    false_pos=false_pos+1
                else:
                    true_neg=true_neg+1
    jaccard_value=(true_pos)/(true_pos+false_pos+false_neg)
    rand_index_value=(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    return jaccard_value,rand_index_value

jaccard_value,rand_index_value=calculateJacRand(points,ground_truth,cluster_final)


print("Jaccard Coefficient = ",jaccard_value)
print("Rand Index = ",rand_index_value)








