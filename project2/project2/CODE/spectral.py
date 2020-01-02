import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.cm as cm
from sklearn.metrics.pairwise import *
from sklearn.cluster import *
from scipy import linalg as LA


def RbfKernel(data1, data2, sigma):
    delta =np.matrix(abs(np.subtract(data1, data2)))
    squaredEuclidean = (np.square(delta).sum(axis=1))
    result = np.exp(-(squaredEuclidean)/(2*sigma**2))
    return result


file=input("Enter the filename: ")
with open(file) as textFile:
    lines=[line.split() for line in textFile]

cluster_count=int(input("Enter the number of Clusters: "))
sigma = float(input("Enter the sigma value "))
data=np.asarray(lines)
gene_id = data[:,0]
ground_truth = data[:,1]

#getting the attributes
points= np.matrix(data[:,2:],dtype=float,copy=False)
ndata = points.shape[0]

result = np.matrix(np.full((ndata , ndata) , 0 , dtype=float))

for i in range(ndata):
	for j in range(ndata):
		result[i ,j] = RbfKernel(points[i,:] , points[j , :] , sigma)


D = np.diag(result.sum(axis=1))
L = D-result

eigen_gap = 0
def transformToSpectral(laplacian):
    k = cluster_count
    e_vals, e_vecs = LA.eig((laplacian))
    ind = e_vals.real.argsort()[:k]
    result = np.ndarray(shape=(laplacian.shape[0],0))
    for i in range(1, ind.shape[0]):
        cor_e_vec = np.transpose(np.matrix(e_vecs[:,np.asscalar(ind[i])]))
        result = np.concatenate((result, cor_e_vec), axis=1)
    return result

def transformToSpectralEigen(laplacian):
    global eigen_gap
    e_vals, e_vecs = LA.eig((laplacian))
    sorted_eigen_values = np.sort(e_vals.real)[::-1]
    eigen_gap = np.argmax(np.diff(e_vals.real))
    k = eigen_gap+1
    ind = e_vals.real.argsort()[:k]
    result = np.ndarray(shape=(laplacian.shape[0],0))
    for i in range(1, ind.shape[0]):
        cor_e_vec = np.transpose(np.matrix(e_vecs[:,np.asscalar(ind[i])]))
        result = np.concatenate((result, cor_e_vec), axis=1)
    return result


# tr = transformToSpectral(L)
choice = input("do you want to enter the centroid points ? 1: Yes 2: No 3. Eigen Gap ")
if choice == '2':
    tr = transformToSpectral(L)
    kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(tr.real)
elif choice == '1':
    tr = transformToSpectral(L)
    print("Enter the points")
    centroid_points = list(map(int , input().split()))
    centroid_points = [ele-1 for ele in centroid_points]
    attributes = tr[centroid_points, :]
    kmeans = KMeans(n_clusters=cluster_count, init = attributes).fit(tr.real)
else:
    tr = transformToSpectralEigen(L)
    print("number of clusters = {}".format(eigen_gap+1))
    kmeans = KMeans(n_clusters=eigen_gap+1, random_state=0).fit(tr.real)

cluster_assignment = kmeans.labels_

cluster_assignment=np.asarray(cluster_assignment,dtype=int)
# print(cluster_assignment)

# Plotting using PCA

pca_plot_matrix = PCA(n_components=2).fit_transform(points)
plot_unique_labels = list(set(cluster_assignment))
unique_naming_list_1=[]

colours_unique_vector = cm.Set1(np.linspace(0, 1, len(plot_unique_labels)))

for i in range(len(plot_unique_labels)):
    dis_rows_index = np.where(cluster_assignment==plot_unique_labels[i])
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
plt.title("Spectral using PCA to visualize "+file,fontweight="bold")
plt.show()


def calculateJacRand(ground_truth,cluster_assignment):
    #compute jaccard coefficient and rand index

    true_pos = 0
    true_neg = 0
    false_pos=0
    false_neg=0
    for i in range(len(data)):
        for j in range(len(data)):
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

