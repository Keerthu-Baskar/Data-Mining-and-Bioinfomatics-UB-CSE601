import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class GMM:
    def __init__(self, k, max_iter=15 , smoothing_value = 0.0000001):
        self.k = k
        self.max_iter = int(max_iter)
        self.smoothing_value = smoothing_value

    def fit(self, X  , mu_given = None , sigma_given = None, phi_given = None , conv_threshold = 0.00000001):
        self.shape = X.shape
        self.n, self.m = self.shape

        if phi_given:
            self.phi = [np.asarray(ele , dtype = float) for ele in phi_given]
        else:
            self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.cluster_prob = np.full( shape=self.shape, fill_value=1/self.k)
        if mu_given:
            self.mu = [np.asarray(ele , dtype = float) for ele in mu_given]
        else:
            row_choice = np.random.randint(low=0, high=self.n, size=self.k)
            self.mu = [  X[row_index,:] for row_index in row_choice ]
        if sigma_given:
            self.sigma = [np.asarray(ele , dtype = float) for ele in sigma_given]
        else:
            self.sigma = [ np.cov(X.T) for _ in range(self.k) ]
        # to avoid singular matrix error
        for i , ele in enumerate(self.sigma):
            np.fill_diagonal(self.sigma[i], ele.diagonal() + self.smoothing_value)
        old_loss = None
        for iteration in range(self.max_iter):
            # E step
            self.cluster_prob = self.probability_prediction(X)
            self.phi = self.cluster_prob.mean(axis=0)

            # M step
            for i in range(self.k):
                weight = self.cluster_prob[:, [i]]
                total_weight = weight.sum()
                self.mu[i] = (X * weight).sum(axis=0) / total_weight
                self.sigma[i] = np.cov(X.T, 
                    aweights=(weight/total_weight).flatten(), 
                    bias=True)
                # to avoid singular matrix error
                for i , ele in enumerate(self.sigma):
                    np.fill_diagonal(self.sigma[i], ele.diagonal() + self.smoothing_value)
            new_loss = self.loss_function(X)
            if old_loss != None and abs(new_loss - old_loss) <= conv_threshold:
                # print("hurray" , iteration)
                break
            old_loss = new_loss
            # print(self.loss_function(X))
            
    def probability_prediction(self, X):
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        cluster_prob = numerator / denominator
        return cluster_prob
    
    def predict(self, X):
        cluster_prob = self.probability_prediction(X)
        return np.argmax(cluster_prob, axis=1)

    def loss_function(self, X):
        N = X.shape[0]
        C = self.cluster_prob.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = multivariate_normal(self.mu[c], self.sigma[c],allow_singular=True)
            self.loss[:,c] = self.cluster_prob[:,c] * (np.log(self.phi[c]+0.00000001)+dist.logpdf(X)-np.log(self.cluster_prob[:,c]+0.000000001))
        self.loss = np.sum(self.loss)
        return self.loss


file_name = input("Enter the name of the file: ")
file = open(file_name, "r")
lines = file.readlines()
clusters = int(input("Enter the the number of clusters: "))
iterations = int(input("Enter the number of iterations: "))
data_matrix = []
centroids = []

# splittng the line into individual data
for line in lines:
    data = line.strip().split("\t")
    data_matrix.append(data)

#Converting it into array and getting the gene id and ground truth
data_array = np.asarray(data_matrix, dtype = float)
gene_id = data_array[:,0]
ground_truth = data_array[:,1]
attributes = np.delete(data_array,np.s_[0:2],axis = 1)

gmm = GMM(k=clusters, max_iter=iterations)
gmm.fit(attributes)
# gmm.fit(attributes , [[0.5 , 0.7] , [1.4,1.8]] , [[[3.05318508, 1.34736627],[1.34736627, 3.78984268]],[[3.05318508, 1.34736627],[1.34736627, 3.78984268]]] , [0.5 , 0.5])
# gmm.fit(attributes , [[0 , 0] , [1 , 1]] , [[[1 , 1],[1 , 1]],[[2 , 2],[2 , 2]]] , [0.5 , 0.5])

cluster_assignment = gmm.predict(attributes)


cluster_assignment=np.asarray(cluster_assignment,dtype=int)
print(cluster_assignment)

#Plotting using PCA

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

        #plt.scatter(x_plot,y_plot,c=colours_unique_vector[i])
plot_unique_labels=[-1.0 if x==0 else x for x in plot_unique_labels]
plot_unique_labels=np.array(plot_unique_labels,dtype=int)

plt.legend(unique_naming_list_1,plot_unique_labels,loc="best",ncol=1,markerfirst=True,shadow=True)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("GMM using PCA to visualize "+file_name,fontweight="bold")
plt.show()


def calculateJacRand(ground_truth,cluster_assignment):
    #compute jaccard coefficient and rand index

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

# print(cluster_assignment)