__author__ = 'Sriram Wall'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD



def pca_svd_tnse_plot(file):
    with open(file) as textFile:
        #lines = file.readlines()
        #lines=file.strip().replace(" ","-").split("\t")
        lines=[line.split("\t") for line in textFile]


#    data = np.genfromtxt('pca_c.v txt')

    #IMPLEMENTING PCA ON OUR OWN

    data=np.asarray(lines[:][:-1])
    # print (data.size())
    #getting attributes except last column
    attr = np.matrix(data[:,:-1],dtype=float)
    #attr = data[:,0:data.shape[1]-1] #all columns except the last is taken as attributes
    disease_col =data[:,-1]
    #implementing pca
    #mean centered step is done
    attr_mean_colwise = attr.mean(0)
    standardized_attr = attr - attr_mean_colwise

    #calculate cov matrix
    attr_cov = np.cov(standardized_attr,rowvar=False)
    #calculation eigen value and eigen vector to obtain max variance
    eigan_values, eigan_vectors = np.linalg.eig(attr_cov)

    #idx = np.argsort(eigan_values)[::-1]
    idx_new=np.argsort(eigan_values)[::-1][:2]
    eigan_vectors=eigan_vectors[:,idx_new]
    #obtaining the dot product of adjusted matrix with eigen vector to plot it
    pca_plot_matrix = np.dot(standardized_attr,eigan_vectors)


    #plotting pca
    unique_naming_list_1=[]
    disease_unique = list(set(disease_col))
    colours_unique_vector = cm.Set1(np.linspace(0, 1, len(disease_unique)))

    for i in range(len(disease_unique)):
        dis_rows_index = np.where(disease_col==disease_unique[i])
        dis_rows = pca_plot_matrix[dis_rows_index]
        x_plot =[dis_rows[:,0]]
        y_plot = [dis_rows[:,1]]
        unique_naming_list_1.append(plt.scatter(x_plot, y_plot, c=colours_unique_vector[i]))

        #plt.scatter(x_plot,y_plot,c=colours_unique_vector[i])
    plt.legend(unique_naming_list_1,disease_unique,loc="best",ncol=1,markerfirst=True,shadow=True)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Principal Component Analysis "+file,fontweight="bold")
    plt.show()

    #IMPLEMENTING SVD

    #svd implementation
    unique_naming_list_3 =[]

    #U, S, vh = np.linalg.svd(attr, full_matrices=True)
    #SVD_PLOT = u[:,[0,1]]

    U_SVD = TruncatedSVD(n_components=2,n_iter=5,tol=0.0).fit_transform(attr)

    for i in range(len(disease_unique)):
        dis_rows_index = np.where(disease_col==disease_unique[i])
        dis_rows = U_SVD[dis_rows_index]
        #dis_rows = SVD_PLOT[dis_rows_index]
        x_plot =[dis_rows[:,0]]
        y_plot = [dis_rows[:,1]]
        unique_naming_list_3.append(plt.scatter(x_plot, y_plot, c=colours_unique_vector[i]))

        #plt.scatter(x_plot,y_plot,c=colours_unique_vector[i])

    plt.legend(unique_naming_list_3,disease_unique,loc="best",ncol=1,markerfirst=True,shadow=True)

    plt.xlabel("Comp 1")
    plt.ylabel("Comp 2")
    plt.title("SVD: "+file,fontweight="bold")

    plt.show()
    #print(disease_unique)

    #IMPLEMENTING TNSE

    unique_naming_list_2=[]

    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
                init='pca', learning_rate=100,n_iter=1000)
    final_tsne=tsne.fit_transform(attr)

    for i in range(len(disease_unique)):
        dis_rows_index = np.where(disease_col==disease_unique[i])
        dis_rows = final_tsne[dis_rows_index]
        x_plot =[dis_rows[:,0]]
        y_plot = [dis_rows[:,1]]
        unique_naming_list_2.append(plt.scatter(x_plot, y_plot, c=colours_unique_vector[i]))

        #plt.scatter(x_plot,y_plot,c=colours_unique_vector[i])

    plt.legend(unique_naming_list_2,disease_unique,loc="best",ncol=1,markerfirst=True,shadow=True)
    plt.xlabel("")
    plt.ylabel("")
    plt.title("TNSE: "+file,fontweight="bold")
    plt.show()


#print(pca_plot_matrix)
# print(attr_cov)
#print(idx_new)
#print(eigan_vectors)

#    print(attr)
 #   print(disease_col)
  #  print(attr_mean_colwise)

file=input("Enter the filename: ")


#pca_svd_tnse_plot("pca_a.txt")

#pca_svd_tnse_plot("pca_b.txt")

#pca_svd_tnse_plot("pca_c.txt")

pca_svd_tnse_plot(file)
