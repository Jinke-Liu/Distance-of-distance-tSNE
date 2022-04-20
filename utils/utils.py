#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:49:14 2021

@author: jinkel


utils needed for distance of distance tsne project 
"""

import numpy as np 
from sklearn import metrics
from scipy.spatial import distance as dist 
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd 


#%%

def dis_of_dis_transform(sim_distances, n_neighbors):
    '''
    Distance of distance transformation on the input distance matrix (TODO: optimize this)
    Input: distance matrix S
    Output: distance matrix S' after transformation using symmetric KNN methods 
    '''
    sorted_indices = np.argsort(sim_distances)
    ddistances = np.full(sim_distances.shape, np.nan)
    for i in range(sim_distances.shape[0]):
        for j in range(i):
            si = sorted_indices[i]
            sj = sorted_indices[j]
            inn = np.append(si[0:int(n_neighbors)], sj[0:int(n_neighbors)])
            ddistances[i, j] = np.nanmean(np.abs(sim_distances[i, inn]-sim_distances[j, inn]))
            ddistances[j, i] = ddistances[i, j]
    np.fill_diagonal(ddistances, 0)
    return ddistances


#%%

def simulate(n_cluster_points, n_cluster, n_noise_points, n_dims, n_neighbor, rnd_seed = 42, cluster_std = 0.2, fig_name='tsne.png'):
    '''
    simulate the data points from the mixture of Gaussians 
    '''
    #%%
    # generate clusters that follow isotropic Gaussian distributions  
    X_cluster, y_cluster = make_blobs(
        n_samples=n_cluster_points, centers=n_cluster, n_features=n_dims,
        random_state=rnd_seed, cluster_std=cluster_std, center_box=(-.5,+.5))
    X_cluster = X_cluster[np.argsort(y_cluster),:]
    y_cluster = y_cluster[np.argsort(y_cluster)]+1
    
    # generate noise points by drawing gaussian examples 
    X_noise, y_noise = make_blobs(
        n_samples=n_noise_points, centers=n_noise_points, n_features=n_dims,
        random_state=rnd_seed, cluster_std=cluster_std, center_box=(-.5,+.5))   
    y_noise = np.zeros(n_noise_points)    
    
    #%%
    # # generate infinitely dense cluster points 
    # X_cluster = np.random.rand(n_cluster, n_dims)
    # X_cluster = np.repeat(X_cluster, n_cluster_points//n_cluster , axis =0 )
    # # add a small fluctuation to the points so they are not exactly the same 
    # fluctuation = np.random.random((n_cluster_points, n_dims))
    # epsilon = 1e-5    
    # fluctuation = fluctuation * 2 * epsilon
    # fluctuation = fluctuation - epsilon
    # X_cluster += fluctuation
    # y_cluster = np.repeat(np.arange(n_cluster),n_cluster_points//n_cluster) + 1 
    
    # # generate noise points uniformly scattered in a unit hypercube
    # X_noise  = np.random.rand(n_noise_points, n_dims) 
    # y_noise = np.zeros(n_noise_points)
    
    #%% merge cluster and noise points 
    X = np.vstack((X_cluster,X_noise))
    y = np.concatenate((y_cluster,y_noise), axis=None)
    # restrict every point within the unit hypercube 
    
    #%%
    # compute the similarity matrix of L1 
#     sim_distances = dist.squareform(dist.pdist(X,'cityblock')) # L1
    sim_distances = dist.squareform(dist.pdist(X,'euclidean')) # L1
#     sim_distances /= n_dims # normalized by the dimensionality 
#     sim_distances = sim_distances / sim_distances.max() # normalization
    ddistances = dis_of_dis_transform(sim_distances,n_neighbor)
    
#     # # PLOT    
#     fig, axa = plt.subplots(2,2, figsize=(8,8))
#     axa[0][0].set_title('dissimilarity matrix');axa[0][0].set_xlabel('point $\it{i}$');axa[0][0].set_ylabel('point $\it{j}$')
#     im1 = axa[0][0].imshow(sim_distances, cmap='hot', interpolation='nearest', aspect='equal');
#     axa[0][1].set_title('dissimilarity matrix with disofdis');axa[0][1].set_xlabel('point $\it{i}$');axa[0][1].set_ylabel('point $\it{j}$')
#     im2 = axa[0][1].imshow(ddistances, cmap='hot', interpolation='nearest', aspect='equal');
#     plt.colorbar(im1,ax=axa[0][0],cax=make_axes_locatable(axa[0][0]).append_axes("right", size="5%", pad=0.05))
#     plt.colorbar(im2,ax=axa[0][1],cax=make_axes_locatable(axa[0][1]).append_axes("right", size="5%", pad=0.05))

#     # tsne 
#     embedded_distances = TSNE(n_components=2, metric='precomputed').fit_transform(sim_distances)
#     embedded_ddistances = TSNE(n_components=2, metric='precomputed').fit_transform(ddistances)
# #     axa[1][0].scatter(embedded_distances[:, 0], embedded_distances[:, 1], c=y ,marker='o',s = 5,facecolors='none') 
#     axa[1][0].scatter(embedded_distances[n_cluster_points:, 0], embedded_distances[n_cluster_points:, 1], c='grey' ,marker='o',s = 20,facecolors='none') 
#     axa[1][0].scatter(embedded_distances[:n_cluster_points, 0], embedded_distances[:n_cluster_points, 1], c=y_cluster ,marker='o',s = 20,facecolors='none') 

#     axa[1][0].set_title('tsne visualization')
# #     axa[1][1].scatter(embedded_ddistances[:, 0], embedded_ddistances[:, 1], c=y ,marker='o',s = 5,facecolors='none') 
#     axa[1][1].scatter(embedded_ddistances[n_cluster_points:, 0], embedded_ddistances[n_cluster_points:, 1], c='grey' ,marker='o',s = 20,facecolors='none') 
#     axa[1][1].scatter(embedded_ddistances[:n_cluster_points, 0], embedded_ddistances[:n_cluster_points, 1], c=y_cluster ,marker='o',s = 20,facecolors='none') 
#     axa[1][1].set_title('tsne visualization with disofdis')
#     plt.tight_layout();plt.savefig(fig_name)
    
    return sim_distances, ddistances, X, y


#%%
def ARI_score(embedded_distances, true_label):
    '''
    given the embedded_distances and the ture label
    kmeans clustering on the embeddings; return the ari clustering score
    '''
    
    # kmeans on the low-dim embeddings
    kmeans = KMeans(n_clusters=len(pd.unique(true_label)), random_state=42).fit(embedded_distances) 

    # # plot 
    # fig, axa = plt.subplots(1,2,figsize=(14,5))
    # sns.scatterplot(x=embedded_distances[:, 0],y= embedded_distances[:, 1], hue=true_label,legend='full',ax=axa[0])
    # axa[0].set_title('tsne visualization'); axa[0].legend()
    # sns.scatterplot(embedded_distances[:, 0], embedded_distances[:, 1], hue=kmeans.labels_, legend='full', ax=axa[1]) 
    # axa[1].set_title('kmeans clustering'); axa[1].legend()
    # plt.show()

    # evaluation of clustering 
    ari_score = metrics.adjusted_rand_score(true_label, kmeans.labels_)
    # ami_score = metrics.adjusted_mutual_info_score(true_label, kmeans.labels_)
    # print('ARI score: {0:.2f}'.format(ari_score))
    # print('AMI score: {0:.2f}'.format(ami_score))

    return ari_score

#%%
def plotKmeans(embedded_distances, true_label, axa):
    '''
    given the tsne embedding distances and the ture label
    perform kmeans clustering on the embeddings; return the axa (list of two ax)
    '''
    
    # kmeans on the low-dim embeddings
    kmeans = KMeans(n_clusters=len(pd.unique(true_label)), random_state=42).fit(embedded_distances) 
    # plot the embedding 
    axa[0].scatter(x=embedded_distances[:, 0],y= embedded_distances[:, 1], c=true_label)
    # sns.scatterplot(x=embedded_distances[:, 0],y= embedded_distances[:, 1], hue=true_label,legend='full',ax=axa[0])
    # axa[0].legend()
    axa[0].set_title('t-SNE');
    axa[0].set_xticklabels([]); axa[0].set_xticks([])
    axa[0].set_yticklabels([]); axa[0].set_yticks([])

    axa[1].scatter(x=embedded_distances[:, 0],y= embedded_distances[:, 1], c=kmeans.labels_)
    # sns.scatterplot(embedded_distances[:, 0], embedded_distances[:, 1], hue=kmeans.labels_, legend='full', ax=axa[1]) 
    # axa[1].legend()
    # evaluation of clustering 
    ari_score = metrics.adjusted_rand_score(true_label, kmeans.labels_)
    axa[1].set_title('K-means (ARI score: {0:.2f})'.format(ari_score));
    axa[1].set_xticklabels([]); axa[1].set_xticks([])
    axa[1].set_yticklabels([]); axa[1].set_yticks([])   
    return axa 
    
#%%
def mean_neighbor_distance(sim_distances, n_noise_points, n_neighbor):
    '''
    mean distance of a random point to its nth nearest neighbor
    '''
    return np.mean(np.array([np.sort(sim_distances[:,i])[n_neighbor] for i in range(n_noise_points)]))


#%%
def mean_distance_shrinkage(A,B,n_cluster_points,n_cluster,n_noise_points):
    '''
    the (normalized) average shrinkage of distance with disofdis trick
    Input: 
    A: distance matrix before the trick
    B: distance matrix after the trick
    Output:
    d1: average distance shrinkage of distance between a cluster point to a noise point 
    d2: average distance shrinkage of distance between two noise points 
    d3: average distance shrinkage of distance between two points in a same cluster 
    '''
    assert(n_cluster_points%n_cluster==0)
    cluster_size = n_cluster_points//n_cluster
        
    d1 = np.mean(A[:n_cluster_points,n_cluster_points:] - B[:n_cluster_points,n_cluster_points:])    
    d2 = np.sum(np.triu(A[n_cluster_points:,n_cluster_points:],1) - np.triu(B[n_cluster_points:,n_cluster_points:],1))/(n_noise_points*(n_noise_points-1)/2)
    d3 = np.mean(np.array([A[i*cluster_size:(i+1)*cluster_size,i*cluster_size:(i+1)*cluster_size][np.triu_indices(cluster_size,k=1)]-
                           B[i*cluster_size:(i+1)*cluster_size,i*cluster_size:(i+1)*cluster_size][np.triu_indices(cluster_size,k=1)] 
                           for i in np.arange(n_cluster,dtype=int)]).flatten())
    ind = ([],[])
    for i in np.arange(n_cluster-1,dtype=int):
        indi = np.array(np.meshgrid([i for i in range(i*cluster_size,(i+1)*cluster_size)],
                                    [i for i in range((i+1)*cluster_size,n_cluster_points)])).T.reshape(-1,2)
        ind[0].extend(indi[:,0])
        ind[1].extend(indi[:,1])
    d4 = np.mean(np.array(A[ind]-B[ind]).flatten())
    return d1, d2, d3, d4

#%%
def mean_distance_fraction(A,B,n_cluster_points,n_cluster,n_noise_points):
    '''
    the (normalized) average shrinkage of distance with disofdis trick as the fraction 
    Input: 
    A: distance matrix before the trick
    B: distance matrix after the trick
    Output:
    f1: average distance shrinkage of distance between a cluster point to a noise point 
    f2: average distance shrinkage of distance between two noise points 
    f3: average distance shrinkage of distance between two points in a same cluster 
    f4: average distance shrinkage of distance between points in two clusters 
    '''
    assert(n_cluster_points%n_cluster==0)
    # add a very small number to prevent divided by zero problem 
    A += 1e-17; B += 1e-17 
    cluster_size = n_cluster_points//n_cluster
    f1 = np.mean(B[:n_cluster_points,n_cluster_points:]/A[:n_cluster_points,n_cluster_points:])
    f2 = np.sum(B[n_cluster_points:,n_cluster_points:][np.triu_indices(n_noise_points,k=1)]/A[n_cluster_points:,n_cluster_points:][np.triu_indices(n_noise_points,k=1)])/(n_noise_points*(n_noise_points-1)/2)
    f3 = np.mean([np.mean(B[i*cluster_size:(i+1)*cluster_size,i*cluster_size:(i+1)*cluster_size][np.triu_indices(cluster_size,k=1)]/
                          A[i*cluster_size:(i+1)*cluster_size,i*cluster_size:(i+1)*cluster_size][np.triu_indices(cluster_size,k=1)]) for i in np.arange(n_cluster,dtype=int)])
    
    ind = ([],[])
    for i in np.arange(n_cluster-1,dtype=int):
        indi = np.array(np.meshgrid([i for i in range(i*cluster_size,(i+1)*cluster_size)],
                                    [i for i in range((i+1)*cluster_size,n_cluster_points)])).T.reshape(-1,2)
        ind[0].extend(indi[:,0])
        ind[1].extend(indi[:,1])
    f4 = np.mean(np.array(B[ind]/A[ind]).flatten())
    
    return f1, f2, f3, f4

