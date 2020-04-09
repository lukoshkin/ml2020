# Author: Maksim Velikanov
import numpy as np

import torch
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from sklearn.neighbors import KNeighborsTransformer

def Takens_Embedding_torch(data, d=5, delay=10):
    # In a collection of time series, transforms(Takens Embedding) each time series into a point cloud
    # for further use in TDA methods.
    
    # data - torch tensor of the shape (n_samples, L, internal_dim), where L is the length of time series
    # internal_dim - dimenssion of time series data(equal to 1 for univariate time series)
    # d - integer, determines dimension of points in point cloud: d * internal_dim. 
    # delay - integer, time delay.
    
    # Returns point_clouds - pytorch tensor of the shape (n_samples, n_points, d * internal_dim)
    # Number of points in each cloud n_points = L - delay * (d - 1)
    
    n_samples, L, internal_dim = data.size()
    point_clouds = torch.cat(
        tuple([torch.cat((data[:, (i*delay):, :], torch.zeros(n_samples, i*delay, internal_dim)), dim=1)
                      for i in range(d)]), dim=2)
    n_points = L - delay * (d-1)
    point_clouds = point_clouds[:, :n_points, :]
    
    return point_clouds

def get_diagrams_torch(point_clouds, maxdim = 1):
    # Calculates persistence diagrams from point clouds. 
    # Complexity of calculation increase with the maximum homology dimension, taken into account
    
    # point_clouds -  pytorch tensor of the shape (n_samples, n_points, dim)
    # maxdim - maximum homology dimension 
    
    # Returns tuple (diagrams_torch, diagrams_np, VR_persistence)
    #diagrams_torch - pytorch tensors of the shape (n_samples, maxdim + 1, n_features, 2)
    # n_features - maximum number of topological features, across different samples.
    # The last axis has the structure [birth_scale, death_scale]
    # The last two elements in tuple are needed for plotting diagrams only
    
    homology_dimensions = tuple(range(maxdim + 1))
    VR_persistence = VietorisRipsPersistence(homology_dimensions = homology_dimensions)
    point_clouds_np = point_clouds.numpy()
    diagrams_np = VR_persistence.fit_transform(point_clouds_np)
    homology_dimensions = diagrams_np[:, :, 2, np.newaxis]
    diagrams_torch = []
    for i in range(maxdim + 1):
        diagrams_fixed_Hdim = np.select([homology_dimensions == i], [diagrams_np[:, :, :2]])
        diagrams_torch.append(torch.FloatTensor(diagrams_fixed_Hdim[:, np.newaxis, :, :]))
    diagrams_torch = torch.cat(tuple(diagrams_torch), dim=1)
    return diagrams_torch, diagrams_np, VR_persistence

def _relu(x, a=0.):
    return torch.max(x-a, torch.full_like(x, 0))

def get_landscapes_torch(diagrams, n_layers=10, n_steps=100):
    
    # diagrams - torch tensor of the shape(n_samples, maxdim + 1, n_features, 2)
    # returns:
    # landscapes_final - torch tensor of the shape(n_samples, maxdim + 1, n_layers, n_steps)
    # max_scale - maximum scale value at which at least one persistent landscapes is non-zero
    
    n_samples, n_hdim, n_features, temp = diagrams.size() 
    scale_min = 0
    scale_max = torch.max(diagrams[:, :, :, 1])
    lin_plus = torch.linspace(scale_min, scale_max, n_steps)
    lin_plus = lin_plus[None, None, None, :].expand(n_samples, n_hdim, n_features, -1)
    relu_plus = _relu(lin_plus - diagrams[:, :, :, 0, None])
    relu_minus = _relu(-lin_plus + diagrams[:, :, :, 1, None])
    landscapes_individual = torch.min(relu_plus, relu_minus)
    landscapes_final, indices = torch.sort(landscapes_individual, dim=2, descending=True)
    landscapes_final = landscapes_final[:, :, :n_layers, :]
    return landscapes_final, scale_max

def landscapes_distances(landscapes, max_scale, n_layers=1):
    
    # landscapes - torch tensor of the shape(n_samples, maxdim + 1, n_layers, n_steps)
    # returns pairwise_distances - torch tensor of the shape(n_samples, n_samples),
    # containing pairwise distances(L2 persistent landscape distance) between samples in dataset 
    landscapes_reduced = landscapes[:, :, :n_layers, :]
    n_samples, n_hdim, n_layers, n_steps = landscapes_reduced.size()
    landscapes_1 = landscapes_reduced[:, None, :, :, :]
    landscapes_2 = landscapes_reduced[None, :, :, :, :]
    pairwise_distances = torch.sum((landscapes_1 - landscapes_2) ** 2, [2,3,4])
    pairwise_distances = (max_scale / n_steps * pairwise_distances) ** 0.5
    return pairwise_distances
    
def get_kNN_score_torch(pairwise_distances, matching_matrix, n_neighbours=5):

    # The score shows how the collection of persistent landscapes, corresponding to each label,
    # are separeted from each other, in the sense of L2 distance in the Hilbert space of persistent landscape
    
    # pairwise_distances - torch tensor of the shape (n_samples, n_samples). 
    # mathcing_matrix - numpy array of the shape (n_samples, n_samples). 1 if samples have the same label, 0 otherwise
    # n_neighbours - integer, number of nearest used to calculate the score
    
    # returns kNN_score - real number between 0 and 1
    
    n_samples = pairwise_distances.size()[0]
    kNN_transformer = KNeighborsTransformer(mode='connectivity', metric='precomputed', n_neighbors=n_neighbours)
    connectivity_matrix = kNN_transformer.fit_transform(pairwise_distances.numpy()).toarray()
    #if(matching_matrix == 0):
    #    matching_matrix = labels.numpy()[:, np.newaxis] == labels.numpy()[np.newaxis, :]
    kNN_score = (np.sum(matching_matrix * connectivity_matrix) - n_samples) / (np.sum(connectivity_matrix) - n_samples)
    return kNN_score

def from_data_to_landscapes(data, d=3, delay=3):
    
    # data - torch tensor of the shape (n_samples, L, internal_dim), where L is the length of time series
    # d - integer, determines dimension of points in point cloud: d * internal_dim. 
    # delay - integer, time delay.
    
    # returns:
    # landscapes_final - torch tensor of the shape(n_samples, maxdim + 1, n_layers, n_steps)
    # max_scale - maximum scale value at which at least one persistent landscapes is non-zero

    point_clouds = Takens_Embedding_torch(data, d=3, delay=3)
    diagrams, diagrams_np, VR_persistence = get_diagrams_torch(point_clouds)
    return get_landscapes_torch(diagrams, n_layers=1)
