import torch
from scipy.stats import *
import numpy as np
import torch
import random

shape=2048

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Set seed for reproducibility


def calculate_statistics(tensor):
    """
    Calculate statistical parameters for an n x 512 PyTorch tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (n, 512)
        
    Returns:
        torch.Tensor: A vector of shape (K, 1) where K is the number of calculated parameters.
    """
    # Ensure the input tensor has the correct shape
    print(tensor.shape)
    assert tensor.dim() == 2 and tensor.size(1) == shape, "Input tensor must have shape (n, 512)"
    
    # Calculate statistical parameters along the rows (dim=0)
    mean = torch.mean(tensor, dim=0)  # Shape: (512,)
    std = torch.std(tensor, dim=0)    # Shape: (512,)
    print("std",std)
    # Add more statistical parameters if needed, e.g., min, max, etc.
    # min_val = torch.min(tensor, dim=0).values  # Shape: (512,)
    # max_val = torch.max(tensor, dim=0).values  # Shape: (512,)
    
    # Combine all parameters into a single vector
    stats_vector = torch.cat([mean, std])  # Shape: (2*512,)
    
    # Reshape to (K, 1) where K is the number of parameters
    return stats_vector.view(-1, 1)

def mahalanobis_distanceVec(vector, cluster_stats):
    """
    Calculate Mahalanobis distance of a vector to a cluster.
    
    Args:
        vector (torch.Tensor): A 512x1 vector to compare.
        cluster_stats (torch.Tensor): A 1024x1 vector containing the mean and std of the cluster.
    
    Returns:
        float: The Mahalanobis distance.
    """
    # Ensure the inputs have the correct shapes
    assert vector.shape == (shape, 1), "Input vector must be of shape (512, 1)"
    assert cluster_stats.shape == (2*shape, 1), "Cluster stats must be of shape (1024, 1)"
    
    # Extract mean and std from the cluster stats
    mean = cluster_stats[:shape].view(-1)  # Shape: (512,)
    std = cluster_stats[shape:].view(-1)  # Shape: (512,)
    
    # Variance is std squared
    variance = std ** 2  # Shape: (512,)
    
    # Calculate the Mahalanobis distance
    diff = vector.view(-1) - mean  # Shape: (512,)
    mahalanobis = torch.sqrt(torch.sum((diff ** 2) / variance))
    
    return mahalanobis.item()

def mahalanobis_distanceTen(tensor, clusters):
    """
    Calculate Mahalanobis distance of a vector to a cluster.
    
    Args:
        vector (torch.Tensor): A 512x1 vector to compare.
        cluster_stats (numpy.ndArray): A 1024xn vector containing the mean and std of the cluster.
    
    Returns:
        float: The Mahalanobis distance.
    """
    tensor=tensor.cpu().numpy().copy()
    clusters = clusters.copy()
    # Extract Cluster 1 mean and standard deviation
    clusters_means = clusters[:shape,:]  # Shape: (512, 1)
    clusters_stds = clusters[shape:,:]   # Shape: (512, 1)

    # Calculate the mean and standard deviation of the tensor
    tensor_mean = np.mean(tensor, axis=1, keepdims=True)  # Shape: (512, 1)
    tensor_std = np.std(tensor, axis=1, keepdims=True)    # Shape: (512, 1)

    mahalanobis_distances = []
    # Loop through each cluster
    for i in range(clusters_means.shape[1]):
        # Get mean and std for the current cluster
        cluster_mean = clusters_means[:, i:i+1]  # Shape: (512, 1)
        cluster_std = clusters_stds[:, i:i+1]    # Shape: (512, 1)
        cluster_std = np.sqrt(tensor_std**2 + cluster_std**2)+cluster_std
        #cluster_std = np.where(cluster_std <1e-6, 1e-6, cluster_std)
        # Construct the covariance matrix
        print(i,cluster_mean.shape)
        print(cluster_std.shape)
        cov_matrix = np.diag(cluster_std.flatten() ** 2) # Shape: (512, 512)
        #cluster_std = np.where(cluster_std <1e-9, 1e-6, cluster_std)
        print(cov_matrix.shape)
        # Calculate the mean difference
        mean_diff = tensor_mean - cluster_mean  # Shape: (512, 1)
        print(mean_diff.shape)
        # Compute Mahalanobis distance
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse covariance matrix
        except np.linalg.LinAlgError:
            print(f"Covariance matrix for cluster {i} is not invertible!")
            mahalanobis_distances.append(0)
            continue
        #inv_cov_matrix = np.linalg.inv(cov_matrix)  # Inverse of covariance matrix
        print(inv_cov_matrix.shape)
        distance = np.sqrt(mean_diff.T @ inv_cov_matrix @ mean_diff)  # Scalar value
        print(distance)
        mahalanobis_distances.append(distance[0, 0])

        # Convert distances to a numpy array for easier handling
    mahalanobis_distances = np.array(mahalanobis_distances)
    return mahalanobis_distances

def is_close(mahalanobis_distance, dim=2*shape, confidence=0.99):
    """
    Determine if a vector is close to the cluster based on Mahalanobis distance.

    Args:
        mahalanobis_distance (float): The Mahalanobis distance of the vector.
        dim (int): The dimensionality of the data (512 in this case).
        confidence (float): Confidence level (e.g., 0.95, 0.99).

    Returns:
        bool: True if the vector is close, False if it's far.
        float: The critical value for the given confidence level.
    """
    # Calculate the critical value from the Chi-squared distribution
    critical_value = chi2.ppf(confidence, df=dim)

    # Compare squared Mahalanobis distance to the critical value
    mahalanobis_squared = mahalanobis_distance #** 2
    print("m2",mahalanobis_squared)
    return mahalanobis_squared <= critical_value, critical_value