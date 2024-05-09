import numpy as np
import scipy.stats
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
def product_of_gaussians(means, covariances):
    """
    Compute the product of multidimensional Gaussian distributions.

    Parameters:
    - means: List of mean vectors for each Gaussian distribution.
    - covariances: List of covariance matrices for each Gaussian distribution.

    Returns:
    - mean_product: Mean vector of the product Gaussian distribution.
    - covariance_product: Covariance matrix of the product Gaussian distribution.
    """

    # Check if the number of distributions is consistent
    if len(means) != len(covariances):
        raise ValueError("Number of means and covariances must be the same.")

    # Initialize variables for the product distribution
    precision_product = np.zeros_like(covariances[0])

    # Compute precision matrix of the product distribution
    for covariance_inv in map(np.linalg.inv, covariances):
        precision_product += covariance_inv

    covariance_product = np.linalg.inv(precision_product)

    # Compute mean of the product distribution
    mean_product = covariance_product @ sum(np.linalg.inv(covariance) @ mean for mean, covariance in zip(means, covariances))

    return mean_product, covariance_product


def TV_estimation(sampleX, sampleY, bin_num):

    histX, bin_edges_X = np.histogram(sampleX, bins=bin_num, density=True)
    histY, bin_edges_Y = np.histogram(sampleY, bins=bin_num, density=True)
    # Calculate the L1 distance estimate
    def l1_distance(x):
        index = np.searchsorted(bin_edges_X, x, side='left')
        if index == 0 or index == bin_num+1:
            s_x = 0
        else:
            s_x = histX[np.searchsorted(bin_edges_X, x, side='left')-1]

        index = np.searchsorted(bin_edges_Y, x, side='left')
        if index == 0 or index == bin_num+1:
            f_x = 0
        else:
            f_x = histY[np.searchsorted(bin_edges_Y, x, side='left')-1]
        return np.abs( s_x - f_x )
    bins = np.unique(np.concatenate((bin_edges_X,bin_edges_Y)))
    lower_bound = np.min(bins)
    upper_bound = np.max(bins)

    distance, _ = quad(l1_distance, lower_bound, upper_bound, epsabs = 1e-3, points = bins, limit = (bin_num+1)*2)
    return distance/2

def generate_positive_semidefinite_matrix(dim):
    # Generate a random matrix
    random_matrix = np.random.rand(dim, dim)

    # Compute its covariance matrix
    covariance_matrix = np.cov(random_matrix, rowvar=False)

    # Ensure positive semi-definite by using Cholesky decomposition
    cholesky_factor = np.linalg.cholesky(covariance_matrix + np.eye(dim)*0.1)
    positive_semidefinite_matrix = cholesky_factor.T @ cholesky_factor
    eigenvalues, _ = np.linalg.eigh(positive_semidefinite_matrix)
    ratio = max(eigenvalues) / min(eigenvalues)
    print(ratio)
    return positive_semidefinite_matrix

def even_odd_layer(num_distributions):
    even_list = []
    odd_list = []
    for i in range(num_distributions):
        layer = math.floor(math.log2(i+1))
        if layer % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

def process_node(node, means, covariances, samples, i, num_nodes, cov_connection):
    parent = (node - 1) // 2
    left_child = 2 * node + 1
    right_child = 2 * node + 2

    temp_means = [means[node]]
    temp_covs = [covariances[node]]

    if node != 0:
        temp_means.append(samples[i - 1, parent, :])
        temp_covs.append(cov_connection)

    if left_child < num_nodes:
        temp_means.append(samples[i - 1, left_child, :])
        temp_covs.append(cov_connection)

    if right_child < num_nodes:
        temp_means.append(samples[i - 1, right_child, :])
        temp_covs.append(cov_connection)

    new_mean, new_cov = product_of_gaussians(temp_means, temp_covs)
    return new_mean, new_cov

def gibbs_sampler_for_binary_tree(num_dimensions, num_iterations, n_layers, means, covariances):
    num_nodes = 2 ** n_layers - 1
    even_list, odd_list = even_odd_layer(num_nodes)
    
    # Initialize samples with zeros
    samples = np.zeros([num_iterations, num_nodes, num_dimensions])

    for node in even_list:
        samples[0, node, :] = np.random.multivariate_normal(np.zeros(num_dimensions),np.identity(num_dimensions), size=1)

    cov_connection = eta * np.identity(num_dimensions)  # Make sure eta is defined

    for i in range(1, num_iterations):
        if i % 10000 == 0:
            print(i)

        for node in odd_list:
            new_mean, new_cov = process_node(node, means, covariances, samples, i, num_nodes, cov_connection)
            samples[i, node, :] = np.random.multivariate_normal(new_mean, new_cov, size=1)
                
        for node in even_list:
            new_mean, new_cov = process_node(node, means, covariances, samples, i+1, num_nodes, cov_connection)
            samples[i, node, :] = np.random.multivariate_normal(new_mean, new_cov, size=1)
    return samples

def detect(array,thres):
    flag = -1
    for i in range(len(array)):
        if array[i] <= thres and flag == -1:
            flag = i
        if array[i] > thres :
            flag = -1
    return flag

# detect the first time that array is smaller than thres. If not detected, return -1.
def weak_detect(array,thres):
    for i in range(len(array)):
        if array[i] <= thres:
            return i
    return -1

def W2_distance(emMean, emCov, trueMean, trueCov):
    return np.sqrt(np.linalg.norm(emMean - trueMean, ord=2) ** 2 +
                   np.trace(emCov + trueCov - 2 * scipy.linalg.sqrtm(scipy.linalg.sqrtm(emCov) @ trueCov @ scipy.linalg.sqrtm(emCov))))

# ablation study on the dimension of the distribution
num_iterations = 100
num_samples = 100
num_layers = 2
num_distributions = 2 ** num_layers - 1

for seed in [0,1,2,3,4]:
    np.random.seed(seed)
    for num_dimensions in [2,4,8,16,32,64]:
        eta = 0.05
        means = [np.zeros(num_dimensions) for _ in range(num_distributions)]
        covariances = [np.identity(num_dimensions) for _ in range(num_distributions)]
        overall_mean, overall_cov = product_of_gaussians(means,covariances)
        final_samples = np.zeros([num_iterations,num_samples, num_dimensions])
        for sample_index in range(num_samples):
            samples = gibbs_sampler_for_binary_tree(num_dimensions, num_iterations, num_layers, means, covariances)
            final_samples[:,sample_index,:] = samples[:,0,:]
        W2s = []
        for iteration_index in range(num_iterations):
            current_samples = final_samples[iteration_index, :,:]
            # estimate the mean here
            emMean = np.mean(current_samples, axis = 0)
            emCov = EmpiricalCovariance(assume_centered=False).fit(current_samples).covariance_
            W2s.append(W2_distance(emMean, emCov, overall_mean,overall_cov))
        plt.plot(W2s)

        print(min(W2s))
    plt.show()
            

            
# results = np.zeros([6,5])

# for seed in [0,1,2,3,4]:
#     np.random.seed(seed)
#     for num_dimensions in [2,4,8,16,32,64]:
#         eta = 0.005
        
#         num_layers = layer
#         num_distributions = 2 ** num_layers - 1
#         means = [np.zeros(num_dimensions) for _ in range(num_distributions)]
#         covariances = [np.identity(num_dimensions)+ np.diag(np.random.uniform(low=0, high=0.1, size = num_dimensions)) for _ in range(num_distributions)]
        
#         overall_mean, overall_cov = product_of_gaussians(means,covariances)

#         samples = gibbs_sampler_for_binary_tree(num_dimensions, num_iterations, num_layers, means, covariances)
        
        
#         projected_samples = np.einsum('ijk,kl->ijl', samples, direction).squeeze()
#         projected_root = projected_samples[:,0]
#         projected_real_samples = np.einsum('ik,kl->il', overall_samples, direction).squeeze()
#         plt.plot(projected_root)
#         plt.title('projected root')
#         plt.show()
#         TVs = []
#         for i in range(burn_in+everyK,num_iterations,everyK):
#             TVs.append(TV_estimation(projected_root[burn_in:i],projected_real_samples,20))
#         print(min(TVs))
#         plt.plot(TVs)
#         plt.show()
#         iteration = burn_in+everyK*(weak_detect(TVs,thres)+1) # if iteration = burn_in, then the TV is always larger than thres
#         print(f"dimensions:{num_dimensions}, thres:{thres}, index:{iteration}")
        
#         index_dimension = int(math.log2(num_dimensions)-1)
#         results[int(seed),index_dimension] = iteration
#     np.save('results.npy',results)
