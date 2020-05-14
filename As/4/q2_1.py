'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

alpha_y = 0.1
def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''

    means = np.zeros((10, 64))
    for i in range(0, 10):
        #i_labels = [train_labels == i]
        means[i,] = train_data[train_labels == i].mean(0)

    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    diag_stabilizer = np.zeros((64,64), float)
    np.fill_diagonal(diag_stabilizer, 0.1)

    for i in range(0,10):
        i_data = train_data[train_labels == i]
        i_data = i_data.T
        i_data -= i_data.mean(axis=1)[:,None]

        N = i_data.shape[1]

        i_cov_matrix = np.dot(i_data, i_data.T.conj()) / float(N - 1)
        i_cov_matrix += diag_stabilizer

        covariances[i] = i_cov_matrix

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    covs_array = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        cov_diag = np.log((cov_diag))
        cov_diag = np.reshape(cov_diag, (8,8))

        covs_array.append(cov_diag)


    all_concat = np.concatenate(covs_array, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)



    Should return an n x 10 numpy array 
    '''

    N = digits.shape[0]
    d = digits.shape[1]
    log_likelihood_matrix = np.zeros((N, 10))

    for i in range(10):
        i_mean = means[i][:,None]
        i_Sigma = covariances[i]
        i_Sigma_inverse = np.linalg.inv(i_Sigma)

        for j in range(N):
            x = digits[j][:,None]

            log_likelihood = 0
            log_likelihood += (-d/2.0 * np.log(2 * np.pi))

            log_likelihood += (-1/2.0 * np.log(np.linalg.det(i_Sigma)))

            x_minus_mu = x - i_mean
            temp = np.matmul(x_minus_mu.T, i_Sigma_inverse)
            temp = np.matmul(temp, x_minus_mu)

            log_likelihood += (-1/2.0 * temp)

            log_likelihood_matrix[j,i] = log_likelihood


    return log_likelihood_matrix


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    if each 8*8 digit representation is unique, we know:

        p(x) = 1/N

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    class_likelihood_matrix = generative_likelihood(digits, means, covariances)
    posterior_matrix = class_likelihood_matrix + np.log(alpha_y)

    return posterior_matrix

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''

    N = digits.shape[0]
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    total_probability = 0
    for i in range(0, N):
        i_label = int(labels[i])
        total_probability += cond_likelihood[i, i_label]

    avg_cond = total_probability / N

    # Compute as described above and return
    return avg_cond

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    digits_predict = np.argmax(cond_likelihood, axis = 1)


    # Compute and return the most likely class
    return digits_predict

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # plot covariance
    plot_cov_diagonal(covariances)

    # Evaluation
    print("Average Log Probability for Train Data:")
    train_avg_conditional = avg_conditional_likelihood(train_data,train_labels, means, covariances)
    print('\t{}'.format(train_avg_conditional))

    print("Average Log Probability for Test Data: ")
    test_avg_condtional = avg_conditional_likelihood(test_data,test_labels, means, covariances)
    print('\t{}'.format(test_avg_condtional))

    # accuracy score
    test_pred = classify_data(test_data, means, covariances)
    print("Accuracy Report on Test Set:")
    print(accuracy_score(test_labels, test_pred))

    test_pred = classify_data(train_data, means, covariances)
    print("Accuracy Report on Train Set:")
    print(accuracy_score(train_labels, test_pred))

if __name__ == '__main__':
    main()