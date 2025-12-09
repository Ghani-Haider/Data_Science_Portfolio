from hashlib import sha1
import numpy as np
import math
import matplotlib.pyplot as plt

"""This function takes actual and predicted ratings and compute total mean square error(mse) in observed ratings.
"""
def computeError(R,predR):
    
    """Your code to calculate MSE goes here""" 
    total_mse = 0
    error = np.zeros((R.shape[0], R.shape[1]))

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if(R[i][j] != 0):
                error[i][j] = (R[i][j] - predR[i][j]) #**2
    return error


"""
This fucntion takes P (m*k) and Q(k*n) matrices alongwith user bias (U) and item bias (I) and returns predicted rating. 
where m = No of Users, n = No of items
"""
def getPredictedRatings(P,Q,U,I):

    """Your code to predict ratinngs goes here"""

    # compute r_hat
    R_hat = np.matmul(P, Q)
    for i in range(R_hat.shape[0]):
        for j in range(R_hat.shape[1]):
            # add user and item bias
            R_hat[i][j] += U[i] + I[j]

    return R_hat
    
"""This fucntion runs gradient descent to minimze error in ratings by adjusting P, Q, U and I matrices based on gradients.
   The functions returns a list of (iter,mse) tuple that lists mse in each iteration
"""
def runGradientDescent(R,P,Q,U,I,iterations,alpha):
   
    stats = []
    
    """Your gradient descent code goes here"""
    for each_iter in range(iterations):
        total_mse = 0
        R_hat = getPredictedRatings(P, Q, U, I)
        error = computeError(R, R_hat)
        features = P.shape[1]
        # iterating over p values
        for i in range(R_hat.shape[0]):
            # iterating over q values
            for j in range(R_hat.shape[1]):
                # iterating over k features
                for k in range(features):
                    # updating p and q matrices by applying gradient descent
                    P[i][k] = P[i][k] + (2 * alpha * error[i][j] * Q[k][j])
                    Q[k][j] = Q[k][j] + (2 * alpha * error[i][j] * P[i][k])
                # updating Ui and Ij linear matrices by applying gradient descent
                U[i] = U[i] + (2 * alpha * error[i][j])
                I[j] = I[j] + (2 * alpha * error[i][j])
                total_mse += (error[i][j])**2

        stats.append((each_iter, total_mse))
    
    """"finally returns (iter,mse) values in a list"""
    return stats
    
""" 
This method applies matrix factorization to predict unobserved values in a rating matrix (R) using gradient descent.
K is number of latent variables and alpha is the learning rate to be used in gradient decent
"""    

def matrixFactorization(R,k,iterations, alpha):

    """Your code to initialize P, Q, U and I matrices goes here. P and Q will be randomly initialized whereas U and I will be initialized as zeros. 
    Be careful about the dimension of these matrices
    """
    shape = R.shape
    P = np.random.rand(shape[0], k)
    Q = np.random.rand(k, shape[1])
    U = np.zeros(shape[0])
    I = np.zeros(shape[1])

    #Run gradient descent to minimize error
    stats = runGradientDescent(R,P,Q,U,I,iterations,alpha)
    
    print('P matrx:')
    print(P)
    print('Q matrix:')
    print(Q)
    print("User bias:")
    print(U)
    print("Item bias:")
    print(I)
    print("P x Q:")
    print(getPredictedRatings(P,Q,U,I))
    plotGraph(stats)
       
    
def plotGraph(stats):
    i = [i for i,e in stats]
    e = [e for i,e in stats]
    plt.plot(i,e)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.show()    
    
""""
User Item rating matrix given ratings of 5 users for 6 items.
Note: If you want, you can change the underlying data structure and can work with starndard python lists instead of np arrays
We may test with different matrices with varying dimensions and number of latent factors. Make sure your code works fine in those cases.
"""
R = np.array([
[5, 3, 0, 1, 4, 5],
[1, 0, 2, 0, 0, 0],
[3, 1, 0, 5, 1, 3],
[2, 0, 0, 0, 2, 0],
[0, 1, 5, 2, 0, 0],
])

k = 3
alpha = 0.01
iterations = 500

matrixFactorization(R,k,iterations, alpha)
