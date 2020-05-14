import numpy as np
import matplotlib.pyplot as plt
import math

def sample_mean(v):
    return (np.sum(v) / v.size)


np.random.seed(42)
# I have used normal distibution to draw samples from
# mu = 1 ; standard deviation = 3, n = 10
mu = 1.0
var = 9
n = 10
dat = np.random.normal(loc = mu, scale = math.sqrt(var), size = 10)


lambda_array = np.linspace(0, 4, 30)



bias_array = (mu/(lambda_array+1) - mu) ** 2
variance_array = (var * (1/n) * (1 / ((lambda_array+1)**2)))
expected_square_distance = bias_array + variance_array


plt.plot(lambda_array, bias_array, label='Bias')
plt.plot(lambda_array, variance_array, label='Variance')
plt.plot(lambda_array, expected_square_distance, label='Expected Squared Error')

plt.xlabel('Lambda')
plt.ylabel('Value')

plt.title("Bias Variance Decomp Graph")

plt.legend()

plt.show()


