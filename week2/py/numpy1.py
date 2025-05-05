# array and matrix

import numpy as np

array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", array_1d)

array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array (Matrix):\n", array_2d)

array_zero = np.zeros((3, 3))
print("Array of zeros:\n", array_zero)

array_ones = np.ones((2, 2))
print("Array of ones:\n", array_ones)
			
array_random = np.random.rand(2, 2)

print("Array of random numbers:\n", array_random)

array_range = np.arange(10)
print("Array of range number:",array_range)


array_a = np.array([1,2,3])
array_b = np.array([4,5,6])
sum_array = array_a + array_b
print("Sum of arrays:",sum_array)

diff_array = array_a - array_b
print("Difference of arrays:",diff_array)

product_array = array_a * array_b
print("Product of arrays:",product_array)

quotient_array = array_a / array_b
print("Quotient of arrays:",quotient_array)


# matrix
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5,6],[7,8]])
matrix_sum = matrix_a + matrix_b
print('Matrix addition:\n', matrix_sum)

matrix_product = np.dot(matrix_a, matrix_b)
print("Matrix multiplication:\n", matrix_product)


matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5,6],[7,8]])
matrix_product = matrix_a * matrix_b
print("Matrix multiplication:\n", matrix_product)

matrix_transpose = matrix_a.T
print("transpose of matrix A:\n",matrix_transpose)

random_array = np.random.rand(5) 
print("Random Array:", random_array)


mean_value = np.mean(random_array)
print("Mean:", mean_value)
# Median of the array
median_value = np.median(random_array)
print("Median:", median_value)
# Standard deviation of the array
std_deviation = np.std(random_array)
print("Standard Deviation:", std_deviation)
