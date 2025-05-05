import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


x = np.linspace(0, 10, 50)  # Create an array of 100 values between 0 and 10
y = np.sin(x)  # Compute the sine of each value in x

plt.figure(figsize=(8, 6))  
plt.plot(x, y)  
plt.title('Sine Wave')  
plt.xlabel('X-axis')  
plt.ylabel('Y-axis')  
plt.grid(True)  
plt.show()  # Display the plot

x = np.random.rand(100)
y = np.random.rand(100)

plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



# Data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Plot
plt.plot(x, y)
plt.title('Line Plot Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
