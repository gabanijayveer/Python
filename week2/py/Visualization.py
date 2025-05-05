

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Set Seaborn style for aesthetics
sns.set(style='whitegrid')  # Correct way to set Seaborn style
# Alternatively, you can use plt.style.use('seaborn-whitegrid') if you want to use Matplotlib's interface
# Create sample data
np.random.seed(0)  # For reproducibility
data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Values': np.random.randint(1, 20, size=4)
}
df = pd.DataFrame(data)
# Bar plot using Matplotlib
plt.figure(figsize=(8, 5))
plt.bar(df['Category'], df['Values'], color='skyblue')
plt.title('Bar Plot of Categories', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.grid(axis='y')
plt.show()
# Create a box plot using Seaborn
plt.figure(figsize=(8, 5))
sns.boxplot(x='Category', y='Values', data=df, palette='pastel')
plt.title('Box Plot of Values by Category', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.show()
