import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Simple Line Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column
# First subplot
axs[0].plot(x, y)
axs[0].set_title('Sine Function', fontsize=14)
axs[0].set_xlabel('X-axis (radians)', fontsize=12)
axs[0].set_ylabel('Y-axis (sine values)', fontsize=12)
# Second subplot
axs[1].plot(x, np.cos(x), color='orange')
axs[1].set_title('Cosine Function', fontsize=14)
axs[1].set_xlabel('X-axis (radians)', fontsize=12)
axs[1].set_ylabel('Y-axis (cosine values)', fontsize=12)
# Adjust layout
plt.tight_layout()  # Adjusts subplot parameters to give specified padding
plt.show()
