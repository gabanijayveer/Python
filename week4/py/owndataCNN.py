# import tensorflow as tf
# import pandas as pd
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split

# # Load CSV data
# data = pd.read_csv('Data.csv')
# X = data.drop(columns=['liked']).values  # Features
# y = data['liked'].values  # Labels

# # Normalize features
# X = X.astype('float32') / X.max()

# # Convert labels to categorical (assuming binary classification)
# y = to_categorical(y, num_classes=2)

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build the model
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(2, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train,
#           epochs=5,
#           batch_size=32,
#           validation_split=0.1)

# # Evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Test accuracy: {accuracy:.4f}')
# print(f'Test loss: {loss:.4f}')
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load CSV data
data = pd.read_csv('Data.csv')
X = data.drop(columns=['liked']).values  # Features
y = data['liked'].values  # Labels

# Normalize features
X = X.astype('float32') / X.max()  # Normalize to [0, 1]

# Reshape X to fit the model as a "vertical" image:
# Original 13 features become 13 rows, 1 column, 1 channel.
X = X.reshape(-1, 13, 1, 1)

# Convert labels to categorical (assuming binary classification)
y = to_categorical(y, num_classes=2)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
# Set input_shape=(height, width, channels) = (13, 1, 1)
model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(13, 1, 1)))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, (3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          epochs=5,
          batch_size=32,
          validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
print(f'Test loss: {loss:.4f}')
