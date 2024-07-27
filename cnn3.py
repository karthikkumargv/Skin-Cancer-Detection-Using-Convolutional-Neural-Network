import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau

from tensorflow import ragged

# Set seed
np.random.seed(21)
from PIL import Image

directory_benign_train = 'C:\\Users\\VKK\\OneDrive\\Documents\\skin_ZIP\\Skin-Cancer-Detection-main\\train\\benign'
directory_malignant_train = 'C:\\Users\\VKK\\OneDrive\\Documents\\skin_ZIP\\Skin-Cancer-Detection-main\\train\\malignant'
directory_benign_test = 'C:\\Users\\VKK\\OneDrive\\Documents\\skin_ZIP\\Skin-Cancer-Detection-main\\test\\benign'
directory_malignant_test = 'C:\\Users\\VKK\\OneDrive\\Documents\\skin_ZIP\\Skin-Cancer-Detection-main\\test\\malignant'

read = lambda imname: np.asarray(Image.open(imname).convert('RGB'))

img_benign_train = [read(os.path.join(directory_benign_train, filename)) for filename in os.listdir(directory_benign_train)]
img_malignant_train = [read(os.path.join(directory_malignant_train, filename)) for filename in os.listdir(directory_malignant_train)]

img_benign_test = [read(os.path.join(directory_benign_test, filename)) for filename in os.listdir(directory_benign_test)]
img_malignant_test = [read(os.path.join(directory_malignant_test, filename)) for filename in os.listdir(directory_malignant_test)]

# Check the type of img_benign_train
print(type(img_benign_train))


# Converting list to numpy array for faster and more convenient operations going forward

X_benign_train = np.array(img_benign_train, dtype='uint8')
X_malignant_train = np.array(img_malignant_train, dtype='uint8')

X_benign_test = np.array(img_benign_test, dtype='uint8')
X_malignant_test = np.array(img_malignant_test, dtype='uint8')

# Check the type of X_benign_train
print(type(X_benign_train))

# Creating labels: benign is 0 and malignant is 1

y_benign_train = np.zeros(X_benign_train.shape[0])
y_malignant_train = np.ones(X_malignant_train.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

# Check the values of y_malignant_train
print(y_malignant_train)


# Merge data to form complete training and test sets
# axis=0 means rows

X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0) 
y_train = np.concatenate((y_benign_train, y_malignant_train), axis=0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)

# Print shapes
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Print the values in y_test
print("y_test values:", y_test)

# shuffling the data

s1 = np.arange(X_train.shape[0])
np.random.shuffle(s1)
X_train = X_train[s1]
y_train = y_train[s1]

s2 = np.arange(X_test.shape[0])
np.random.shuffle(s2)
X_test = X_test[s2]
y_test = y_test[s2]

# Print an example of shuffled orders for X_train
print("Shuffle orders example for X_train:", s1)
# Displaying the first few images of training set

fig = plt.figure(figsize=(17, 5))
columns = 5
rows = 3

# Determine the number of images to plot (minimum of dataset size and columns*rows)
num_images = min(columns * rows, X_train.shape[0])

for i in range(1, num_images + 1):
    ax = fig.add_subplot(rows, columns, i)
    if y_train[i] == 0:
        ax.set_title('Benign')  # Corrected the method to set title
    else:
        ax.set_title('Malignant')  # Corrected the method to set title
    plt.imshow(X_train[i], interpolation='nearest')

plt.show()

# Convert labels to categorical
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=2)


# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model with VGG16 base
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_categorical, validation_split=0.2, epochs=5, batch_size=32, verbose=1)

# Plotting the results
fig = plt.figure(figsize=(20, 40))
columns = 3
rows = 5

# Summarize model history for accuracy and loss for training and validation

# 1. Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# 2. Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test_categorical)[1]
print("Test Accuracy:", accuracy)

# Learning rate annealer
learning_rate_annealer = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=1e-7)

# Train the model with learning rate annealer
history = model.fit(X_train, y_train_categorical, validation_split=0.2, epochs=5, batch_size=32, verbose=1, callbacks=[learning_rate_annealer])

# Predictions
y_pred_categorical = model.predict(X_test)
y_pred = np.argmax(y_pred_categorical, axis=-1)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Visualize actual vs predicted labels on a subset of the test data
fig = plt.figure(figsize=(20, 40)) 
columns = 3
rows = 5

for i in range(columns * rows):
    ax = fig.add_subplot(rows, columns, i + 1)
    
    # Get the actual and predicted labels
    actual_label = 'Benign' if y_test[i] == 0 else 'Malignant'
    pred_label = 'Benign' if y_pred[i] == 0 else 'Malignant'
    
    # Set title color based on correctness
    title_color = 'green' if actual_label == pred_label else 'red'
    
    # Set the title and display the image
    ax.set_title(f'Actual: {actual_label}\nPredicted: {pred_label}', color=title_color)
    plt.imshow(X_test[i], interpolation='nearest')

plt.show()

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Weighted Precision: {precision}")
print(f"Weighted Recall: {recall}")
print(f"Weighted F1-Score: {f1}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

