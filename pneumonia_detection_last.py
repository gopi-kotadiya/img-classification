import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Step 1: Define parameters
img_height, img_width = 150, 150
batch_size = 32

train_data_path = 'C:\\Users\\Gopi\\Desktop\\pnemonia\\chest_xray\\train'  # Update this path
test_data_path = 'C:\\Users\\Gopi\\Desktop\\pnemonia\\chest_xray\\test'    # Update this path

# Print the paths for debugging
print(f"Training data path: {train_data_path}")
print(f"Testing data path: {test_data_path}")

# Step 2: Load and preprocess the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

try:
    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
except FileNotFoundError as e:
    print(f"Error loading training data: {e}")

test_datagen = ImageDataGenerator(rescale=1./255)

try:
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )
except FileNotFoundError as e:
    print(f"Error loading testing data: {e}")

# Step 3: Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(train_generator, 
                    epochs=20,
                    validation_data=test_generator)



# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

# Improved function to check if the image is grayscale or colorful
def is_xray_image1(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    # Split the image into its B, G, R components
    b, g, r = cv2.split(img)

    # Calculate the variance of the differences between channels
    color_variance = np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)) + np.mean(np.abs(r - b))

    # Set a threshold for detecting if the image is grayscale or not
    threshold = 15  # A higher threshold allows for more color variance
    if color_variance > threshold:
        return False  # The image is colorful
    return True  # The image is likely grayscale

# Step 7: Make predictions on new images
def predict_image(img_path):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    file_extension = os.path.splitext(img_path)[1].lower()

    if file_extension not in valid_extensions:
        return "Unsupported file format. Please provide a .jpg, .jpeg, or .png image."
    
    if not is_xray_image1(img_path):
        return "The provided image does not appear to be an X-ray."

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probability = predictions[0][0]
    label = "Pneumonia" if probability > 0.5 else "Normal"
    percentage = probability * 100 if probability > 0.5 else (1 - probability) * 100
    return f"{label} (Pneumonia probability: {percentage:.2f}%)"

# After training the model
model.save('pneumonia_detection_model.h5')

# Reload the model
from tensorflow.keras.models import load_model
loaded_model = load_model('pneumonia_detection_model.h5')


# Example usage
result = predict_image('C:\\Users\\Gopi\\Desktop\\pnemonia\\chest_xray\\val\\NORMAL\\NORMAL2-IM-1430-0001.jpeg') 
print(f'The prediction is: {result}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
