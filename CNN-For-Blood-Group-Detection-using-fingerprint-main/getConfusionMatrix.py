import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Directory paths
data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\dataset_blood_group'
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'
test_data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\ValidationData'

# Image dimensions and batch size
IMG_WIDTH = 96
IMG_HEIGHT = 103
BATCH_SIZE = 32

# Load the trained model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Initialize data generator for evaluation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Prepare the data generator for test data
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,  # Preserve order for correct predictions
    color_mode='rgb'
)

# Get true labels for validation data
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Predict the probabilities for all validation images
predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
predicted_labels = np.argmax(predictions, axis=1)

# Generate the confusion matrix for validation data
conf_matrix_validation = confusion_matrix(true_labels, predicted_labels)

# Now, prepare the generator for test data (if you have separate test data)
test_generator_test = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,  # Preserve order for correct predictions
    color_mode='rgb'
)

# Get true labels for test data
true_labels_test = test_generator_test.classes

# Predict the probabilities for all test images
predictions_test = model.predict(test_generator_test, steps=test_generator_test.samples // BATCH_SIZE + 1)
predicted_labels_test = np.argmax(predictions_test, axis=1)

# Generate the confusion matrix for test data
conf_matrix_test = confusion_matrix(true_labels_test, predicted_labels_test)

# Plot confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Confusion matrix for validation data
disp_validation = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_validation, display_labels=class_names)
disp_validation.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=axes[0])
axes[0].set_title("Validation Data Confusion Matrix")

# Confusion matrix for test data
disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=class_names)
disp_test.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=axes[1])
axes[1].set_title("Test Data Confusion Matrix")

# Save the combined confusion matrix image
conf_matrix_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\confusion_matrix.png'
plt.tight_layout()
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved at {conf_matrix_path}")

plt.show()
