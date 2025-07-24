import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt

# Directory paths
data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\dataset_blood_group'
model_path = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\savedModel.keras'
test_data_dir = r'C:\Users\USER\Desktop\Stuff\Code\BloodGroupDetection\ValidationData'

# Image dimensions and settings
IMG_WIDTH = 96   # Corrected to 96 (Width)
IMG_HEIGHT = 103  # Corrected to 103 (Height)
BATCH_SIZE = 32
target_epochs = 50

# Initialize data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.1  # 10% for validation split
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Corrected order: Height x Width
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    color_mode='rgb',  # Ensure RGB images for training
    subset='training'  # Training subset
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Corrected order: Height x Width
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb',
    subset='validation'  # Validation subset
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)  # Dynamically set num_classes based on detected classes
print(f"Detected classes: {class_names}")

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

# Build or load the model
def create_high_accuracy_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (7, 7), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Regularization to prevent overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Dynamically use num_classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
high_acc_model = create_high_accuracy_model()

# Callbacks
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
csv_logger = CSVLogger('training_log.csv', append=True)

# Train and capture history
def train_and_plot():
    try:
        history_high_acc = high_acc_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=target_epochs,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr, checkpoint, csv_logger]
        )
    

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    finally:
        print("\nEvaluating the model on the test dataset...")
        test_eval = high_acc_model.evaluate(test_generator)
        print(f"Test Dataset Evaluation - Loss: {test_eval[0]}, Accuracy: {test_eval[1]}")
        print("Saving the model...")
        high_acc_model.save(model_path)
        print(f"Model saved at {model_path}. Exiting.")

train_and_plot()
