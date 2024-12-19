# Cat-Dog-Classification
**Cat and Dog Classifier using CNN and TensorFlow**  This project builds a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow. It demonstrates the application of deep learning in image classification, showcasing the power of CNNs in handling visual data. 🐱🐶📊

## Overview
This project involves building a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow. The model is trained on a dataset of labeled images to accurately distinguish between the two classes. This project demonstrates the application of deep learning techniques in image classification.

## Dataset
The dataset used for this project consists of images of cats and dogs. The images are preprocessed and augmented using the `ImageDataGenerator` class from TensorFlow.

## Model Architecture
The CNN model consists of the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Fully connected (Dense) layers with ReLU and sigmoid activation

## Training
The model is trained using the `fit` method with the following parameters:
- Steps per epoch: `train_generator.samples // 20`
- Epochs: `10`
- Validation data: `test_generator`
- Validation steps: `test_generator.samples // 20`

## Usage
To train the model, run the following code:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 20,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // 20
)
```

## Results
The trained model achieves high accuracy in classifying images of cats and dogs. The performance can be further improved by fine-tuning the model and using more advanced techniques.

## License
This project is licensed under the MIT License.

---

Feel free to customize this README file further based on your specific project details! 😊📄
