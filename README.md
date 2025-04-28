# MNIST-dataset-DNN-Model
Building Deep Neural network model with misnt Dataset in huawei Bootcamp
MNIST Digit Classifier using Deep Neural Network (DNN)
Overview
This project implements a Deep Neural Network (DNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

The model consists of multiple fully connected layers (Dense layers) with ReLU activation, and uses softmax for the output layer to predict the probability distribution over 10 classes (digits 0–9).

Project Structure
Load Data: Load and inspect the MNIST dataset.

Data Preprocessing:

Normalize pixel values to [0, 1].

Reshape images from (28x28) to (784,).

One-hot encode the labels.

Model Building:

4 Dense layers:

512 units with ReLU activation

256 units with ReLU activation

124 units with ReLU activation

Output layer: 10 units with softmax activation

Training:

Optimizer: Adam with learning rate 0.001

Loss: Categorical Crossentropy

Batch size: 128

Epochs: 10

Evaluation: Evaluate the model on the test dataset.

Saving: Save the trained model to ./mnist_model/final_DNN_model.h5.

How to Run
Install the required libraries:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
Run the script:

bash
Copy
Edit
python your_script_name.py
After training, the model will be saved inside the mnist_model directory.

Results
Achieved ~accuracy on test set: (depends slightly, typically around 97%+)

Visualizations
A sample visualization of 9 images from the training set is displayed at the beginning to understand the dataset.

Requirements
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

OS (for saving the model)

Folder Structure
Copy
Edit
├── mnist_model/
│   └── final_DNN_model.h5
├── your_script_name.py
├── README.md
License
This project is for educational purposes.

