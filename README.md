# Brain-Tumor-29-Classes
This project aims to detect the type of brain tumor from MRI images using a convolutional neural network (CNN) implemented with TensorFlow.


## Dataset
The dataset used in this project contains MRI images of brain tumors of 44 different types: meningioma, glioma, pituitary, and astrocytoma. The dataset was obtained from Kaggle. The dataset is divided into 3 sets: train, validation, and test sets.


## Preprocessing
The images were resized to a fixed size of 256x256 and the pixel values were normalized between 0 and 1.


## Model-1
The CNN model consists of 4 convolutional layers, each followed by a max pooling layer, and 2 fully connected layers. The output layer has 29 nodes. The model was trained for 50 epochs using the Adam optimizer.


## model-2 
 transfer learning was used with the MobileNet model as the base. The last layer of the MobileNet model was replaced with a new output layer with 29 nodes, corresponding to the number of classes in the dataset. The model was trained for 15 epochs

## Results
- The trained model -1 achieved an accuracy of 79.51% on the test set.
- The trained model -2 achieved an accuracy of 87.32% on the test set.


##Dependencies
The project requires the following dependencies:

- TensorFlow
- numpy
- gdown
- Pillow


You can install the dependencies using the following command:
```bash
  pip install tensorflow numpy gdown Pillow
```


## Usage
To run the project, simply run the Jupyter notebook Brain Tumor Type Detection.ipynb. The notebook contains all the code for downloading the dataset, preprocessing the images, training the model, and evaluating the results.
