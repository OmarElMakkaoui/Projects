CNN code:  implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset. It loads and preprocesses the dataset, normalizing the pixel values and converting labels into a categorical format. The model follows a LeNet-inspired architecture, using two convolutional layers.

LDA code : trains an LDA classifier by computing class means and scatter matrices, and projects the data into a lower-dimensional space for better class separation. The trained model is then used to classify test samples based on their distance from projected class centroids. 

ADA code : this first trains a single decision tree for baseline classification, then builds an AdaBoost ensemble model, iteratively updating sample weights to improve weak learners.
