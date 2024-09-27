# SCT_ML_3
Using SVM to classify images of dogs and cats from the dataset.

The dataset used for the creation of this model contains 2 folders, the "Cat" folder contains 12,500 different images of cats, and the "Dog" folder contains 12,500 different images of dogs.

The data1.pickle file contains preprocessed images along with the corresponding labels (0 or 1).

The model.sav file contains the trained SVM model. The training is done on 50% of the data.


Now, for testing, we read the model.sav file, and keep the test size as 1%.

As output, the model generates a grayscale blurred image, and gives it's prediction whether it's a cat or a dog, along with the accuracy.

It can be seen that the code is commented at various points as it has either been used for setting the file paths, or training the model.
