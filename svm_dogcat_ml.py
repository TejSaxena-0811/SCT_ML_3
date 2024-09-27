import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # support vector classifier

# # Set directory to the current folder ("ML")
# dir = os.getcwd()

# categories = ['Cat', 'Dog']

# data = []

# for category in categories:
#     path = os.path.join(dir, category)  # Path to 'Cat' or 'Dog' folder
#     label = categories.index(category)  # label 0 = Cat, label 1 = Dog.

#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)  # Path to each image
#         img_gray = cv2.imread(imgpath, 0)  # Read the image in grayscale

#         # Resizing the image to a fixed size
#         img_resized = cv2.resize(img_gray, (50,50))

#         image = np.array(img_resized).flatten()  # converts 2D image into 1D array.

#         data.append([image , label])

# # print(len(data))


# pick_in = open('data1.pickle' , 'wb')
# pickle.dump(data , pick_in)
# pick_in.close()


#     #     # Using matplotlib to display the image, in grayscale
#     #     plt.imshow(img_gray, cmap='gray')
#     #     plt.title(category)
#     #     plt.axis('off')
#     #     plt.show()
#     #     break  # Break after displaying the first image
#     # break  # Break after processing the first category


pick_in = open('data1.pickle' , 'rb')  # data1.pickle contains preprocessed image data along with corresponding labels.
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature , label in data:
    features.append(feature)
    labels.append(label)


# xtrain, xtest, ytrain, ytest = train_test_split(features , labels , test_size = 0.50)
# #  first we train the data on 50% data, that is, test size = 50% too.
# #  this trained data is saved in model.sav file.
# #  the training is done now, so for testing, we keep the test size as 1% (done in the following lines).


# model = SVC(C = 1, kernel = 'poly', gamma = 'auto')
# model.fit(xtrain , ytrain)

# pick = open("model.sav" , "wb")  # model.sav contains the trained SVM model.
# pickle.dump(model , pick)
# pick.close()


xtrain, xtest, ytrain, ytest = train_test_split(features , labels , test_size = 0.01)

pick = open("model.sav" , "rb")
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)

accuracy = model.score(xtest , ytest)

categories = ['Cat' , 'Dog']

print("Accuracy: " , accuracy)
print("Prediction: " , categories[prediction[0]])

mypet = np.array(xtest[0]).reshape(50,50)
plt.imshow(mypet , cmap = 'gray')
plt.show()