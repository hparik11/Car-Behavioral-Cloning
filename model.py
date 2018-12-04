import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle


def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while(1):
        shuffle(samples)
        for offset in range(0, num_samples , batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # print(batch_sample)
                image = batch_sample[0]
                image = cv2.imread(image)
                # print(image.shape)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(image)
                measurements.append(center_angle)
                
                left = batch_sample[1]
                right = batch_sample[2]
                
                left_image = cv2.imread(left)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                images.append(left_image)
                measurements.append(center_angle + 0.2)
                
                right_image = cv2.imread(right)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                images.append(right_image)
                measurements.append(center_angle - 0.2)

            augmented_images, augmented_measurements = [], []
            for image , angle in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(angle*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train , y_train)

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def trainAndSave(model, modelFile, epochs = 3):
    """
    Train the model `model` using 'mse' lost and 'adam' optimizer for the epochs `epochs`.
    The model is saved at `modelFile`
    """
    model.compile(loss = 'mse', optimizer = 'adam')
    history_object = model.fit_generator(train_generator, \
                                        samples_per_epoch = len(train_samples)*6, \
                                        validation_data = validation_generator, \
                                        nb_val_samples = len(validation_samples), \
                                        nb_epoch = epochs, verbose = 1)

    model.save(modelFile)
    print("Model saved at " + modelFile)
    return history_object

def LeNetModel():
    """
    Creates a LeNet model.
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def NvidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def modifiedNvidiaModel():
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(36,5,5, subsample = (2,2), activation = "relu"))
    model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
    # model.add(MaxPooling2D())
    model.add(Convolution2D(64,3,3, activation = "relu"))
    model.add(Convolution2D(64,3,3, activation = "relu"))
    # model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.05))
    model.add(Dense(50))
    model.add(Dropout(.05))
    model.add(Dense(10))
    model.add(Dropout(.05))
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    lines = []
    with open('data2/driving_log1.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    train_samples , validation_samples = train_test_split(lines, test_size = 0.2 )
    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    train_generator = generator(train_samples ,batch_size = 32)
    validation_generator = generator(validation_samples , batch_size = 32)

    modified_nvidia_model = modifiedNvidiaModel()

    history_object = trainAndSave(modified_nvidia_model, 'md_nvidia_new2.h5', 3)

    ### print the keys contained in the history object
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])