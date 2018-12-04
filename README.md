# Self Driving Car Behavioral Cloning

## Overview
This project consists of the development and training of a Deep Neural Network to clone human behavior for self-driving purpose. It uses the Car Simulator for training, test and validation. 

During the training phase, we navigate our car inside the simulator using the keyboard. While we navigating the car the simulator records training images and respective steering angles. Then we use those recorded data to train our neural network. Trained model was tested on two tracks, namely training track and validation track. 

<img src="images/simulator.png" width="750" height="500"/>

## Getting started

### Installation 

To run this project, you need [Miniconda](https://conda.io/miniconda.html) installed(please visit [this link](https://conda.io/docs/install/quick.html) for quick installation instructions.)

To create an environment for this project use the following command:

```
conda env create -f environment.yml
```

After the environment is created, it needs to be activated with the command:

```
source activate self-driving
```

### Access 

The source code for this project is available at [project code](https://github.com/hparik11/Car-Behavioral-Cloning).

### Files

The following files are part of this project:
- **model.py** :   Contains the Convolution Neural Network implemented using Keras framework, together with the pipeline necessary to train and validate the model;
- **md_nvidia.h5** :   Nvidia self driving car architecture's model weights saved during training (model.py);
- **drive.py** : Script that connects the NN with the Car Simulator for testing and validation;
- **data** :     Data source used for training and validation.
- **run.mp4** :   Video of the car driving autonomously on track one.

### Dependency

This project requires the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim) in order to use the Convolution Neural Network to drive the car autonomously.


## How to use this project

### Training the Model

The Convolution Neural Network provided with this project can be trained by executing the following command:

```sh
python model.py
```

### Driving Autonomously
To drive the car autonomously, this project requires the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim). After installing and starting the simulator, the car can be driven autonomously around the track by executing the following command:
```sh
python drive.py model.h5
```

## Training Data 

The data set used to train the model was obtained from the Udacity Simulator. The procedure to collect data followed an empirical process. First, we have driven three laps, in one direction only, at the first track  trying to stay as much as possible aligned at the center of the road. After this first batch of data collection, we tried to train the model. From the behavior shown by our car, we have observed that our model failed on the sharp corners where the side lanes are blurred. From this observation, we  collected more data concerning these corners, which are the corners after the bridge. 

The recovery from sides was achieved using the left and right camera images by adding a "correction angle" to the steering to force the car to regain the center of the road. Finally, the data collection was also augmented by flipping the center camera image otherwise the car would have a tendency to turn to the left because we collected data following only one direction. All this together proved sufficient to achieve the smooth steering shown in our results ([video](https://github.com/otomata/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4)).

Our final data set had 72576 number of pictures. During training, we shuffled the dataset at each epoch and divided it in 80% for training and 20% for validation, to analyze if the model is over fitting or under fitting (model.py lines 80-101).

## Image Pre-Processing (Generators)

Pre-processing the data set is an important step not only to reduce the training time but also to achieve more favorable results. For every image, we first change the color space from BGR, which is the opencv color space, to RBG, which is the simulator color space. Then, we crop the image to remove areas that are not important for the model. Finally, we resize the to the final size of 200x66 pixels. Figure 2, 3, 4 and 5 depict every step of our pre-processing functions


On my first iteration, I tried [LeNet](http://yann.lecun.com/exdb/lenet/) model and [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model. This experiments could be found at [clone.py](clone.py).
The visualizations I used to create this report could be found at [Visualizations.ipynb](Visualizations.ipynb).

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

