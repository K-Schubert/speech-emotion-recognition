# SER
Speech Emotion Recognition on the Emo-DB dataset.

## Project Description
Using audio speech data, build a model that can classify audio samples into different emotions and deploy it using Flask and Docker. For more details about the data, see the ipython notebook.

## How-to-use
The ```.zip``` file contains all the files to build a Flask app and containerize it with Docker. Below, you will find a description about the contents of each directory and instructions to build and run the Docker container.

### Notebooks
You can find the notebook containing the exploratory data analysis and modelling in ```/notebooks/Report_SER.ipynb```. 

### Scripts
You can find all the scripts necessary to run the Flask app in ```/scripts```.
- ```/scripts/app.py``` is the Flask app.
- ```/scripts/commands.sh``` runs the ```app.py``` and ```/notebooks/Report_SER.ipynb``` when lauching the Flask app.
- ```/scripts/download_dataset.py``` downloads and unzips the raw data from http://emodb.bilderbar.info/index-1024.html into ```/data```. The raw wav files are in ```/data/wav```.
- ```/scripts/data_augmentation.py``` creates the augmented dataset into ```/data/aug```.
- ```/scripts/compute_mfccs.py``` creates the features for modelling and analysis and saved them in pickle format into ```/data/pickles/data.pkl```.
- ```/scripts/pickledataset.py``` is used to save the train-val-test sets in pickle format.
- ```/scripts/train_model.py``` trains the final model and saves it as a pickle in ```/models```.

### Models
You can find the final trained model ```/models/knn.pkl``` in .pkl format here.

### Templates
You can find the .html templates ```/templates/pred.html``` and ```/templates/train.html``` for the Flask web app here. 

### Static
You can find the static files to render and style the html templates here.

- ```/static/css``` contains the css stylesheets.

### Plots
You can find the plots for the slides here.

### Slides
You can find the slides for the final presentation here.

### Others
The ```Dockerfile``` and ```requirements.txt``` contain instructions to build the Docker container.

### Data
The ```/data``` directory is created when building the Docker image. It is thus not present in the ```.zip``` file.

The data is stored in the ```/data``` folder in the Docker container.  
- ```/data/wav``` contains the raw .wav audio files.
- ```/data/aug``` contains the augmented .wav audio files.
- ```/data/pickles``` contains the processed train-val-test datasets in .pkl format.

## Build Docker Image
To build the Docker image you must cd to the directory containing the ```Dockerfile``` and run ```docker build -t <name> .``` where ```<name>``` is the name of the image you want to build (eg. ```docker build -t visium_ser:1.0 .```).

## Run Docker Container
To run the Docker container, you must execute ```docker run <name> -p 5000:5000 -p 8888:8888``` where ```<name>``` is the name of the image you want to run (eg. ```docker run -p 5000:5000 -p 8888:8888 visium_ser:1.0```). We are mapping port 5000 of the container to port 5000 on the local computer (Flask), and port 8888 of the container to port 8888 on the local computer (notebook).

## Using the Flask app
Once the container is running, you can access the jupyter notebook ```Report_SER.ipynb``` at ```localhost:8888``` through your web browser (you might need a token that will appear in the terminal). You can also access the Flask app prediction and training endpoints at ```localhost:5000/pred``` and ```localhost:5000/train```. The former allows you to make predictions with the trained model on an audio sample using the test dataset, and the latter allows you to train the final model with different hyperparameters and obtain model diagnostics.
