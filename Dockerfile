FROM python:3.7

MAINTAINER kieran schubert <schubert.kieran@gmail.com>

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install jupyter

WORKDIR /visium

COPY ./requirements.txt requirements.txt

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pandas
RUN pip install matplotlib==3.2.2
RUN pip install tensorflow==2.2
RUN pip install keras

# copy project
COPY . .

WORKDIR /visium/notebooks
RUN chmod 777 Report_SER.ipynb

WORKDIR /visium/scripts

RUN chmod -R 777 ./

RUN python download_dataset.py
RUN python data_augmentation.py
RUN python compute_mfccs.py
RUN python train_model.py

# set app port
EXPOSE 8888 5000


# Run app.py and Report_SER.ipynb when the container launches
CMD ["/bin/bash", "commands.sh"]
