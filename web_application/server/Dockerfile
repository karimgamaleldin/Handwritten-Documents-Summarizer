# Using Python 3.11-slim as the base image
FROM python:3.11-slim

# Update the package list and install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    graphviz \
    libgraphviz-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgirepository-1.0-1 \
    vim \
    gosu \
    python3-opencv \
    g++ \
    python3-matplotlib \
    gedit \
    curl \
    unzip \
    git \
    wget \
    python3-yaml \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the Python requirements file and install Python dependencies
COPY requirements.txt .

RUN pip install -r requirements.txt

# Added later, don't want to install all the dependencies again :)
RUN pip install pillow 

# Copy the rest of your application's code into the container
COPY . .

# Command to run the application
CMD ["python", "-m", "server"]