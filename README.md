# Lunar Event Detection Project

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Data](#data)
4. [Model Training](#model-training)
5. [Event Detection](#event-detection)
6. [Configuration](#configuration)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

The Lunar Event Detection Project is an innovative application of machine learning techniques to identify seismic events in lunar data. By leveraging the YOLO (You Only Look Once) object detection algorithm, this project aims to automatically detect and classify seismic events in plots generated from lunar mission data.

## Setup

To get started with the project:

1. Install the required dependencies, including the Ultralytics YOLO implementation.
2. Clone the project repository.
3. Set up the appropriate Python environment.

Detailed setup instructions can be found in the project documentation.

## Data

The project utilizes lunar seismic plots stored in the `lunarplots/` directory. These plots are visual representations of seismic data collected during lunar missions, providing the foundation for our event detection system.

## Model Training

The core of the project involves training a custom YOLO model on the lunar seismic data. The training process is documented in the `training_yolo_on_mars.ipynb` notebook. This notebook guides users through the steps of preparing the data, configuring the YOLO model, and executing the training process.

## Event Detection

Once the model is trained, it can be used to detect events in new lunar plots. The `annotate_lunar.ipynb` notebook demonstrates how to apply the trained model to a set of lunar plots, identifying and annotating potential seismic events. This process involves loading the trained model, processing the lunar plot images, and outputting the results with detected events highlighted.

## Configuration

The project uses a `config.yml` file to manage various parameters and settings. This configuration file allows users to customize aspects such as:

- Data sources and file paths
- Detection thresholds and filters
- Output formats and plotting options

Users can modify this file to tailor the project to their needs or experiment with different settings.
