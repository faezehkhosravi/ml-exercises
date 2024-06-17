# Machine Learning Exercises

This repository contains Python code examples that demonstrate my proficiency in scientific programming and the required mathematical concepts for the Master's program in Computer Vision & Data Science.

## Overview

The provided code performs the following tasks:

1. **Data Loading**:
   - Loads the "Wine Quality" dataset from the UCI Machine Learning Repository.

2. **Data Preprocessing**:
   - Handles missing values by filling them with the mean of each column.
   - Creates a new feature by dividing the 'fixed acidity' by 'volatile acidity'.

3. **Exploratory Data Analysis**:
   - Visualizes the distribution of the new feature using a histogram.
   - Displays the correlation matrix of the dataset features.

4. **Model Training**:
   - Prepares the data for modeling by selecting features and target variable.
   - Splits the data into training and testing sets.
   - Trains a linear regression model on the training data.
   - Evaluates the model using Mean Squared Error (MSE) on the test data.
   - Visualizes the actual vs. predicted values for the test data.

## Code Details

The code is divided into the following sections:

### Data Loading

The dataset is loaded from a URL, and basic information about the dataset is displayed.

### Data Preprocessing

Missing values are filled with the mean of each column, and a new feature is created.

### Exploratory Data Analysis

The distribution of the new feature is visualized, and the correlation matrix is displayed.

### Model Training

The data is prepared for modeling, split into training and testing sets, and a linear regression model is trained and evaluated.

## Requirements

- Python 3.8
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/ml-exercises.git
   cd ml-exercises
