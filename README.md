# Stock Price Prediction with CNN-Transformer Model

This project uses a combination of Convolutional Neural Networks (CNN) and Transformer models to predict stock prices based on historical stock data and macroeconomic factors. The preprocessing stage includes the integration of key macroeconomic data, as well as financial technical indicators like Volatility, MACD (Moving Average Convergence Divergence), MACD Signal, and RSI (Relative Strength Index).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing Details](#preprocessing-details)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains the code to build, train, and evaluate a hybrid CNN-Transformer model for stock price prediction. The model takes historical stock data (including daily prices, volume, and technical indicators) and macroeconomic data as input to predict future stock prices. The preprocessing step also incorporates macroeconomic factors and key technical indicators such as:

- **Volatility**
- **MACD (Moving Average Convergence Divergence)**
- **MACD Signal**
- **RSI (Relative Strength Index)**

The goal of this project is to develop a robust model that uses both deep learning and traditional financial indicators to predict stock prices, offering a more holistic approach to stock market forecasting.

## Features

- Integration of macroeconomic data and financial indicators for enhanced predictions.
- CNN and Transformer layers for handling sequential data and capturing long-range dependencies.
- Preprocessing using feature scaling and data augmentation techniques.
- RMSE-based evaluation for model performance.
- Model is trained on a daily stock dataset and saved for future predictions.

## Installation

To get started with this project, you need to have Python 3.x installed along with the required libraries. You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.x
- TensorFlow >=2.0
- Numpy
- Pandas
- Scikit-learn
- Matplotlib

## Usage

To run the stock prediction model, follow these steps:

1. **Prepare your datasets**: You will need two datasets:
    - **Stock Data**: This should contain daily stock information (date, open, high, low, close, volume).
    - **Macroeconomic Data**: This dataset should contain relevant economic indicators (e.g., GDP, inflation, unemployment rate).
    
    Both datasets should be in CSV format.

2. **Place your datasets in the correct directory**: Update the file paths in the code to point to your stock data and macroeconomic data.

3. **Run the script**: You can execute the script to preprocess the data, train the model, and evaluate predictions:

    ```bash
    python stock_prediction.py
    ```

    The trained models will be saved in the `saved_models/` directory.

## Preprocessing Details

### 1. **Stock Data Preprocessing**:
   - The stock dataset is merged with macroeconomic data on the basis of the date.
   - The data is normalized using MinMaxScaler to bring values between 0 and 1.

### 2. **Incorporation of Technical Indicators**:
   - **Volatility**: A measure of price fluctuations over time.
   - **MACD**: A momentum indicator showing the relationship between two moving averages.
   - **MACD Signal**: The moving average of the MACD, used to identify buy/sell signals.
   - **RSI**: A momentum oscillator used to identify overbought or oversold conditions.

These indicators are used to enhance the model's ability to capture market patterns and trends.

## Model Architecture

The model uses a hybrid architecture consisting of:

- **Convolutional Neural Network (CNN)**: This component extracts features from the stock data.
- **Transformer Block**: It captures long-term dependencies and relationships within the time series data.

The model is built with the following layers:
- **Input Layer**: Accepts preprocessed stock data and macroeconomic data.
- **CNN Layer**: Extracts features using convolution and dropout layers.
- **Transformer Layer**: Applies Multi-Head Attention for capturing dependencies.
- **Dense Layers**: Final layers for output prediction, including a dropout layer for regularization.

## Evaluation

The model is evaluated based on **Root Mean Squared Error (RMSE)**, which measures the difference between the predicted and actual stock prices in their original scale.

Additionally, the model predictions are visualized to compare the actual prices with the predicted values.

## Contributing

Feel free to fork this repository, make changes, and open pull requests. If you find any bugs or issues, please report them in the issues section.

