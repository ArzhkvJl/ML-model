# Linear Regression: Predicting Time to First Job After Graduation

This project demonstrates a simple machine learning pipeline using linear regression to predict how long it takes students to find their first job after graduation based on their academic performance - GPA.

## Project Overview
- **Goal:** Predict the number of months it takes for students to find their first job after graduation using academic performance as the main feature.
- **Approach:** The script trains a linear regression model, removes outliers to improve model fit, and visualizes the results.

## Dataset Format
The CSV file should have the following columns:

| x   | y   |
|-----|-----|
| GPA | Months to First Job |

- `x`: Academic performance 
- `y`: Number of months to first job after graduation

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies with:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage
1. Ensure your dataset (`xy_dataset.csv`) is in the project directory.
2. Run the script:
   ```bash
   python linear_regression.py
   ```
      **Note:** You can also use the Jupyter notebook `linear_regression.ipynb` for step-by-step explanations, interactive exploration, and visualization. To open it, run:

3. The script will:
   - Load the dataset
   - Train a linear regression model
   - Remove outliers until the model's R² is above 0.5
   - Visualize the data and regression line
   - Print model statistics and predictions
   - Predict the goal output value for the new input

## Output
- Console output with model statistics and predictions
- A plot showing the relationship between academic performance and months to first job

## Customization
- You can modify the dataset or script to use different features or targets as needed.

## Multiple Regression

This project also demonstrates multiple linear regression, which predicts the time to first job after graduation using more than one feature - academic performance and attendance.

- **Goal:** Predict the number of months it takes for students to find their first job after graduation using multiple features such as GPA and attendance.
- **Approach:** The script trains a multiple linear regression model and visualizes the results.

### Dataset Format
The CSV file should have the following columns:

| x1   | x2         | y                   |
|-----|------------|---------------------|
| GPA | Attendance | Months to First Job |

- `x1`: Academic performance 
- `x2`: Attendance % 
- `y`: Number of months to first job after graduation

### Usage
1. Ensure your dataset (e.g., `multidimentional_dataset.csv`) is in the project directory.
2. Run the script:
   ```bash
   python multiple_linear_regression.py
   ```
   **Note:** You can also use the Jupyter notebook `multiple_linear_regression.ipynb` for step-by-step explanations, interactive exploration, and visualization.

3. The script/notebook will:
   - Load the dataset
   - Train a multiple linear regression model
   - Visualize the data and regression plane in 3D
   - Print model statistics and predictions
   - Predict the goal output value for the new input

### Output
- Console output with model statistics and predictions
- A 3D plot showing the relationship between features (GPA, attendance) and months to first job

### Customization
- You can modify the dataset or scripts to use different features or targets as needed.

