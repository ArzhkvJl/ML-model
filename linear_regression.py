import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


def rmse(y_true, y_pred):
    error = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(error))


# Path of the file to read
csv_file_path = 'xy_dataset.csv'

# Read the file into a DataFrame
student_data = pd.read_csv(csv_file_path)

# Print summary statistics of the dataset
print(student_data.describe())

# Extract academic performance (feature) and time to first job (target)
academic_performance = student_data['x'].values.reshape((-1, 1))  # e.g., GPA or exam score
months_to_first_job = student_data['y'].values  # months to first job after graduation

# Create a plot
fig, ax = plt.subplots()

# Fit linear regression model
model = LinearRegression().fit(academic_performance, months_to_first_job)
coefficient_of_determination = model.score(academic_performance, months_to_first_job)
print('coefficient of determination:', coefficient_of_determination)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Predict the response
predicted_months = model.predict(academic_performance)
print('predicted response:', predicted_months, sep='\n')

# Plot the regression line
ax.plot(academic_performance, predicted_months, color='darkcyan')

# Calculate RMSE for each data point
errors = np.zeros((len(months_to_first_job), 1))


for i in range(len(months_to_first_job)):
    errors[i][0] = rmse(months_to_first_job[i], predicted_months[i])

# Iteratively remove outliers until the model's R^2 is above 0.5
while coefficient_of_determination <= 0.5:
    error_ind = np.argmax(errors)
    academic_performance = np.delete(academic_performance, error_ind).reshape((-1, 1))
    months_to_first_job = np.delete(months_to_first_job, error_ind)
    errors = np.delete(errors, error_ind)
    model = LinearRegression().fit(academic_performance, months_to_first_job)
    coefficient_of_determination = model.score(academic_performance, months_to_first_job)

# Final prediction and plot after outlier removal
predicted_months = model.predict(academic_performance)
ax.scatter(academic_performance, months_to_first_job, color='firebrick', label='Student Data')
ax.plot(academic_performance, predicted_months, color='b', label='Regression Line')
print(academic_performance)
print(months_to_first_job)

coefficient_of_determination = model.score(academic_performance, months_to_first_job)
print('coefficient of determination:', coefficient_of_determination)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

ax.set_xlabel('Academic Performance (e.g., GPA)')
ax.set_ylabel('Months to First Job')
ax.set_title('Predicting Time to First Job Based on Academic Performance')
ax.legend()
plt.show()

# Predict time to first job for new academic performance values
new_academic_performance = np.random.uniform(3.2, 5.0, 5).reshape((-1, 1))
print("Average Grades: \n", new_academic_performance)
new_predicted_months = model.predict(new_academic_performance)
print("Months to First Job: ", new_predicted_months)
