#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR

# Read the dataset from the csv file
df = pd.read_csv("are_blue_pills_magics.csv", index_col=0)
data = df.to_numpy()
dose = data[:, 0].reshape(-1, 1)
score = data[:, 1].reshape(-1, 1)

# Perform a linear regression
lr = MyLR([0, 0], max_iter=50000)
lr.fit_(dose, score)
thetas = lr.thetas
print(f'thetas: {repr(thetas)}')
predictions = lr.predict_(dose)
print(f'predictions: {repr(predictions)}')
print(f'MSE: {lr.mse_(score, predictions)}')

# Plot a graph with the data and the hypothesis you get for the spacecraft\
# piloting score versus the quantity of "blue pills"
plt.scatter(dose, score, label='True values')
plt.scatter(dose, predictions, color='green', marker='x')
plt.plot(dose, dose * thetas[1, 0] + thetas[0, 0], color='green',
         linestyle='dashed', label='Predictions')
plt.xlabel('Quantity of blue pill (in micrograms)')
plt.ylabel('Space driving score')
plt.legend()
plt.grid()
plt.show()

# Plot the loss function J(θ) in function of θ_1
plt.grid()
theta_0_s = np.linspace(77, 97, 6)
theta_1_s = np.linspace(-15, -3, 100)
for theta_0 in theta_0_s:
    J = []
    for theta_1 in theta_1_s:
        lr = MyLR([theta_0, theta_1], max_iter=50000)
        predictions = lr.predict_(dose)
        J.append(lr.loss_(score, predictions))
    plt.plot(theta_1_s, J, label=f'J(θ_0={theta_0}, θ_1)')
plt.xlim((-15, -3))
plt.ylim((10, 150))
plt.xlabel('θ_1')
plt.ylabel('cost function J(θ_0, θ_1)')
plt.legend()
plt.show()

# Plot the loss function J(θ) in function of θ_0
plt.grid()
theta_0_s = np.linspace(-50, 250, 400)
theta_1_s = np.linspace(-14.5, -2.5, 6)
for theta_1 in theta_1_s:
    J = []
    for theta_0 in theta_0_s:
        lr = MyLR([theta_0, theta_1], max_iter=50000)
        predictions = lr.predict_(dose)
        J.append(lr.loss_(score, predictions))
    plt.plot(theta_0_s, J, label=f'J(θ_0, θ_1={theta_1})')
plt.xlim((-50, 250))
plt.ylim((-100, 2000))
plt.xlabel('θ_0')
plt.ylabel('cost function J(θ_0, θ_1)')
plt.legend()
plt.show()
