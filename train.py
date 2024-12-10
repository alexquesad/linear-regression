import numpy as np
import matplotlib.pyplot as plt
import csv

def load_data(filename='data.csv'):
    try:
        # Load data, skipping the header
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        mileages = data[:, 0]
        prices = data[:, 1]
        return mileages, prices
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def normalize(mileages, prices):
    mileage_min, mileage_max = np.min(mileages), np.max(mileages)
    price_min, price_max = np.min(prices), np.max(prices)
    # Standardize so prices and mileages are between 0 and 1
    mileages_norm = (mileages - mileage_min) / (mileage_max - mileage_min)
    prices_norm = (prices - price_min) / (price_max - price_min)
    return mileages_norm, prices_norm, mileage_min, mileage_max, price_min, price_max

def train_model(mileages, prices, learning_rate=0.1, max_iterations=1000):
    mileages_norm, prices_norm, mileage_min, mileage_max, price_min, price_max = normalize(mileages, prices)

    theta0 = 0.0
    theta1 = 0.0
    
    m = len(mileages_norm)
    
    prev_mse = float('inf')
    iterations = 0
    
    while iterations < max_iterations:
        predictions = theta0 + theta1 * mileages_norm
        
        cur_mse = np.mean((predictions - prices_norm)**2)
        
        # Check for convergence
        if abs(prev_mse - cur_mse) < 1e-7:
            break
        
        gradient0 = learning_rate * np.sum(predictions - prices_norm) / m
        gradient1 = learning_rate * np.sum((predictions - prices_norm) * mileages_norm) / m
        
        theta0 -= gradient0
        theta1 -= gradient1
        
        prev_mse = cur_mse
        iterations += 1
    
    # Denormalize thetas
    denorm_theta1 = (price_max - price_min) * theta1 / (mileage_max - mileage_min)
    denorm_theta0 = price_min + (price_max - price_min) * theta0 - denorm_theta1 * mileage_min
    
    return denorm_theta0, denorm_theta1, mileage_min, mileage_max, price_min, price_max

def save_parameters(theta0, theta1, filename='params.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])

def plot_data_and_regression_line(mileages, prices, theta0, theta1, mileage_min, mileage_max, price_min, price_max):
    plt.figure(figsize=(10, 6))
    plt.scatter(mileages, prices, color='blue', label='Data points')
    # Plot regression line
    x_line = np.linspace(mileage_min, mileage_max, 100)
    y_line = theta0 + theta1 * x_line
    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.title('Car Price vs Mileage')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calculate_precision(mileages, prices, theta0, theta1):
    predictions = theta0 + theta1 * mileages
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - prices))
    
    # Mean Squared Error (MSE)
    mse = np.mean((predictions - prices) ** 2)
    
    # R-squared (Coefficient of Determination)
    ss_total = np.sum((prices - np.mean(prices)) ** 2)
    ss_residual = np.sum((prices - predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return mae, mse, r_squared

if __name__ == "__main__":
    mileages, prices = load_data('data.csv')
    if mileages is not None and prices is not None:
        theta0, theta1, mileage_min, mileage_max, price_min, price_max = train_model(mileages, prices)
        save_parameters(theta0, theta1)
        plot_data_and_regression_line(mileages, prices, theta0, theta1, mileage_min, mileage_max, price_min, price_max)
    
        # Display precision metrics
        mae, mse, r_squared = calculate_precision(mileages, prices, theta0, theta1)
        print(f"Precision Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared: {r_squared:.2f}")