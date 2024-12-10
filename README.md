# **Car Price Prediction Project**

This project implements a simple **linear regression model** to estimate car prices based on their mileage. The project includes two programs: one for training the model and another for predicting car prices. It also features additional functionalities such as data visualization and performance evaluation.

---

## **Features**

1. **Training Program:**

   - Reads a dataset (`data.csv`) containing mileage and car prices.
   - Trains a linear regression model using gradient descent.
   - Saves the learned parameters (\(\theta_0\) and \(\theta_1\)) to `params.csv`.

2. **Prediction Program:**

   - Prompts the user to input a car's mileage.
   - Uses the trained parameters to estimate the car's price.

3. **Bonus Features:**
   - **Data Visualization:**
     - Plots the dataset points and the regression line for easy inspection.
   - **Performance Metrics:**
     - Computes precision metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to evaluate the model's accuracy.

---

## **File Descriptions**

- **`train.py`:**
  - The training program.
  - Performs linear regression, saves the model parameters, and evaluates performance.
- **`predict.py`:**

  - The prediction program.
  - Loads the trained parameters and predicts car prices based on user-provided mileage.

- **`data.csv`:**

  - The dataset used for training.
  - A CSV file containing two columns: `Mileage` (in kilometers) and `Price` (in dollars).

- **`params.csv`:**
  - File where the trained parameters (\(\theta_0\) and \(\theta_1\)) are saved for use in the prediction program.

---

## **How to Run the Project**

### **1. Train the Model**

1. Place your dataset in the root directory and name it `data.csv`.
   - The file should have a header row with two columns: `Mileage` and `Price`.
2. Run the training script:
   ```bash
   python train.py
   ```
3. After training:
   - The learned parameters (\(\theta_0\) and \(\theta_1\)) are saved to `params.csv`.
   - A plot showing the data points and the regression line will be displayed.
   - Performance metrics (MAE, MSE, R²) will be printed.

---

### **2. Predict Car Prices**

1. Ensure the trained parameters (`params.csv`) exist in the root directory.
2. Run the prediction script:
   ```bash
   python predict.py
   ```
3. Enter the mileage when prompted, and the program will display the estimated car price.

---

## **Model Evaluation**

- **Metrics Reported:**

  - **MAE (Mean Absolute Error):** The average absolute deviation of predictions from actual values.
  - **MSE (Mean Squared Error):** Penalizes larger deviations more heavily.
  - **R² (R-squared):** Indicates how much of the variance in the data is explained by the model.

- **Interpretation:**
  - A low MAE and MSE indicate high accuracy.
  - An R² value close to 1 suggests the model explains most of the variance in car prices.

---

## **Project Structure**

```
Car Price Prediction Project
│
├── train.py         # Training script
├── predict.py       # Prediction script
├── data.csv         # Dataset (to be provided by the user)
├── params.csv       # Saved model parameters (generated after training)
└── README.md        # Project documentation
```

---

## **Dependencies**

- Python 3.x
- Required libraries:
  - `numpy`
  - `matplotlib`

Install dependencies using:

```bash
pip install numpy matplotlib
```

---

## **Future Improvements**

- Add more features to the dataset (e.g., car age, brand, condition).
- Experiment with non-linear regression models for better accuracy.
- Enhance the user interface for better interactivity.

---

This project provides a hands-on approach to implementing a machine learning model from scratch, focusing on fundamental concepts of linear regression and gradient descent.
