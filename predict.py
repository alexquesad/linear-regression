import csv

def load_parameters(filename='params.csv'):
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            row = next(reader)
            theta0 = float(row[0])
            theta1 = float(row[1])
        return theta0, theta1
    except Exception as e:
        print("Error reading the parameters:", e)
        return 0, 0  # Default values if file is not found or has an incorrect format

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

if __name__ == "__main__":
    theta0, theta1 = load_parameters()
    
    if theta0 == 0 and theta1 == 0:
        print("Model is not trained yet")
    else:
        while True:
            try:
                mileage = float(input("Insert a mileage to get an estimated car price 🚗:\n"))
                
                if mileage < 0:
                    print("❌ Mileage must be a positive number. Please try again.")
                    continue
                
                estimated_price = predict_price(mileage, theta0, theta1)
                
                if estimated_price < 0:
                    print("The predicted price is below 0. You cannot sell your car 😥.")
                else:
                    print(f"Estimated price is: {estimated_price:.2f}")
                break
            
            except ValueError:
                print("❌ Invalid input. Please enter a positive number.")


