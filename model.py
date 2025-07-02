import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = r"E:\Minor Project\Random Crop Pred Model\crop_predictor_model\Prediction model\Updated Final.csv"
df = pd.read_csv(file_path)

# Encode crop labels
label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Label'])

# Scale the features
scaler = StandardScaler()

# Train the Crop Prediction Model
def train_crop_model():
    features = df[['Nitrogen(kg/ha)', 'Phosphorus(Kg/Ha)', 'Pottasium(Kg/Ha)', 'Temperature', 'Rainfall', 'PH', 'Humidity']]
    target = df['Label_Encoded']
    
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    return clf

clf = train_crop_model()
# Function to predict crop and estimate yield & profit
def predict_crop_and_profit(nitrogen, phosphorus, potassium, temperature, rainfall, ph, humidity, total_land, main_crop_land):
    remaining_land = total_land - main_crop_land  # Cultivable land
    loss_percentage = 10  # Assumed loss percentage

    # Convert user input into DataFrame with proper column names
    user_input = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, rainfall, ph, humidity]], 
                              columns=['Nitrogen(kg/ha)', 'Phosphorus(Kg/Ha)', 'Pottasium(Kg/Ha)', 'Temperature', 'Rainfall', 'PH', 'Humidity'])

    # Scale the input data
    input_data_scaled = scaler.transform(user_input)

    # Predict possible crops
    possible_crops_encoded = clf.predict_proba(input_data_scaled)
    sorted_indices = np.argsort(possible_crops_encoded[0])[::-1]  # Sort crops by probability (descending)

    for crop_encoded in sorted_indices:
        predicted_crop = label_encoder.inverse_transform([crop_encoded])[0]
        crop_data = df[df['Label'] == predicted_crop]

        if crop_data.empty:
            continue  # Skip if no data found

        # Extract cost values
        market_price = crop_data['Market Price(Rupees)'].values[0]
        land_preparation = crop_data['Land Preparation Cost'].values[0]
        initial_manure = crop_data['Initial Manure Cost'].values[0]
        fertilizer_cost = crop_data['Fertilizer Cost'].values[0]
        irrigation_cost = crop_data['Irrigation Cost'].values[0]

        # Estimate yield per acre using cost-based formula
        scaling_factor = (fertilizer_cost + initial_manure + irrigation_cost) / 10  
        estimated_yield_per_acre = max((remaining_land * scaling_factor) / market_price, 500)  # Ensuring a minimum yield

        # Calculate total production
        total_production = estimated_yield_per_acre * remaining_land

        # Adjust for climate/pest loss
        adjusted_production = total_production * (1 - (loss_percentage / 100))

        # Calculate total expenses per acre
        total_expenses_per_acre = sum([
            land_preparation, initial_manure, crop_data['Seed Cost'].values[0], crop_data['Weed Removal Cost'].values[0],
            fertilizer_cost, crop_data['Manpower Cost'].values[0], crop_data['Tractor Diesel Cost'].values[0],
            irrigation_cost, crop_data['Pest Control Cost'].values[0], crop_data['Harvesting Cost'].values[0],
            crop_data['Transportation Cost'].values[0], crop_data['Miscellaneous Cost'].values[0]
        ])
        
        total_expenses = total_expenses_per_acre * remaining_land

        # Calculate revenue
        total_revenue = adjusted_production * market_price

        # Calculate profit
        profit = total_revenue - total_expenses

        # If profit is positive, return this crop
        if profit > 0:
            return f"""
            Predicted Crop: {predicted_crop}
            
             Cultivable Land: {remaining_land} acres
             Estimated Yield per Acre: {estimated_yield_per_acre:.2f} kg
             Market Price: ₹{market_price} per kg
             Total Expenses: ₹{total_expenses}
             Total Production: {total_production:.2f} kg
             Loss Percentage: {loss_percentage}%
             Adjusted Production (after loss): {adjusted_production:.2f} kg
             Total Revenue: ₹{total_revenue:.2f}
              Final Profit: ₹{profit:.2f}
            """

    # If no profitable crop found, return this message
    return "No crop is suitable for the given conditions."


# # Example Usage
# result = predict_crop_and_profit(nitrogen=50, phosphorus=30, potassium=20, temperature=25, 
#                                  rainfall=100, ph=6.5, humidity=70, total_land=100, main_crop_land=80)
# print(result)
