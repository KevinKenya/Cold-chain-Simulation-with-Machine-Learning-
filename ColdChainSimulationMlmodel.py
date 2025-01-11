# -*- coding: utf-8 -*-
"""Cold Chain Warehouse and Logistics Simulation with Machine Learning-COMPLETE and RUNNABLE"""

import pandas as pd
import numpy as np
import random
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- I. Setup and Parameters ---
random.seed(42)  # For reproducibility
np.random.seed(42)

# Mount Google Drive (if using Colab) - Adapt as needed for your environment
from google.colab import drive
try:
    drive.mount('/content/drive')
except ModuleNotFoundError:
    pass
save_path = '/content/drive/My Drive/'  # input your own path

warehouse_capacity = 20000  # Total pallet capacity of the warehouse
num_simulations = 10000  # Number of rows to generate. Adjust based on your memory constraints.
product_types = ['Poultry/Fish', 'Pharmaceuticals (Animal)', 'Supermarket Supplies']
product_distribution = [0.6, 0.1, 0.3]
product_density = {'Poultry/Fish': 0.6, 'Pharmaceuticals (Animal)': 0.4, 'Supermarket Supplies': 0.5}
temperature_zones = ['Frozen (-40°C)', 'Frozen (-20°C)', 'Chilled (+4°C)', 'Ambient (+25°C)']
temp_zone_probabilities = [0.3, 0.3, 0.3, 0.1]  # Must sum to 1
storage_time_means = {'Poultry/Fish': 90, 'Pharmaceuticals (Animal)': 30, 'Supermarket Supplies': 14}
storage_time_stddevs = {'Poultry/Fish': 30, 'Pharmaceuticals (Animal)': 10, 'Supermarket Supplies': 7}
order_size_means = {'Poultry/Fish': 1000, 'Pharmaceuticals (Animal)': 250, 'Supermarket Supplies': 500}
order_size_stddevs = {'Poultry/Fish': 500, 'Pharmaceuticals (Animal)': 100, 'Supermarket Supplies': 200}
max_pallet_weight = 540  # kg
storage_cost_per_pallet_per_week = 14  # $
handling_cost_per_pallet = 17  # $
order_management_cost_per_order = 5  # $

# Vehicle Parameters
vehicle_types = ['Large Truck', 'Delivery Van', 'Motorcycle (Petrol)', 'Motorcycle (Electric)']
vehicle_distribution = [0.2, 0.3, 0.3, 0.2]  # Distribution of vehicle types
vehicle_capacity = {'Large Truck': 20000, 'Delivery Van': 5000, 'Motorcycle (Petrol)': 500, 'Motorcycle (Electric)': 500}  # in kg
fuel_type = {'Large Truck': 'Diesel', 'Delivery Van': 'Diesel', 'Motorcycle (Petrol)': 'Petrol', 'Motorcycle (Electric)': 'Electric'}
fuel_efficiency = {'Large Truck': {'Urban': 5, 'Rural': 4, 'Highway': 6},
                   'Delivery Van': {'Urban': 8, 'Rural': 6, 'Highway': 9},
                   'Motorcycle (Petrol)': {'Urban': 15, 'Rural': 12, 'Highway': 18},
                   'Motorcycle (Electric)': {'Urban': 0, 'Rural': 0, 'Highway': 0}} # Electric consumes no fuel in this simplified model
diesel_price_per_liter_ksh = 173  # KSh
petrol_price_per_liter_ksh = 180  # KSh
electricity_price_per_kwh = 0.2 # Example, adjust
ksh_to_usd = 1/130

route_types = ['Urban', 'Rural', 'Highway']
route_distribution = [0.5, 0.3, 0.2]
distance_means = {'Urban': 25, 'Rural': 75, 'Highway': 150}
distance_stddevs = {'Urban': 10, 'Rural': 25, 'Highway': 50}
delivery_time_means = {'Urban': 1, 'Rural': 3, 'Highway': 5}  # hours
delivery_time_stddevs = {'Urban': 0.5, 'Rural': 1, 'Highway': 1.5}
peak_hour_multiplier = 1.5
truck_loading_unloading_cost_per_5_tonnes = 25  # $

# First-Mile Logistics Parameters
origin_locations = ['Mombasa', 'Jinja', 'Kisumu']  # Add more locations as needed
origin_distribution = [0.4, 0.3, 0.3]
cross_border_fees = {'Jinja': 60000}  # KSh - Add fees for other cross-border locations
inbound_transport_cost_per_km = {'Mombasa': {'Large Truck': 50, 'Delivery Van': 60, 'Rail': 25},  # Example costs in KSh per km
                                 'Jinja': {'Large Truck': 75, 'Delivery Van': 90},  # Higher cost due to cross-border transport
                                 'Kisumu': {'Large Truck': 60, 'Delivery Van': 70}}


# --- II. Warehouse Data Simulation ---
warehouse_data = []
for _ in range(num_simulations):
    product = random.choices(product_types, weights=product_distribution)[0]
    temperature_zone = random.choices(temperature_zones, weights=temp_zone_probabilities)[0]
    storage_time = int(np.random.normal(loc=storage_time_means[product], scale=storage_time_stddevs[product]))
    storage_time = max(1, storage_time)
    weight = random.uniform(100, max_pallet_weight)
    volume = weight / product_density[product]
    order_size = int(np.random.normal(loc=order_size_means[product], scale=order_size_stddevs[product]))
    order_size = max(1, order_size)
    revenue_per_pallet = np.random.randint(50, 200)  # Simulate revenue - Replace with data or more sophisticated logic if available
    storage_cost = (storage_time // 7) * storage_cost_per_pallet_per_week
    handling_cost = handling_cost_per_pallet
    order_management_cost = order_management_cost_per_order
    warehouse_data.append([product, storage_time, temperature_zone, weight, volume, order_size, revenue_per_pallet,
                           storage_cost, handling_cost, order_management_cost])

warehouse_columns = ['product_type', 'storage_time', 'temperature_zone', 'weight', 'volume', 'order_size', 'revenue_per_pallet', 'storage_cost', 'handling_cost', 'order_management_cost']
df_warehouse = pd.DataFrame(warehouse_data, columns=warehouse_columns)

# --- III. Feature Engineering (Warehouse) ---
df_warehouse['pallet_days'] = df_warehouse['storage_time']
df_warehouse['weight_per_pallet'] = df_warehouse['weight']
df_warehouse['volume_per_pallet'] = df_warehouse['volume']

# Simulate Weekend and Peak Season Storage
df_warehouse['weekend_storage'] = 0
df_warehouse['peak_season_storage'] = 0
for index, row in df_warehouse.iterrows():
    start_date = datetime.date(2024, 1, 1) + datetime.timedelta(days=index)
    end_date = start_date + datetime.timedelta(days=row['storage_time'])
    if any(day >= 5 for day in (start_date.weekday(), end_date.weekday())):
        df_warehouse.loc[index, 'weekend_storage'] = 1
    if 11 <= start_date.month <= 12 or 0 <= start_date.month <= 1:
        df_warehouse.loc[index, 'peak_season_storage'] = 1

# Downcasting (Warehouse)
for col in df_warehouse.select_dtypes(include=np.number):
    df_warehouse[col] = pd.to_numeric(df_warehouse[col], downcast='integer')


# --- IV. Logistics Data Simulation ---
logistics_data = []
for _ in range(num_simulations):
    vehicle = random.choices(vehicle_types, weights=vehicle_distribution)[0]
    route = random.choices(route_types, weights=route_distribution)[0]
    distance = int(np.random.normal(loc=distance_means[route], scale=distance_stddevs[route]))
    distance = max(1, distance)
    order_weight = random.uniform(100, 5000)  # Simulate order weight

    fuel_price_ksh = diesel_price_per_liter_ksh if fuel_type[vehicle] == 'Diesel' else petrol_price_per_liter_ksh
    fuel_price_usd = fuel_price_ksh * ksh_to_usd
    if fuel_type[vehicle] == 'Electric':
        fuel_price_usd = electricity_price_per_kwh # Use electricity price for electric vehicles

    peak_hour_delivery = random.random() < 0.3
    delivery_time = int(np.random.normal(loc=delivery_time_means[route], scale=delivery_time_stddevs[route]))
    if peak_hour_delivery:
        delivery_time = int(delivery_time * peak_hour_multiplier)
    delivery_time = max(1, delivery_time)

    fuel_consumed = distance / fuel_efficiency[vehicle][route] if fuel_type[vehicle] != 'Electric' else 0 # No fuel consumption for electric vehicles
    fuel_cost_usd = fuel_consumed * fuel_price_usd

    loading_unloading_cost = (order_weight / 5000) * truck_loading_unloading_cost_per_5_tonnes

    origin_location = random.choices(origin_locations, weights=origin_distribution)[0]
    is_cross_border = origin_location in cross_border_fees
    cross_border_cost = cross_border_fees.get(origin_location, 0) * ksh_to_usd if is_cross_border else 0

    transport_cost_per_km_ksh = inbound_transport_cost_per_km.get(origin_location, {}).get(vehicle, 0)
    inbound_transport_cost = (distance * transport_cost_per_km_ksh) * ksh_to_usd if transport_cost_per_km_ksh else 0

    logistics_data.append([vehicle, route, distance, fuel_price_usd, delivery_time, order_weight,
                           fuel_consumed, inbound_transport_cost, loading_unloading_cost, origin_location, is_cross_border, cross_border_cost])

logistics_columns = ['vehicle_type', 'route_type', 'distance', 'fuel_price', 'delivery_time', 'order_weight',
                    'fuel_consumed', 'inbound_transport_cost', 'loading_unloading_cost', 'origin_location', 'is_cross_border', 'cross_border_cost']
df_logistics = pd.DataFrame(logistics_data, columns=logistics_columns)


# --- V. Feature Engineering (Logistics) ---
df_logistics['delivery_cost_per_km'] = df_logistics['inbound_transport_cost'] / df_logistics['distance'] if df_logistics['distance'].any() else 0 # Handle potential division by zero
df_logistics['delivery_cost_per_order'] = df_logistics['inbound_transport_cost'] / df_logistics['order_weight']
df_logistics['delivery_speed'] = df_logistics['distance'] / df_logistics['delivery_time']
df_logistics['peak_hour_delivery'] = df_logistics.apply(lambda row: row['delivery_time'] > delivery_time_means.get(row['route_type'], 0), axis=1).astype(int)

# --- VI. Combine and Enhance Data ---
df_combined = pd.concat([df_warehouse, df_logistics], axis=1)
df_combined['cost_per_pallet'] = df_combined['storage_cost'] + df_combined['handling_cost']

# Combined Financial Features
df_combined['total_revenue'] = df_combined['revenue_per_pallet'] * df_combined['order_size']
df_combined['total_cost'] = (df_combined['cost_per_pallet'] * df_combined['order_size']) + df_combined['inbound_transport_cost'] + df_combined['loading_unloading_cost'] + df_combined['cross_border_cost']
df_combined['net_profit'] = df_combined['total_revenue'] - df_combined['total_cost']
df_combined['roi'] = (df_combined['net_profit'] / df_combined['total_cost']) * 100  # Calculate ROI as a percentage
# Combined Feature Engineering (Enhanced)
df_combined['product_temp'] = df_combined['product_type'] + '_' + df_combined['temperature_zone']
df_combined['vehicle_route'] = df_combined['vehicle_type'] + '_' + df_combined['route_type']

# Add Seasonality (Example - Sinusoidal for annual seasonality)
day_of_year = df_combined.index.map(lambda x: (datetime.date(2024, 1, 1) + datetime.timedelta(days=x)).timetuple().tm_yday)
df_combined['seasonality'] = np.sin(2 * np.pi * day_of_year / 365.25)

# Add Vehicle Capacity Utilization
df_combined['vehicle_capacity_utilization'] = df_combined['order_weight'] / df_combined['vehicle_type'].map(vehicle_capacity)
# Logarithmic Transformation of ROI (if the distribution is skewed)
df_combined['log_roi'] = np.log1p(df_combined['roi']) # Add 1 to avoid errors with negative or zero ROI values
# --- VII. Data Preprocessing ---
# One-Hot Encoding (Memory Optimization - Use sparse representation)
categorical_features = ['product_temp', 'vehicle_route', 'product_type', 'temperature_zone', 'vehicle_type', 'route_type', 'origin_location'] # Added origin_location
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_features = ohe.fit_transform(df_combined[categorical_features])

# Scaling (Memory Optimization - Fit and transform separately)
numerical_features = ['distance', 'fuel_price', 'storage_time', 'weight', 'volume', 'order_size', 'revenue_per_pallet', 'storage_cost', 'handling_cost', 'order_management_cost', 'pallet_days', 'weight_per_pallet', 'volume_per_pallet', 'delivery_cost_per_km', 'delivery_cost_per_order', 'delivery_speed', 'peak_hour_delivery', 'fuel_consumed', 'inbound_transport_cost', 'loading_unloading_cost', 'is_cross_border', 'cross_border_cost', 'cost_per_pallet', 'total_revenue', 'total_cost', 'seasonality', 'vehicle_capacity_utilization']
scaler = StandardScaler()

# --- VIII. Model Training ---
def train_and_evaluate(target_variable):
    X = df_combined.drop(target_variable, axis=1)
    y = df_combined[target_variable]

    # Split data (memory optimization - scale and encode within each split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])

    X_train_ohe = ohe.transform(X_train[categorical_features])
    X_test_ohe = ohe.transform(X_test[categorical_features])

    X_train_final = np.hstack((X_train_num, X_train_ohe.toarray())) # Combine numerical and encoded features
    X_test_final = np.hstack((X_test_num, X_test_ohe.toarray()))

    model = RandomForestRegressor(random_state=42, n_jobs=-1) # Use all available cores for faster training
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model for {target_variable}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    return model

# Train models for different target variables
model_net_profit = train_and_evaluate('net_profit')
model_roi = train_and_evaluate('roi')
# ... (Train other models as needed)

# --- IX. Save Data and Models (Optional) ---
# ... (Code to save dataframes and trained models - use pickle or joblib for models)

print("Completed.")

# --- IX. Save Data and Models ---
df_combined.to_csv(save_path + 'combined_data.csv', index=False)

import joblib  # Use joblib for sklearn models
joblib.dump(model_net_profit, save_path + 'model_net_profit.joblib')
joblib.dump(model_roi, save_path + 'model_roi.joblib')
# ... (Save other models similarly)

print("Completed. Data and models saved to:", save_path)
