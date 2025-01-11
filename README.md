# Cold-chain-Simulation-with-Machine-Learning-
## Project Overview

This project simulates a cold chain warehouse and logistics operation, incorporating machine learning to predict key performance indicators (KPIs) such as net profit and return on investment (ROI). The simulation models various aspects of the cold chain, including warehouse operations (storage, handling, order management), logistics (vehicle types, routes, fuel consumption, delivery times), financial aspects (costs, revenue), seasonality, and first-mile logistics.

## Motivation

The global cold chain logistics market is experiencing significant growth, driven by increasing demand for temperature-sensitive products like pharmaceuticals, food, and beverages. Efficient and optimized cold chain operations are crucial for ensuring product quality, minimizing waste, and maximizing profitability. This project aims to demonstrate how data simulation and machine learning can be used to model and analyze cold chain operations, providing insights for optimization and decision-making.

## Methodology

The project follows these key steps:

1. **Data Simulation:** Generates synthetic data for warehouse operations, logistics, and financial aspects. The simulation incorporates various parameters, including:
    *   **Warehouse:** Product types, temperature zones, storage times, order sizes, weights, volumes, revenue, and costs.
    *   **Logistics:** Vehicle types, route types, distances, fuel prices, delivery times, order weights, fuel consumption, loading/unloading costs, origin locations, cross-border fees, and inbound transport costs.
    *   **Financial:** Storage costs, handling costs, order management costs, revenue per pallet, fuel costs, loading/unloading costs, cross-border costs, total revenue, total cost, net profit, and ROI.
    *   **Seasonality:** Simulates seasonal variations in demand.
    *   **First-Mile Logistics:** Models inbound transportation from different origin locations, considering cross-border fees and transport costs.

2. **Feature Engineering:** Creates new features from the simulated data to enhance the predictive power of the machine learning models. These features include:
    *   `pallet_days`, `weight_per_pallet`, `volume_per_pallet`
    *   `weekend_storage`, `peak_season_storage`
    *   `delivery_cost_per_km`, `delivery_cost_per_order`, `delivery_speed`
    *   `peak_hour_delivery`
    *   `cost_per_pallet`, `total_revenue`, `total_cost`, `net_profit`, `roi`
    *   `product_temp`, `vehicle_route`
    *   `seasonality`
    *   `vehicle_capacity_utilization`, `log_roi`

3. **Machine Learning:** Trains `RandomForestRegressor` models to predict `net_profit` and `roi` based on the simulated data and engineered features.

4. **Data and Model Persistence:** Saves the generated data to a CSV file and the trained models using `joblib` for later use.

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Python script:**

    ```bash
    python ColdChainSimulationMlmodel.py
    ```

    This will generate the data, train the models, and save the results to the specified `save_path`.

2. **Modify Parameters (Optional):**

    You can modify the simulation parameters at the beginning of the `ColdChainSimulationMlmodel.py` file to explore different scenarios.

## Data

The generated data is saved to `combined_data.csv` and includes the following columns:

*   **Warehouse Data:**
    *   `product_type`: Type of product (e.g., 'Poultry/Fish', 'Pharmaceuticals (Animal)', 'Supermarket Supplies').
    *   `storage_time`: Storage duration in days.
    *   `temperature_zone`: Temperature zone for storage (e.g., 'Frozen (-40°C)', 'Chilled (+4°C)').
    *   `weight`: Weight of the pallet in kg.
    *   `volume`: Volume of the pallet in cubic meters.
    *   `order_size`: Number of pallets in the order.
    *   `revenue_per_pallet`: Revenue generated per pallet.
    *   `storage_cost`: Cost of storage.
    *   `handling_cost`: Cost of handling.
    *   `order_management_cost`: Cost of managing the order.
    *   `pallet_days`: Total pallet days (storage time * number of pallets).
    *   `weight_per_pallet`: Weight per pallet.
    *   `volume_per_pallet`: Volume per pallet.
    *   `weekend_storage`: Indicator for weekend storage (1 if yes, 0 if no).
    *   `peak_season_storage`: Indicator for peak season storage (1 if yes, 0 if no).

*   **Logistics Data:**
    *   `vehicle_type`: Type of vehicle used for delivery (e.g., 'Large Truck', 'Delivery Van').
    *   `route_type`: Type of route (e.g., 'Urban', 'Rural', 'Highway').
    *   `distance`: Distance of the delivery route in km.
    *   `fuel_price`: Price of fuel in USD.
    *   `delivery_time`: Delivery time in hours.
    *   `order_weight`: Total weight of the order in kg.
    *   `fuel_consumed`: Amount of fuel consumed in liters.
    *   `inbound_transport_cost`: Cost of inbound transportation in USD.
    *   `loading_unloading_cost`: Cost of loading and unloading in USD.
    *   `origin_location`: Origin location of the goods (e.g., 'Mombasa', 'Jinja', 'Kisumu').
    *   `is_cross_border`: Indicator for cross-border delivery (True if yes, False if no).
    *   `cross_border_cost`: Cost of cross-border transportation in USD.
    *   `delivery_cost_per_km`: Delivery cost per kilometer.
    *   `delivery_cost_per_order`: Delivery cost per order.
    *   `delivery_speed`: Average delivery speed in km/h.
    *   `peak_hour_delivery`: Indicator for peak hour delivery (1 if yes, 0 if no).

*   **Combined and Engineered Features:**
    *   `cost_per_pallet`: Total cost per pallet (storage + handling).
    *   `total_revenue`: Total revenue for the order.
    *   `total_cost`: Total cost for the order (including logistics and warehouse costs).
    *   `net_profit`: Net profit for the order.
    *   `roi`: Return on investment for the order (%).
    *   `product_temp`: Combined product type and temperature zone.
    *   `vehicle_route`: Combined vehicle type and route type.
    *   `seasonality`: Seasonality factor.
    *   `vehicle_capacity_utilization`: Percentage of vehicle capacity utilized.
    *   `log_roi`: Logarithm of ROI.

## Models

Two `RandomForestRegressor` models are trained:

*   **model_net_profit.joblib:** Predicts `net_profit`.
*   **model_roi.joblib:** Predicts `roi`.

The models are saved using `joblib` and can be loaded for making predictions on new data.

**Model Performance:**

The performance of the models on the test set is printed at the end of the script. The metrics include:

*   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.
*   **R-squared (R2):** Proportion of variance in the target variable explained by the model.

## Future Improvements

*   **Incorporate Temperature Fluctuations:** Model temperature variations during transport and storage.
*   **More Realistic Distributions:** Use more realistic probability distributions for order sizes, weights, and delivery times.
*   **Stochastic Elements:** Add random events like delays, disruptions, and demand fluctuations.
*   **Hyperparameter Tuning:** Optimize model parameters using techniques like `GridSearchCV` or `RandomizedSearchCV`.
*   **Model Comparison:** Evaluate other regression models (e.g., `XGBoost`, `LightGBM`, `Neural Networks`).
*   **Cross-Validation:** Implement k-fold cross-validation for more robust performance evaluation.
*   **Sensitivity Analysis:** Analyze the impact of key parameters on the predicted outcomes.
*   **Optimization:** Use optimization techniques to find the best combination of parameters for maximizing profit or ROI.
*   **Time Series Forecasting:** Incorporate time series models to forecast future demand.
*   **Granular Location Data:** Use more precise location data and integrate with a mapping API for accurate distance and travel time calculations.
*   **Refine First-Mile Logistics:** Add more detail to the first-mile logistics simulation, such as different transportation modes, warehousing at origin, and handling costs at different stages.
*   **Consider Sustainability Metrics:** Include metrics related to environmental impact, such as carbon emissions, to assess the sustainability of different logistics strategies.

## License

This project is licensed under the Creative Commons License(LICENSE).

## Contact Kevin Chege - aiwithafrica@gmail.com - https://www.linkedin.com/in/kevin-chege-328029a
