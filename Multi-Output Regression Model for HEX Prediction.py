import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Function to simulate color using Kubelka-Munk equations
def kubelka_munk_simulation(X, pigment_properties, layer_thickness=2):
    # Kubelka-Munk simulation
    # Assuming X contains pigment concentrations for each formulation
    # Example: A simple linear combination of pigment properties and features
    absorption = np.exp(-pigment_properties * layer_thickness)
    scattering = (1 - np.exp(-2 * pigment_properties * layer_thickness)) / (2 * pigment_properties * layer_thickness)
    simulated_color = (1 - absorption) / (2 * scattering)

    # Combine with input features
    simulated_color = np.dot(X, simulated_color)

    return simulated_color


# Custom loss function for hair color prediction
def custom_loss(predicted_color, simulated_color, actual_color):
    # Example: Combination of mean squared error between predicted and simulated colors,
    # and mean squared error between predicted and actual colors
    loss = mean_squared_error(predicted_color, simulated_color) + mean_squared_error(predicted_color, actual_color)

    return loss


# Load your dataset
file_path = 'Input dataset.csv'
df = pd.read_csv(file_path)

# Extract features (X) and target (y)
X = df[['Black', 'Brown', 'Blonde', 'Clear', 'Red', 'Blue', 'Yellow', 'pH', 'Developer Strength', 'Developer Ratio']]
y = df[['Hex Code']]  # Assuming 'Hex Code' is your target column

# Data Splitting
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Optical properties for Kubelka-Munk simulation (replace with actual data)
pigment_properties = np.random.uniform(0.1, 0.9, size=train_X.shape[1])  # Example: Random values between 0.1 and 0.9

# Standardize the data (optional but often recommended)
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

# Initialize the Multi-Output Regression Model (Random Forest Regressor is used here)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    predicted_color = model.predict(val_X_scaled)

    # Simulate color using Kubelka-Munk equations
    simulated_color = kubelka_munk_simulation(val_X_scaled, pigment_properties)

    # Calculate the loss
    loss = custom_loss(predicted_color, simulated_color, val_y)

    # Backward pass and optimization (Note: PyTorch is not used here as it was in the previous example)
    model.fit(train_X_scaled, train_y)  # Update the model using the training data

    # Print loss for monitoring training progress
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# After training, you can use the trained model to predict RGB values for new formulations

# Testing if the model predicts the RGB values correctly:
# 1. Standardize the Input (if used during training)
new_formulation = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 6.0, 10.0, 1.0]]
new_formulation_scaled = scaler.transform(new_formulation)

# 2. Predict HEX Code
predicted_hex = model.predict(new_formulation_scaled)

print(f'Predicted HEX Code: {predicted_hex}')
