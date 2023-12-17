import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the input dataset
file_path = 'path/to/your/dataset.csv'
df = pd.read_csv(file_path)

# Extract features (X) and target (y)
X = df[['Black', 'Brown', 'Blonde', 'Clear', 'Red', 'Blue', 'Yellow', 'pH', 'Developer Strength', 'Developer Ratio']]
y = df[['Hex Code']]  # This is the target output. It is a simplification of individual RGB values to make the model faster.

# Data Splitting
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (optional but often recommended)
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

# Initialize the Multi-Output Regression Model (Random Forest Regressor is used here)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

# Train the model
model.fit(train_X_scaled, train_y)

# Predict on the validation set
val_predictions = model.predict(val_X_scaled)

# Evaluate the model
mse = mean_squared_error(val_y, val_predictions)
print(f'Mean Squared Error on Validation Set: {mse}')

# Now, the trained model can be used to predict RGB values for new formulations

# Testing if the model predicts the RGB values correctly:
# 1. Standardize the Input (if used during training)
new_formulation = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 6.0, 10.0, 1.0]]
new_formulation_scaled = scaler.transform(new_formulation)

# 2. Predict HEX Code
predicted_hex = model.predict(new_formulation_scaled)

print(f'Predicted HEX Code: {predicted_hex}')
