#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF FATIGUE LIMIT FOR HCF FOR LOW_CARBON STEEL

# In[1]:


#Importing the necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import preprocessing, linear_model
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


# In[2]:


df2 = pd.read_excel('dataset2.xlsx')
df3 = pd.read_excel('dataset3.xlsx')
df4 = pd.read_excel('dataset4.xlsx')


# ### Merging all the three dataframes

# In[3]:


#Dropping unnecessary columns
df2 = df2.drop(columns=['Materials' , 'σy (MPa)'])
df3 = df3.drop(columns=['Materials' , 'σy (MPa)'])
df4 = df4.drop(columns=['Materials'])


# In[4]:


merged_df = pd.concat([df2, df3, df4], ignore_index=True)
merged_df.head()


# In[5]:


merged_df.describe()


# ## Analyzing relationship between Fatigue limit and HB

# In[6]:


#First seeing the plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['HB'], merged_df['σL (MPa)'], color='black', alpha=0.5)  # Change color and alpha as needed
plt.title('Relationship between UTS and Fatigue Limit')
plt.xlabel('HB (Independent Variable)')
plt.ylabel('Fatigue Limit (Dependent Variable)')
plt.grid(True)
plt.show()


# ### Comaparing various ML models 

# In[7]:


target_df = merged_df[['HB', 'σL (MPa)']].copy()
target_df.describe()


# In[8]:


from pycaret.regression import *

# Assuming target_df is your DataFrame with columns 'HB' and 'SigmaL'
# Rename 'HB' and 'SigmaL' columns to 'HB' and 'target' for PyCaret
target_df.rename(columns={'HB': 'HB', 'σL (MPa)': 'target'}, inplace=True)

# Initialize the regression setup
reg_setup = setup(data=target_df, target='target', train_size=0.8, session_id=42)

# Compare models
best_model = compare_models()


# ### Applying Linear Regression : Pycaret gave it the best score

# In[9]:


# Step 1: Split the data into input (X) and target (y) variables
X = merged_df[['HB']]  # Independent variable (Brinell hardness)
y = merged_df['σL (MPa)']  # Dependent variable (fatigue limit)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3: Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[10]:


# Step 4: Predictions
y_pred = model.predict(X_test)

plt.scatter(X_test.values, y_test.values, color='blue', label='Experimental')
plt.plot(X_test.values, y_pred, color='red', label='Predicted')
plt.xlabel('Brinell Hardness (HB)')
plt.ylabel('Fatigue Limit (SigmaL)')
plt.title('Linear Regression: Experimental vs. Predicted Fatigue Limit')
plt.legend()
plt.show()


# In[11]:


# Step 7: Get the coefficients and intercept of the linear regression model
slope = model.coef_[0]
intercept = model.intercept_
print("Slope:", slope)
print("Intercept:", intercept)

# Step 8: Derive the mathematical equation
equation = f"SigmaL = {slope:.2f} * HB + {intercept:.2f}"
print("Mathematical Equation:", equation)


# #### Comparing it with the literature review relation 

# In[12]:


# Define empirical relation (example)
def empirical_relation(hb):
    return 1.72 * hb

# Step 3: Plot the experimental values\
plt.figure(figsize=(12, 8))
plt.scatter(merged_df['HB'], merged_df['σL (MPa)'], color='black', label='Experimental')

# Step 4: Plot the linear regression line
plt.plot(merged_df['HB'], model.predict(merged_df[['HB']]), color='black', linestyle='-', label='Linear Regression')

# Step 5: Plot the empirical relation
hb_values = np.linspace(merged_df['HB'].min(), merged_df['HB'].max(), 100)
plt.plot(hb_values, empirical_relation(hb_values), color='black', linestyle='--', label='Empirical Relation')

# Step 6: Set plot labels and title

plt.xlabel('Brinell Hardness (HB)')
plt.ylabel('Fatigue Limit (SigmaL)')
plt.title('Experimental vs. Predicted Fatigue Limit')
plt.legend()

# Step 7: Show the plot
plt.show()


# #### Seeing what percentage of experimental value is present in +-15% of predicted value

# In[13]:





brinell_hardness_array = merged_df['HB'].values.reshape(-1, 1)
fatigue_limit_array = merged_df['σL (MPa)'].values



# Make predictions
predicted_fatigue_limit = model.predict(brinell_hardness_array)

# Calculate deviation
deviation_10_percent = 0.10 * predicted_fatigue_limit
deviation_15_percent = 0.15 * predicted_fatigue_limit
deviation_20_percent = 0.20 * predicted_fatigue_limit

# Count percentage within deviation
within_10_percent = ((fatigue_limit_array >= predicted_fatigue_limit - deviation_10_percent) & (fatigue_limit_array <= predicted_fatigue_limit + deviation_10_percent)).mean() * 100
within_15_percent = ((fatigue_limit_array >= predicted_fatigue_limit - deviation_15_percent) & (fatigue_limit_array <= predicted_fatigue_limit + deviation_15_percent)).mean() * 100
within_20_percent = ((fatigue_limit_array >= predicted_fatigue_limit - deviation_20_percent) & (fatigue_limit_array <= predicted_fatigue_limit + deviation_20_percent)).mean() * 100

print("Percentage within ±10%:", within_10_percent)
print("Percentage within ±15%:", within_15_percent)
print("Percentage within ±20%:", within_20_percent)



# In[14]:


# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot
plt.scatter(merged_df['HB'], merged_df['σL (MPa)'], color='black', label='Actual Fatigue Limit')

# Regression line
plt.plot(merged_df['HB'], predicted_fatigue_limit, color='blue', label='Linear Regression')

# Deviation bands
plt.fill_between(merged_df['HB'], predicted_fatigue_limit - deviation_10_percent, predicted_fatigue_limit + deviation_10_percent, color='none', edgecolor='blue', linestyle='dotted', linewidth=3, alpha=0.3, label='±10% Deviation')
plt.fill_between(merged_df['HB'], predicted_fatigue_limit - deviation_15_percent, predicted_fatigue_limit + deviation_15_percent, color='none', edgecolor='green', linestyle='dotted', linewidth=3, alpha=0.3, label='±15% Deviation')
plt.fill_between(merged_df['HB'], predicted_fatigue_limit - deviation_20_percent, predicted_fatigue_limit + deviation_20_percent, color='none', edgecolor='red', linestyle='dotted', linewidth=3, alpha=0.3, label='±20% Deviation')

# Set plot labels, title, and legend
plt.title('Relationship between Brinell Hardness and Fatigue Limit')
plt.xlabel('Brinell Hardness')
plt.ylabel('Fatigue Limit')
plt.legend()
plt.grid(True)
plt.show()


# ## NEURAL NETWORK APPROACH

# In[15]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have your data loaded into X (UTS) and y (fatigue limit)
# You need to preprocess your data, normalize it, and split it into training and testing sets

# Extract UTS and fatigue limit columns
X_nn = merged_df['HB'].values
y_nn = merged_df['σL (MPa)'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nn.reshape(-1, 1))
y_scaled = scaler.fit_transform(y_nn.reshape(-1, 1))


#DOING SOME STUFF BECAUSE SOME WARNINGS WERE COMING AND CHATGPT SUGGESTED ME TO DO IT
# Reshape y to 1-dimensional array
y_scaled = y_scaled.ravel()




# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Split train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# ### Hyper-parameter Tuning 

# ### a) For a Single hidden layer

# In[16]:


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def find_best_hidden_layer_size(hidden_layer_sizes, X_train, y_train, X_test, y_test):
    # Define the parameter grid for grid search
    param_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': ['relu'],  # activation function
        'solver': ['adam'],       # optimization algorithm
        'alpha': [0.0001],        # L2 penalty (regularization term)
    }

    # Create the MLPRegressor model
    mlp = MLPRegressor(random_state=42, max_iter=1000)

    # Perform grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Extract results from grid search
    results = grid_search.cv_results_
    hidden_layer_sizes = [param['hidden_layer_sizes'][0] for param in results['params']]
    mean_test_score = -results['mean_test_score']

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(hidden_layer_sizes, mean_test_score, marker='o', linestyle='-')
    plt.title('Grid Search Results')
    plt.xlabel('Number of Neurons in Hidden Layer')
    plt.ylabel('Mean Squared Error')
    plt.xticks(hidden_layer_sizes)
    plt.grid(True)
    plt.show()

    # Best parameters and corresponding mean squared error
    print("Best Parameters:", grid_search.best_params_)
    print("Best Mean Squared Error:", -grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print("Test Mean Squared Error:", test_mse)


#Showing the plot for the single layer
result_1_20 = [(n,) for n in range(1, 21)]
find_best_hidden_layer_size(result_1_20, X_train, y_train, X_val, y_val)


# ### b) Defining a function which takes the list of possibilites of neural architecture and returns the best combinations among them

# In[17]:


#Generates all the possible combinations for neural network having two hidden layers and total number of neurons as sum value
def generate_hidden_layer_sizes(sum_value):
    # Generate all possible combinations of 2 hidden layers with total number of neurons equal to total_neurons
    combinations = [(x, sum_value - x) if x >= sum_value - x else (sum_value - x, x) for x in range(1, sum_value // 2 + 1)]
    return combinations


#Gives us the result by trying all the combinations
def find_best_hidden_layer_size_multi(hidden_layer_sizes, X_train, y_train, X_test, y_test):
    # Define the parameter grid for grid search
    param_grid = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': ['relu'],  # activation function
        'solver': ['adam'],       # optimization algorithm
        'alpha': [0.0001],        # L2 penalty (regularization term)
    }

    # Create the MLPRegressor model
    mlp = MLPRegressor(random_state=42, max_iter=1000)

    # Perform grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters and corresponding mean squared error
    best_params = grid_search.best_params_
    best_mean_squared_error = -grid_search.best_score_

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)

    # Create a DataFrame to display results
    results_df = pd.DataFrame({
        'Parameter': ['Best Parameters', 'Best Mean Squared Error', 'Test Mean Squared Error'],
        'Value': [best_params, best_mean_squared_error, test_mse]
    })

    return results_df


# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Total Neurons', 'Best Parameters', 'Best Mean Squared Error', 'Test Mean Squared Error'])
for total_neurons in range(8,22):
    hidden_layer_sizes = generate_hidden_layer_sizes(total_neurons)
    result = find_best_hidden_layer_size_multi(hidden_layer_sizes, X_train, y_train, X_val, y_val)
    # Append the results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame({
            'Total Neurons': total_neurons,
            'Best Parameters': [result.iloc[0]['Value']['hidden_layer_sizes']],
            'Best Mean Squared Error': [result.iloc[1]['Value']],
            'Test Mean Squared Error': [result.iloc[2]['Value']]
        })], ignore_index=True)


    
results_df


# ### Creating the Model 

# In[18]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define the model architecture
layer1_neurons = 8
layer2_neurons = 5
model = Sequential([
    Dense(layer1_neurons, activation='relu', input_shape=(1,)),
    Dense(layer2_neurons, activation='relu'),
    Dense(1)  # No activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                               
# Train the model
batch_size = 8  # You can adjust this value
history = model.fit(X_train, y_train, epochs=1000, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Retrieve the epoch at which early stopping occurred
stopped_epoch = early_stopping.stopped_epoch
print("Early stopping occurred at epoch:", stopped_epoch)


# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[19]:


# Evaluate the model on the testing data
test_loss = model.evaluate(X_test, y_test)
test_loss


# In[20]:



# import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot expected vs predicted values against X_test
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Expected', alpha=0.5)
plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.5)
plt.xlabel('X_test')
plt.ylabel('Values')
plt.title('Expected vs Predicted Values')
plt.legend()
plt.show()


# In[21]:



# Assuming you have trained your model and have the weights available
weights = model.get_weights()

# Extract the weights and biases for each layer
W1, b1, W2, b2, W_output, b_output = weights

# Define a function to compute the output given an input
def predict_output(input_feature):
    # Compute the output of the first hidden layer
    Z1 = np.dot(input_feature, W1) + b1
    Z1_relu = np.maximum(Z1, 0)

    # Compute the output of the second hidden layer
    Z2 = np.dot(Z1_relu, W2) + b2
    Z2_relu = np.maximum(Z2, 0)

    # Compute the output of the neural network (regression output)
    output = np.dot(Z2_relu, W_output) + b_output

    return output

# Assuming your input data has only one feature
input_feature = X_train

# Compute the output using the neural network
predicted_output = predict_output(input_feature)

# Fit a linear model to approximate the relationship
# Assuming input_feature and predicted_output are 1D arrays
coefficients = np.polyfit(input_feature.flatten(), predicted_output.flatten(), 1)

# Extract slope and intercept
slope, intercept = coefficients

# Print the equation
print("Approximate equation: y =", slope, "* x +", intercept)


# In[ ]:




