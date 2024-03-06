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


# # Analyzing relationship between Fatigue limit and UTS

# In[5]:


#First seeing the plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['σu (MPa)'], merged_df['σL (MPa)'], color='blue', alpha=0.5)  # Change color and alpha as needed
plt.title('Relationship between UTS and Fatigue Limit')
plt.xlabel('UTS (Independent Variable)')
plt.ylabel('Fatigue Limit (Dependent Variable)')
plt.grid(True)
plt.show()


# We observe that the scatter plot becomes more dispersive after UTS > 1100Mpa. So its better to analyse the results by splitting the dataset into two parts.\n
# 1) Low-carbon steel having UTS <= 1100 MPa.\
# 2) Low-carbon steel having UTS > 1100 Mpa.

# In[6]:


from sklearn.linear_model import LinearRegression

# Split the dataset into two subsets based on UTS threshold (1100 MPa)
df_less_than_1100 = merged_df[merged_df['σu (MPa)'] < 1100]
df_greater_than_1100 = merged_df[merged_df['σu (MPa)'] >= 1100]

# Create separate Linear Regression models for each subset
model_less_than_1100 = LinearRegression()
model_greater_than_1100 = LinearRegression()

# Fit the models to the data
model_less_than_1100.fit(df_less_than_1100[['σu (MPa)']], df_less_than_1100['σL (MPa)'])
model_greater_than_1100.fit(df_greater_than_1100[['σu (MPa)']], df_greater_than_1100['σL (MPa)'])


# In[7]:


import matplotlib.pyplot as plt

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(df_less_than_1100['σu (MPa)'], df_less_than_1100['σL (MPa)'], color='blue', marker = '*', label='UTS < 1100 MPa')
plt.scatter(df_greater_than_1100['σu (MPa)'], df_greater_than_1100['σL (MPa)'], color='red', marker = 'o', label='UTS >= 1100 MPa')

# Plot the regression lines
plt.plot(df_less_than_1100['σu (MPa)'], model_less_than_1100.predict(df_less_than_1100[['σu (MPa)']]), color='blue', linewidth=2)
plt.plot(df_greater_than_1100['σu (MPa)'], model_greater_than_1100.predict(df_greater_than_1100[['σu (MPa)']]), color='red', linewidth=2)

# Set plot labels and title
plt.title('Relationship between UTS and Fatigue Limit')
plt.xlabel('UTS')
plt.ylabel('Fatigue Limit')
plt.legend()
plt.grid(True)
plt.show()


# In[8]:


# Generate data for the scatter bands
x_values_less_than_1100 = np.linspace(df_less_than_1100['σu (MPa)'].min(), df_less_than_1100['σu (MPa)'].max(), 100)
x_values_greater_than_1100 = np.linspace(df_greater_than_1100['σu (MPa)'].min(), df_greater_than_1100['σu (MPa)'].max(), 100)

# Calculate lower and upper bounds of scatter bands
y_mean_less_than_1100 = model_less_than_1100.predict(x_values_less_than_1100.reshape(-1, 1))
y_mean_greater_than_1100 = model_greater_than_1100.predict(x_values_greater_than_1100.reshape(-1, 1))

deviation_less_than_1100 = 0.15 * y_mean_less_than_1100  # ±10% deviation
deviation_greater_than_1100 = 0.15 * y_mean_greater_than_1100  # ±10% deviation

# Plot the scatter plots and regression lines for UTS <= 1100 and UTS > 1100
plt.figure(figsize=(10, 6))

# Scatter plot and regression line for UTS <= 1100
plt.scatter(df_less_than_1100['σu (MPa)'], df_less_than_1100['σL (MPa)'], color='black', label='UTS <= 1100 MPa')
plt.plot(x_values_less_than_1100, y_mean_less_than_1100, color='black', label='Linear Regression (UTS <= 1100 MPa)')
plt.fill_between(x_values_less_than_1100, y_mean_less_than_1100 - deviation_less_than_1100, y_mean_less_than_1100 + deviation_less_than_1100,  color='none', edgecolor='black', linestyle='dotted', linewidth=3, alpha=0.3)

# Scatter plot and regression line for UTS > 1100
plt.scatter(df_greater_than_1100['σu (MPa)'], df_greater_than_1100['σL (MPa)'], color='black', label='UTS > 1100 MPa')
plt.plot(x_values_greater_than_1100, y_mean_greater_than_1100, color='black', label='Linear Regression (UTS > 1100 MPa)')
plt.fill_between(x_values_greater_than_1100, y_mean_greater_than_1100 - deviation_greater_than_1100, y_mean_greater_than_1100 + deviation_greater_than_1100, color='none', edgecolor='black', linestyle='dotted', linewidth=3, alpha=0.3)

# Set plot labels, title, and legend
plt.title('Relationship between UTS and Fatigue Limit')
plt.xlabel('UTS')
plt.ylabel('Fatigue Limit')
plt.legend()
plt.grid(True)
plt.show()


# Getting the co-efficients and relations that that the model gave me along with its score

# In[9]:


# Assuming you have fitted your model, model_less_than_1100 and model_greater_than_1100, to your data

# For UTS <= 1100
slope_less_than_1100 = model_less_than_1100.coef_[0]
intercept_less_than_1100 = model_less_than_1100.intercept_
equation_less_than_1100 = f"Fatigue Limit = {slope_less_than_1100:.2f} * σu + {intercept_less_than_1100:.2f}"

print("Linear Regression Equation (UTS <= 1100):")
print(equation_less_than_1100)

# For UTS > 1100
slope_greater_than_1100 = model_greater_than_1100.coef_[0]
intercept_greater_than_1100 = model_greater_than_1100.intercept_
equation_greater_than_1100 = f"Fatigue Limit = {slope_greater_than_1100:.2f} * σu + {intercept_greater_than_1100:.2f}"

print("\nLinear Regression Equation (UTS > 1100):")
print(equation_greater_than_1100)


# In[10]:


# Split the dataset into two subsets based on UTS threshold (1400 MPa)
df_less_than_1400 = merged_df[merged_df['σu (MPa)'] < 1400]
df_greater_than_1400 = merged_df[merged_df['σu (MPa)'] >= 1400]


# In[21]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming you already have your dataframes df_less_than_1100 and df_greater_than_1100

# Generate data for the scatter bands and regression lines (assuming you already have these defined)
# x_values_less_than_1100, x_values_greater_than_1100, y_mean_less_than_1100, y_mean_greater_than_1100,
# deviation_less_than_1100, deviation_greater_than_1100
x_values_less_than_1400 = np.linspace(df_less_than_1400['σu (MPa)'].min(), df_less_than_1400['σu (MPa)'].max(), 100)
x_values_greater_than_1400 = np.linspace(df_greater_than_1400['σu (MPa)'].min(), df_greater_than_1400['σu (MPa)'].max(), 100)

# # Split dataframes based on whether UTS is less than or greater than 1400 MPa
# df_less_than_1400 = df_less_than_1100[df_less_than_1100['σu (MPa)'] < 1400]
# df_greater_than_1400 = df_greater_than_1100[df_greater_than_1100['σu (MPa)'] >= 1400]

# Plot for UTS <= 1400 MPa
plt.figure(figsize=(10, 6))

# Scatter plot and regression line for UTS <= 1100
plt.scatter(df_less_than_1100['σu (MPa)'], df_less_than_1100['σL (MPa)'], color='black', label='Experimental (UTS <= 1100 MPa)')
plt.plot(df_less_than_1100['σu (MPa)'], model_less_than_1100.predict(df_less_than_1100[['σu (MPa)']]), color='black', linewidth=2)
plt.plot(x_values_less_than_1400, 0.5 * x_values_less_than_1400, color='black', linestyle='--', label='y = 0.5*x (σu < 1400 MPa)')

# Scatter plot and regression line for UTS > 1100
plt.scatter(df_greater_than_1100['σu (MPa)'], df_greater_than_1100['σL (MPa)'], color='black', label='Experimental (UTS > 1100 MPa)')
plt.plot(df_greater_than_1100['σu (MPa)'], model_greater_than_1100.predict(df_greater_than_1100[['σu (MPa)']]), color='black', linewidth=2)
plt.plot(x_values_greater_than_1400, np.full_like(x_values_greater_than_1400, 700), color='black', linestyle='--', label='y = 700 MPa (σu ≥ 1400 MPa)')

# Set plot labels, title, and legend
plt.title('Relationship between UTS and Fatigue Limit')
plt.xlabel('UTS')
plt.ylabel('Fatigue Limit')
plt.legend()
plt.grid(True)
plt.show()


# ## NEURAL NETWORK METHOD¶

# In[12]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have your data loaded into X (UTS) and y (fatigue limit)
# You need to preprocess your data, normalize it, and split it into training and testing sets

# Extract UTS and fatigue limit columns
X = merged_df['σu (MPa)'].values
y = merged_df['σL (MPa)'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1))
y_scaled = scaler.fit_transform(y.reshape(-1, 1))


#DOING SOME STUFF BECAUSE SOME WARNINGS WERE COMING AND CHATGPT SUGGESTED ME TO DO IT
# Reshape y to 1-dimensional array
y_scaled = y_scaled.ravel()




# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Split train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# ### Hyper-parameter tuning

# #### Analysing for a single hidden layer

# In[13]:


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


# #### Defining a function which takes the list of possibilites of neural architecture and returns the best combinations among them  

# In[14]:


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
for total_neurons in range(8,21):
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

# In[15]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Assuming you have X_train, X_val, X_test, y_train, y_val, y_test already defined

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


# In[16]:


# Evaluate the model on the testing data
test_loss = model.evaluate(X_test, y_test)
test_loss


# In[17]:



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


# In[18]:


# Get the learned weights and biases
weights = model.get_weights()

# Extract the weights and biases for each layer
W1, b1, W2, b2, W_output, b_output = weights

# Assuming your input data has only one feature
input_feature = X_train  # Replace X_train with your input data

# Compute the output of the first hidden layer
Z1 = input_feature.dot(W1) + b1
# Apply ReLU activation function
Z1_relu = np.maximum(Z1, 0)

# Compute the output of the second hidden layer
Z2 = Z1_relu.dot(W2) + b2
# Apply ReLU activation function
Z2_relu = np.maximum(Z2, 0)

# Compute the output of the neural network (regression output)
output = Z2_relu.dot(W_output) + b_output

# 'output' now contains the predicted values from your trained neural network


# In[19]:


# Print the computed output
print("Computed output:", output)


# In[20]:


import numpy as np

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




