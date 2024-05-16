#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('combined_data.xlsx')
# Drop the extra index column
df = df.drop(columns=['Unnamed: 0']) 
df


# ## Analyzing the dataset

# ### 1. Checking for any missing value

# In[3]:


df.info()


# ### 2. Finding the correlation between target and features

# In[4]:


# Separate features and target
features = df.drop(columns=["σL (MPa)"])  # Features
target = df["σL (MPa)"]  # Target


# In[5]:


# Correlation analysis
correlation = features.corrwith(target)
print("Correlation with σL (MPa):\n", correlation)


# In[6]:


# Plot correlation
plt.barh(features.columns, correlation)
plt.xlabel("Correlation with σL (MPa)")
plt.ylabel("Feature")
plt.title("Feature Correlation")
plt.show()


# ### HB , RA  for model-2

# ### Creating the variation for each of the input parameters

# In[7]:


# Removing the unecessary features
df = df.drop(columns = ['E (GPa)' , 'σu (MPa)', 'σf (MPa)', 'εf'])
df


# In[8]:


# Adding relevant variations of the features
# for HB
df['exp_hb'] = df['HB'].apply(lambda x: np.exp(0.0036 * x))
df['log_hb'] = df['HB'].apply(lambda x: np.log(x))


# for RA
df['exp_ra'] = df['RA (%)'].apply(lambda x: np.exp(-0.013 * x))
df['log_ra'] = df['RA (%)'].apply(lambda x: np.log(x))
df['square_ra'] = df['RA (%)'].apply(lambda x: x ** 2)


# Reordering dataframe
df = df[['HB','log_hb','exp_hb','RA (%)', 'log_ra', 'exp_ra' , 'square_ra' , 'σL (MPa)']]


# In[9]:


df


# In[13]:


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define your input parameters and their variations
params = {
    'x1': ['HB', 'log_hb', 'exp_hb'],
    'x2': ['RA (%)', 'log_ra', 'exp_ra'],
    
}

#Results dictionary
results = {}

best_accuracy = 0
best_combination = None

# Generate all possible combinations of parameter variations
for x1_variation in params['x1']:
    for x2_variation in params['x2']:
        combination = x1_variation + " , " + x2_variation + 'square_ra'
        print(combination)
        # Prepare dataset with the current combination
        X = df[[x1_variation , x2_variation , 'square_ra']]
        y = df["σL (MPa)"]

        #Test-train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Creating a basic neural network model
        model = Sequential([
        Dense(8, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(7, activation='relu'),
        Dense(4, activation='relu'),
        #     Dense(10, activation='relu'),
        #     Dense(10, activation='relu'),
        Dense(1)  # Output layer
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # Train the model with early stopping
        history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        # Predict on test set
        y_pred = model.predict(X_test_scaled)

        # Calculate and print accuracy
        accuracy = (1 - mean_squared_error(y_test, y_pred) / np.var(y_test)) * 100
        print("Accuracy:", accuracy)

        #Putting the results in the dictionary
        results[combination] = accuracy


        # Check if current combination gives better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combination = (x1_variation, x2_variation, 'square_ra')

# Print the best combination and its accuracy
print("Best Combination:", best_combination)
print("Best Accuracy:", best_accuracy)


# In[14]:


results


# ### log(hb) , ra , ra^2

# ### Hyper-parameter tuning : grid-search algorithm

# In[15]:


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings('ignore')
# Assuming you have some data X and y
X = df[['log_hb', 'RA (%)', 'square_ra']]
y = df["σL (MPa)"]

#Test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(n,) * num_layers for num_layers in range(2, 6) for n in range(2, 11)],
    'batch_size': [4,8],  # You can adjust these batch sizes
    'early_stopping': [True],  # Enable early stopping
#     'validation_fraction': [0.1, 0.2, 0.3],  # Fraction of training data to use as validation set
    'activation': ['identity', 'tanh', 'relu','sigmoid'],
#     'n_iter_no_change': [5, 10, 15]  # Number of iterations with no improvement to wait before stopping
}

# Create the classifier
clf = MLPRegressor(max_iter=1000)

# Create the GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)


# ###  Creating model based on optimized parameters and hyperparamter

# In[17]:


import tensorflow as tf
def identity_activation(x):
    return tf.identity(x)

# Define the neural network model
model = Sequential([
    Dense(9, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(9, activation='relu'),
#     Dense(4, activation='relu'),
#     Dense(10, activation='relu'),
#     Dense(10, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=4, validation_split=0.2, callbacks=[early_stopping], verbose=1)



# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = (1 - mean_squared_error(y_test, y_pred) / np.var(y_test)) * 100
print("Accuracy:", accuracy)


# In[18]:


import matplotlib.pyplot as plt

def plot_comparison(y_pred, y_test):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Actual vs. Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Example usage
# Replace y_pred and y_test with your actual data
# y_pred = [your predicted values]
# y_test = [your actual values]

plot_comparison(y_pred, y_test.values)
# plot_comparison()


# ### Saving and importing the model

# In[19]:


# Save the model
model.save("my_model_2.h5")


# In[ ]:




