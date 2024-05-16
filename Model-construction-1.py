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


# ### uts, f, e for Model 1

# ### Creating the variation of each of the input parameters

# In[7]:


# Removing the unecessary features
df = df.drop(columns = ['E (GPa)' , 'RA (%)', 'HB'])
df


# In[8]:


# For uts
df['exp_σu'] = df['σu (MPa)'].apply(lambda x: np.exp(0.0008 * x))
df['log_σu'] = df['σu (MPa)'].apply(lambda x: np.log(x))

# For sigmaf
df['exp_σf'] = df['σf (MPa)'].apply(lambda x: np.exp(0.0006 * x))
df['log_σf'] = df['σf (MPa)'].apply(lambda x: np.log(x))

# for episolonf
df['exp_εf'] = df['εf'].apply(lambda x: np.exp(-0.73 * x))
df['log_εf'] = df['εf'].apply(lambda x: np.log(x))


# Reordering dataframe
df = df[['σu (MPa)','log_σu','exp_σu','σf (MPa)', 'log_σf', 'exp_σf' , 'εf' , 'log_εf' , 'exp_εf' , 'σL (MPa)']]


# In[63]:


df


# In[9]:


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define your input parameters and their variations
params = {
    'x1': ['σu (MPa)', 'log_σu', 'exp_σu'],
    'x2': ['σf (MPa)', 'log_σf', 'exp_σf'],
    'x3': ['εf', 'log_εf', 'exp_εf']
}

#Results dictionary
results = {}

best_accuracy = 0
best_combination = None

# Generate all possible combinations of parameter variations
for x1_variation in params['x1']:
    for x2_variation in params['x2']:
        for x3_variation in params['x3']:
            combination = x1_variation + " , " + x2_variation + " , " + x3_variation
            print(combination)
            # Prepare dataset with the current combination
            X = df[[x1_variation , x2_variation , x3_variation]]
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
                best_combination = (x1_variation, x2_variation, x3_variation)

# Print the best combination and its accuracy
print("Best Combination:", best_combination)
print("Best Accuracy:", best_accuracy)


# In[10]:


results


# ### log(uts) + log(sigmaf), ef

# ### Hyper-parameter tuning : grid-search algorithm

# In[11]:


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings('ignore')
# Assuming you have some data X and y
X = df[['log_σu', 'log_σf', 'εf']]
y = df["σL (MPa)"]

#Test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(n,) * num_layers for num_layers in range(2, 4) for n in range(2, 11)],
    'batch_size': [4],  # You can adjust these batch sizes
    'early_stopping': [True],  # Enable early stopping
#     'validation_fraction': [0.1, 0.2, 0.3],  # Fraction of training data to use as validation set
    'activation': ['identity', 'tanh', 'relu'],
#     'n_iter_no_change': [5, 10, 15]  # Number of iterations with no improvement to wait before stopping
}

# Create the classifier
clf = MLPRegressor(max_iter=1000)

# Create the GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)


# ## Creating model based on optimized parameters and hyperparamter

# In[16]:


import tensorflow as tf
def identity_activation(x):
    return tf.identity(x)

# Define the neural network model
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
history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=4, validation_split=0.2, callbacks=[early_stopping], verbose=1)



# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = (1 - mean_squared_error(y_test, y_pred) / np.var(y_test)) * 100
print("Accuracy:", accuracy)


# In[17]:


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


# ### Saving the model and importing it into another file

# In[18]:


# Save the model
model.save("my_model_1.h5")


# In[ ]:




