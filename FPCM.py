

# # INDIRECT METHODS

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


# In[5]:


df2 = pd.read_excel('dataset2.xlsx')
df3 = pd.read_excel('dataset3.xlsx')
df4 = pd.read_excel('dataset4.xlsx')


# In[6]:


#Dropping unnecessary columns
df2 = df2.drop(columns=['Materials' , 'σy (MPa)'])
df3 = df3.drop(columns=['Materials' , 'σy (MPa)'])
df4 = df4.drop(columns=['Materials'])

merged_df = pd.concat([df2, df3, df4], ignore_index=True)
merged_df.head()


# In[7]:


merged_df.describe()


# In[8]:


# Count the number of NaN values in each column
na_counts = merged_df.isna().sum()

print("Number of NaN values in each column:")
print(na_counts)


# In[9]:


# Drop rows with any NaN values
merged_df_cleaned = merged_df.dropna()

print("DataFrame after dropping rows with any NaN value:")
print(merged_df_cleaned)


# In[10]:


# Count the number of NaN values in each column
na_counts = merged_df_cleaned.isna().sum()

print("Number of NaN values in each column:")
print(na_counts)


# In[11]:


merged_df_cleaned.describe()


# ### Creating a Pipeline for Indirect method evaluation via FPCM

# In[13]:


def evaluate_sigmaF(f, u):
    return 1.12 * f * ((f/u)** 0.893)
def evaluate_b(f, u):
    return (-1 * 0.0792) - 0.179 * math.log10(f/u)

def evaluate_fatigueLimit_predicted_FPCM(coeff , exp , cycles):
    ans = coeff * (cycles ** exp)
    return ans


# ### Creating a Pipeline for Indirect method evaluation via USM

# In[14]:


def evaluate_fatigueLimit_predicted_USM(u, cycles):
    coeff = 1.9018 * u
    exp = -0.12
    ans = coeff * (cycles ** exp)
    return ans


# In[19]:


import math

# Define your custom function1
def prediction_via_FPCM(row):
    # Extract values from the row
    f = row['σf (MPa)']
    u = row['σu (MPa)']
    
    # Perform some operation based on the values
    coeff = evaluate_sigmaF(f,u)
    exp = evaluate_b(f,u)
    
    #Defining the cycles
    cycles = 10**6
    result = evaluate_fatigueLimit_predicted_FPCM(coeff , exp, cycles)
    
    return result

# Define your custom function2
def prediction_via_USM(row):
    # Extract values from the row
    f = row['σf (MPa)']
    u = row['σu (MPa)']
    
    #Defining the cycles
    cycles = 10**6
    result = evaluate_fatigueLimit_predicted_USM(u , cycles)
    
    return result

# Apply the custom function to each row and assign the result to new columns
merged_df_cleaned = merged_df_cleaned.copy()  # Create a copy of the DataFrame
merged_df_cleaned['Fatigue_FPCM'] = merged_df_cleaned.apply(prediction_via_FPCM, axis=1)  # Apply along rows
merged_df_cleaned['Fatigue_USM'] = merged_df_cleaned.apply(prediction_via_USM, axis=1)  # Apply along rows


print("DataFrame with new columns:")
merged_df_cleaned






# Example DataFrame with original and predicted values
original_values = merged_df_cleaned['σL (MPa)']  # Assuming 'Original' is the column name for original values
predicted_values = merged_df_cleaned['Fatigue_FPCM']  # Assuming 'Predicted' is the column name for predicted values

# Plot the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(original_values, predicted_values, color='black', alpha=0.5)
plt.plot([original_values.min(), original_values.max()], [original_values.min(), original_values.max()], color='black', linestyle='--')  # Add 45-degree line
plt.title('Original vs Predicted Values by FPCM')
plt.xlabel('Original Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()





# Example DataFrame with original and predicted values
original_values = merged_df_cleaned['σL (MPa)']  # Assuming 'Original' is the column name for original values
predicted_values = merged_df_cleaned['Fatigue_USM']  # Assuming 'Predicted' is the column name for predicted values

# Plot the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(original_values, predicted_values, color='black', alpha=0.5)
plt.plot([original_values.min(), original_values.max()], [original_values.min(), original_values.max()], color='black', linestyle='--')  # Add 45-degree line
plt.title('Original vs Predicted Values by USM')
plt.xlabel('Original Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


# In[ ]:




