#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
# Define the custom activation function
def identity_activation(x):
    return tf.identity(x)

# Load the model with the custom_objects argument
loaded_model_1 = tf.keras.models.load_model("my_model_1.h5", custom_objects={"identity_activation": identity_activation})
loaded_model_2 = tf.keras.models.load_model("my_model_2.h5", custom_objects={"identity_activation": identity_activation})


# In[2]:


df = pd.read_excel('combined_data.xlsx')
# Drop the extra index column
df = df.drop(columns=['Unnamed: 0']) 
df


# In[3]:



# Adding columns for comparision metrics
# For Model 1
df['log_σu'] = df['σu (MPa)'].apply(lambda x: np.log(x))
df['log_σf'] = df['σf (MPa)'].apply(lambda x: np.log(x))

# For Model 2
df['log_hb'] = df['HB'].apply(lambda x: np.log(x))
df['square_ra'] = df['RA (%)'].apply(lambda x: x ** 2)

# Rearranging the columns
df = df[['E (GPa)', 'σu (MPa)','log_σu','σf (MPa)', 'log_σf' , 'εf' , 'RA (%)', 'square_ra', 'HB' , 'log_hb',  'σL (MPa)']
       ]
df


# In[4]:


df.info()


# In[5]:


#Based on FPCM and USM

def evaluate_sigmaF(f, u):
    return 1.12 * f * ((f/u)** 0.893)
def evaluate_b(f, u):
    return (-1 * 0.0792) - 0.179 * math.log10(f/u)

def evaluate_fatigueLimit_predicted_FPCM(coeff , exp , cycles):
    ans = coeff * (cycles ** exp)
    return ans

def evaluate_fatigueLimit_predicted_USM(u, cycles):
    coeff = 1.9018 * u
    exp = -0.12
    ans = coeff * (cycles ** exp)
    return ans


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


# In[6]:


import math
from sklearn.preprocessing import StandardScaler
# Predicting values via different methods
#Conventional Methods
df['Fatigue_FPCM'] = df.apply(prediction_via_FPCM, axis=1)  # Apply along rows
df['Fatigue_USM'] = df.apply(prediction_via_USM, axis=1)  # Apply along rows

# Load your trained StandardScaler for each model




#Neural Network model-1
df_model_1 = df[['log_σu', 'log_σf', 'εf']]

scaler_model_1 = StandardScaler()
scaler_model_1 = scaler_model_1.fit(df_model_1)  # X_train_model_1 should be the data used to train model 1

scaled_df_model_1 = pd.DataFrame(scaler_model_1.transform(df_model_1), columns=df_model_1.columns)
predictions_1 = loaded_model_1.predict(scaled_df_model_1)
df['Model-1'] =  predictions_1

#Neural Network model-2
df_model_2 = df[['log_hb', 'RA (%)', 'square_ra']]

scaler_model_2 = StandardScaler()
scaler_model_2 = scaler_model_2.fit(df_model_2)  # X_train_model_2 should be the data used to train model 2

scaled_df_model_2 = pd.DataFrame(scaler_model_2.transform(df_model_2), columns=df_model_2.columns)
predictions_2 = loaded_model_2.predict(scaled_df_model_2)
df['Model-2'] =  predictions_2



print("DataFrame with new columns:")
df


# In[7]:


# Calculate absolute differences for each method
df['abs_diff_model1'] = abs(df['σL (MPa)'] - df['Model-1'])
df['abs_diff_model2'] = abs(df['σL (MPa)'] - df['Model-2'])
df['abs_diff_model3'] = abs(df['σL (MPa)'] - df['Fatigue_FPCM'])
df['abs_diff_model4'] = abs(df['σL (MPa)'] - df['Fatigue_USM'])


# Define deviation thresholds
#For Model1
deviation_10_percent_model1 = 0.10 * df['Model-1']
deviation_15_percent_model1 = 0.15 * df['Model-1']
deviation_20_percent_model1 = 0.20 * df['Model-1']

#For Model2
deviation_10_percent_model2 = 0.10 * df['Model-2']
deviation_15_percent_model2 = 0.15 * df['Model-2']
deviation_20_percent_model2 = 0.20 * df['Model-2']

#For Model3
deviation_10_percent_model3 = 0.10 * df['Fatigue_FPCM']
deviation_15_percent_model3 = 0.15 * df['Fatigue_FPCM']
deviation_20_percent_model3 = 0.20 * df['Fatigue_FPCM']


#For Model4
deviation_10_percent_model4 = 0.10 * df['Fatigue_USM']
deviation_15_percent_model4 = 0.15 * df['Fatigue_USM']
deviation_20_percent_model4 = 0.20 * df['Fatigue_USM']


# Calculate the percentage of values within the specified deviations for each method
df['within_10_percent_model1'] = ((df['abs_diff_model1'] <= deviation_10_percent_model1) * 100).astype(int)
df['within_15_percent_model1'] = ((df['abs_diff_model1'] <= deviation_15_percent_model1) * 100).astype(int)
df['within_20_percent_model1'] = ((df['abs_diff_model1'] <= deviation_20_percent_model1) * 100).astype(int)

df['within_10_percent_model2'] = ((df['abs_diff_model2'] <= deviation_10_percent_model2) * 100).astype(int)
df['within_15_percent_model2'] = ((df['abs_diff_model2'] <= deviation_15_percent_model2) * 100).astype(int)
df['within_20_percent_model2'] = ((df['abs_diff_model2'] <= deviation_20_percent_model2) * 100).astype(int)

df['within_10_percent_model3'] = ((df['abs_diff_model3'] <= deviation_10_percent_model3) * 100).astype(int)
df['within_15_percent_model3'] = ((df['abs_diff_model3'] <= deviation_15_percent_model3) * 100).astype(int)
df['within_20_percent_model3'] = ((df['abs_diff_model3'] <= deviation_20_percent_model3) * 100).astype(int)

df['within_10_percent_model4'] = ((df['abs_diff_model4'] <= deviation_10_percent_model4) * 100).astype(int)
df['within_15_percent_model4'] = ((df['abs_diff_model4'] <= deviation_15_percent_model4) * 100).astype(int)
df['within_20_percent_model4'] = ((df['abs_diff_model4'] <= deviation_20_percent_model4) * 100).astype(int)




# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Model': ['Model-1', 'Model-2', 'FPCM Model', 'USM Model'],
    'Within 10%': [df['within_10_percent_model1'].mean(), df['within_10_percent_model2'].mean(), df['within_10_percent_model3'].mean(),df['within_10_percent_model4'].mean()],
    'Within 15%': [df['within_15_percent_model1'].mean(), df['within_15_percent_model2'].mean(), df['within_15_percent_model3'].mean(), df['within_15_percent_model4'].mean()],
    'Within 20%': [df['within_20_percent_model1'].mean(), df['within_20_percent_model2'].mean(), df['within_20_percent_model3'].mean(), df['within_20_percent_model4'].mean()],

})

results_df


# ### Plotting the graph

# In[ ]:




