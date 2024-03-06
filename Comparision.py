

# In[1]:


#Importing the necessary library
import copy  # Importing the copy module
import math
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


# ### Merging and Cleaning the dataframes

# In[3]:


#Dropping unnecessary columns
df2 = df2.drop(columns=['Materials' , 'σy (MPa)'])
df3 = df3.drop(columns=['Materials' , 'σy (MPa)'])
df4 = df4.drop(columns=['Materials'])

#Merging all the three dataframes
merged_df = pd.concat([df2, df3, df4], ignore_index=True)


# In[4]:


# Drop rows with any NaN values
merged_df_cleaned = merged_df.dropna()

print("DataFrame after dropping rows with any NaN value:")
merged_df_cleaned.describe()


# In[5]:


# Based on UTS

def prediction_via_uts(row):
    uts = row['σu (MPa)']
    if uts <= 1100:
        return 0.49 * uts - 58.61
    else:
        return 0.27 * uts + 183.35
        



# Based on HB
def prediction_via_HB(row):
    hb = row['HB']
    return 1.44 * hb + 9.38



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




# Apply the custom function to each row and assign the result to new columns
merged_df_cleaned = merged_df_cleaned.copy()  # Create a copy of the DataFrame
merged_df_cleaned['Fatigue_UTS'] = merged_df_cleaned.apply(prediction_via_uts, axis=1)  # Apply along rows
merged_df_cleaned['Fatigue_HB'] = merged_df_cleaned.apply(prediction_via_HB, axis=1)  # Apply along rows
merged_df_cleaned['Fatigue_FPCM'] = merged_df_cleaned.apply(prediction_via_FPCM, axis=1)  # Apply along rows
merged_df_cleaned['Fatigue_USM'] = merged_df_cleaned.apply(prediction_via_USM, axis=1)  # Apply along rows


print("DataFrame with new columns:")
merged_df_cleaned


# In[6]:


# Calculate absolute differences for each method
merged_df_cleaned['abs_diff_model1'] = abs(merged_df_cleaned['σL (MPa)'] - merged_df_cleaned['Fatigue_UTS'])
merged_df_cleaned['abs_diff_model2'] = abs(merged_df_cleaned['σL (MPa)'] - merged_df_cleaned['Fatigue_HB'])
merged_df_cleaned['abs_diff_model3'] = abs(merged_df_cleaned['σL (MPa)'] - merged_df_cleaned['Fatigue_FPCM'])
merged_df_cleaned['abs_diff_model4'] = abs(merged_df_cleaned['σL (MPa)'] - merged_df_cleaned['Fatigue_USM'])


# Define deviation thresholds
#For Model1
deviation_10_percent_model1 = 0.10 * merged_df_cleaned['Fatigue_UTS']
deviation_15_percent_model1 = 0.15 * merged_df_cleaned['Fatigue_UTS']
deviation_20_percent_model1 = 0.20 * merged_df_cleaned['Fatigue_UTS']

#For Model2
deviation_10_percent_model2 = 0.10 * merged_df_cleaned['Fatigue_HB']
deviation_15_percent_model2 = 0.15 * merged_df_cleaned['Fatigue_HB']
deviation_20_percent_model2 = 0.20 * merged_df_cleaned['Fatigue_HB']

#For Model3
deviation_10_percent_model3 = 0.10 * merged_df_cleaned['Fatigue_FPCM']
deviation_15_percent_model3 = 0.15 * merged_df_cleaned['Fatigue_FPCM']
deviation_20_percent_model3 = 0.20 * merged_df_cleaned['Fatigue_FPCM']


#For Model4
deviation_10_percent_model4 = 0.10 * merged_df_cleaned['Fatigue_USM']
deviation_15_percent_model4 = 0.15 * merged_df_cleaned['Fatigue_USM']
deviation_20_percent_model4 = 0.20 * merged_df_cleaned['Fatigue_USM']


# Calculate the percentage of values within the specified deviations for each method
merged_df_cleaned['within_10_percent_model1'] = ((merged_df_cleaned['abs_diff_model1'] <= deviation_10_percent_model1) * 100).astype(int)
merged_df_cleaned['within_15_percent_model1'] = ((merged_df_cleaned['abs_diff_model1'] <= deviation_15_percent_model1) * 100).astype(int)
merged_df_cleaned['within_20_percent_model1'] = ((merged_df_cleaned['abs_diff_model1'] <= deviation_20_percent_model1) * 100).astype(int)

merged_df_cleaned['within_10_percent_model2'] = ((merged_df_cleaned['abs_diff_model2'] <= deviation_10_percent_model2) * 100).astype(int)
merged_df_cleaned['within_15_percent_model2'] = ((merged_df_cleaned['abs_diff_model2'] <= deviation_15_percent_model2) * 100).astype(int)
merged_df_cleaned['within_20_percent_model2'] = ((merged_df_cleaned['abs_diff_model2'] <= deviation_20_percent_model2) * 100).astype(int)

merged_df_cleaned['within_10_percent_model3'] = ((merged_df_cleaned['abs_diff_model3'] <= deviation_10_percent_model3) * 100).astype(int)
merged_df_cleaned['within_15_percent_model3'] = ((merged_df_cleaned['abs_diff_model3'] <= deviation_15_percent_model3) * 100).astype(int)
merged_df_cleaned['within_20_percent_model3'] = ((merged_df_cleaned['abs_diff_model3'] <= deviation_20_percent_model3) * 100).astype(int)

merged_df_cleaned['within_10_percent_model4'] = ((merged_df_cleaned['abs_diff_model4'] <= deviation_10_percent_model4) * 100).astype(int)
merged_df_cleaned['within_15_percent_model4'] = ((merged_df_cleaned['abs_diff_model4'] <= deviation_15_percent_model4) * 100).astype(int)
merged_df_cleaned['within_20_percent_model4'] = ((merged_df_cleaned['abs_diff_model4'] <= deviation_20_percent_model4) * 100).astype(int)


# Calculate standard deviation for each method
std_dev_model1 = merged_df_cleaned['abs_diff_model1'].std()
std_dev_model2 = merged_df_cleaned['abs_diff_model2'].std()
std_dev_model3 = merged_df_cleaned['abs_diff_model3'].std()
std_dev_model4 = merged_df_cleaned['abs_diff_model4'].std()

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Model': ['UTS Model', 'HB Model', 'FPCM Model', 'USM Model'],
    'Within 10%': [merged_df_cleaned['within_10_percent_model1'].mean(), merged_df_cleaned['within_10_percent_model2'].mean(), merged_df_cleaned['within_10_percent_model3'].mean(), merged_df_cleaned['within_10_percent_model4'].mean()],
    'Within 15%': [merged_df_cleaned['within_15_percent_model1'].mean(), merged_df_cleaned['within_15_percent_model2'].mean(), merged_df_cleaned['within_15_percent_model3'].mean(), merged_df_cleaned['within_15_percent_model4'].mean()],
    'Within 20%': [merged_df_cleaned['within_20_percent_model1'].mean(), merged_df_cleaned['within_20_percent_model2'].mean(), merged_df_cleaned['within_20_percent_model3'].mean(), merged_df_cleaned['within_20_percent_model4'].mean()],
    'Standard Deviation': [std_dev_model1, std_dev_model2, std_dev_model3, std_dev_model4]
})

results_df


# ### Plotting the graph

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# Step 1: Select a set of randomly picked indices
num_points = 10  # Number of randomly picked points
random_indices = np.random.choice(len(merged_df), num_points, replace=False)

# Step 2: Calculate relative error for each model at the selected indices
relative_errors_uts = (merged_df_cleaned.iloc[random_indices]['Fatigue_UTS'] - merged_df_cleaned.iloc[random_indices]['σL (MPa)']) / merged_df_cleaned.iloc[random_indices]['σL (MPa)']
relative_errors_hb = (merged_df_cleaned.iloc[random_indices]['Fatigue_HB'] - merged_df_cleaned.iloc[random_indices]['σL (MPa)']) / merged_df_cleaned.iloc[random_indices]['σL (MPa)']
relative_errors_fpcm = (merged_df_cleaned.iloc[random_indices]['Fatigue_FPCM'] - merged_df_cleaned.iloc[random_indices]['σL (MPa)']) / merged_df_cleaned.iloc[random_indices]['σL (MPa)']
relative_errors_usm = (merged_df_cleaned.iloc[random_indices]['Fatigue_USM'] - merged_df_cleaned.iloc[random_indices]['σL (MPa)']) / merged_df_cleaned.iloc[random_indices]['σL (MPa)']

# Step 3: Plot line graph showing the variation of relative error for each model on separate subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 20))

# Plot for UTS
axes[0].plot(random_indices, relative_errors_uts, label='UTS')
axes[0].axhline(y=0, color='r', linestyle='-', label='0', linewidth=1)  # Add red horizontal line at y=0
axes[0].axhline(y=0.2, color='k', linestyle='-', label='+0.2', linewidth=1)
axes[0].axhline(y=-0.2, color='k', linestyle='-', label='-0.2', linewidth=1)
axes[0].axhline(y=0.4, color='k', linestyle='--', label='+0.4', linewidth=1)
axes[0].axhline(y=-0.4, color='k', linestyle='--', label='-0.4', linewidth=1)
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Relative Error')
axes[0].set_title('Relative Error Comparison for Fatigue_UTS')
axes[0].legend()

# Plot for HB
axes[1].plot(random_indices, relative_errors_hb, label='HB')
axes[1].axhline(y=0, color='r', linestyle='-', label='0', linewidth=1)  # Add red horizontal line at y=0
axes[1].axhline(y=0.2, color='k', linestyle='-', label='+0.2', linewidth=1)
axes[1].axhline(y=-0.2, color='k', linestyle='-', label='-0.2', linewidth=1)
axes[1].axhline(y=0.4, color='k', linestyle='--', label='+0.4', linewidth=1)
axes[1].axhline(y=-0.4, color='k', linestyle='--', label='-0.4', linewidth=1)
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Relative Error')
axes[1].set_title('Relative Error Comparison for Fatigue_HB')
axes[1].legend()

# Plot for FPCM
axes[2].plot(random_indices, relative_errors_fpcm, label='FPCM')
axes[2].axhline(y=0, color='r', linestyle='-', label='0', linewidth=1)  # Add red horizontal line at y=0
axes[2].axhline(y=0.2, color='k', linestyle='-', label='+0.2', linewidth=1)
axes[2].axhline(y=-0.2, color='k', linestyle='-', label='-0.2', linewidth=1)
axes[2].axhline(y=0.4, color='k', linestyle='--', label='+0.4', linewidth=1)
axes[2].axhline(y=-0.4, color='k', linestyle='--', label='-0.4', linewidth=1)
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Relative Error')
axes[2].set_title('Relative Error Comparison for Fatigue_FPCM')
axes[2].legend()

# Plot for USM
axes[3].plot(random_indices, relative_errors_usm, label='USM')
axes[3].axhline(y=0, color='r', linestyle='-', label='0', linewidth=1)  # Add red horizontal line at y=0
axes[3].axhline(y=0.2, color='k', linestyle='-', label='+0.2', linewidth=1)
axes[3].axhline(y=-0.2, color='k', linestyle='-', label='-0.2', linewidth=1)
axes[3].axhline(y=0.4, color='k', linestyle='--', label='+0.4', linewidth=1)
axes[3].axhline(y=-0.4, color='k', linestyle='--', label='-0.4', linewidth=1)
axes[3].set_xlabel('Index')
axes[3].set_ylabel('Relative Error')
axes[3].set_title('Relative Error Comparison for Fatigue_USM')
axes[3].legend()

plt.tight_layout()
plt.show()


# In[ ]:




