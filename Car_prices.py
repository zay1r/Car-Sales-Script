#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the libraries that will be needed for this project
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


# In[4]:


car_data_2 = pd.read_csv(r'C:\Users\Isaiah\Desktop\Data Projects\Car_sale_data\used_car_sales.csv')


# In[5]:


#making a list of all the columns to compare it to my other dataset
column_list= car_data_2.columns.tolist()
print(column_list)


# In[6]:


min_year= car_data_2['Manufactured Year'].min()
max_year= car_data_2['Manufactured Year'].max()


# In[7]:


max_year


# In[8]:


min_year


# In[9]:


car_data_2.info


# In[22]:


print(car_data_2["Sold Date"].dtype)


# In[10]:


print(car_data_2['Sales Agent Name'].unique())


# In[26]:


car_data_2['Sold Date'] = pd.to_datetime(car_data_2['Sold Date'])  # Converts object type to datetime


# In[28]:


print(car_data_2["Sold Date"].dtype)


# In[30]:


car_data_2.head()


# In[36]:


# Function to generate a random date within the given range
#def random_date(start_date, end_date):
  #  return start_date + pd.to_timedelta(np.random.randint((end_date - start_date).days + 1), unit='D')

# Define date range
#start_date = pd.to_datetime("2023-01-01")
#end_date = pd.to_datetime("2024-12-31")

# Replace '1970-01-01' with a random date in the range
#car_data_2.loc[car_data_2['Sold Date'] == pd.to_datetime("1970-01-01"), 'Sold Date'] = car_data_2.loc[car_data_2['Sold Date'] == pd.to_datetime("1970-01-01"), 'Sold Date'].apply(lambda x: random_date(start_date, end_date))




# In[68]:


# Function to generate a random date within the given range
def random_date(start_date, end_date):
    return start_date + pd.to_timedelta(np.random.randint((end_date - start_date).days + 1), unit='D')

# Define the new date range
start_date = pd.to_datetime("2015-01-01")
end_date = pd.to_datetime("2024-12-31")

# Replace **all** values in 'Sold Date' with a random date in the range
car_data_2["Sold Date"] = car_data_2["Sold Date"].apply(lambda x: random_date(start_date, end_date))



# In[120]:


# Convert 'Purchased Date' and 'Sold Date' to datetime
car_data_2["Purchased Date"] = pd.to_datetime(car_data_2["Purchased Date"])
car_data_2["Sold Date"] = pd.to_datetime(car_data_2["Sold Date"])


# In[122]:


import pandas as pd
import numpy as np

# Function to generate a random date within the valid range
def random_date(purchased_date, max_date):
    start_date = purchased_date  # Ensure Sold Date is not before Purchased Date
    end_date = min(max_date, pd.to_datetime("2024-12-31"))  # Ensure it does not exceed final dataset date
    return start_date + pd.to_timedelta(np.random.randint((end_date - start_date).days + 1), unit='D')

# Define the maximum allowed date
max_date = pd.to_datetime("2024-12-31")

# Replace all values in 'Sold Date' with a random date that falls after 'Purchased Date' but before '2024-12-31'
car_data_2["Sold Date"] = car_data_2.apply(lambda row: random_date(row["Purchased Date"], max_date), axis=1)

#print(car_data_2[["Purchased Date", "Sold Date"]])


# In[124]:


car_data_2.head(10)


# In[126]:


def random_price(row):
    lower_bound = row['Purchased Price-$'] - 300
    upper_bound = row['Price-$'] + 1000
    return np.random.randint(lower_bound, upper_bound + 1)

# Apply function where 'Sold Price-$' is 0
#car_data_2.loc[car_data_2['Sold Price-$'] == 0, 'Sold Price-$'] = car_data_2[car_data_2['Sold Price-$'] == 0].apply(random_price, axis=1)

car_data_2["Sold Price-$"] = car_data_2.apply(random_price, axis=1)



# In[104]:


car_data_2.head(10)


# In[ ]:


#Profit margin 

#car_data_2.loc[car_data_2['Margin-%'] == 0, 'Margin-%'] = (([car_data_2['Sold Price-$']-[car_data_2['Purchased Price-$'])/[car_data_2['Purchased Price-$'])*100


#car_data_2["Margin-%"] = ((car_data_2["Sold Price-$"] - car_data_2["Purchased Price-$"]) / car_data_2["Sold Price-$"]) * 100


# In[128]:


car_data_2["Margin-%"] = ((car_data_2["Sold Price-$"] - car_data_2["Purchased Price-$"]) / car_data_2["Sold Price-$"]) * 100

# Round up to the nearest whole number
car_data_2["Margin-%"] = car_data_2["Margin-%"].apply(lambda x: np.ceil(x))



# In[108]:


car_data_2.head()


# In[130]:


def calculate_commission(row):
    commission = 0.2 * (row["Sold Price-$"] - row["Purchased Price-$"])
    return max(commission, 0)  # Ensure commission is not negative

# Apply the function to the column
car_data_2["Sales Commission-$"] = car_data_2.apply(calculate_commission, axis=1)

# Save the updated CSV
#car_data_2.to_csv(r'C:\Users\Isaiah\Desktop\Data Projects\Car_sale_data\updated_car_sales.csv', index=False)

print(car_data_2[["Sold Price-$", "Purchased Price-$", "Sales Commission-$"]])


# In[132]:


car_data_2["Sales Commission-$"] = car_data_2["Sales Commission-$"].round(2)


# In[114]:


car_data_2.head()


# In[134]:


car_data_2.to_csv(r'C:\Users\Isaiah\Desktop\Data Projects\Car_sale_data\updated_car_sales_4.csv', index=False)


# In[ ]:




