#!/usr/bin/env python
# coding: utf-8

# # Title: Problem for Covid - 19 Data Analysis Project using Python

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


URL = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"


# # 1. Import the dataset using Pandas from above mentioned url.

# In[7]:


df = pd.read_csv( "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv")


# In[8]:


df.head()


# # 2. High Level Data Understanding:

# ## a. Find no. of rows & columns in the dataset

# In[10]:


df.shape


# In[9]:


print(f"No of rows : {df.shape[0]}")


# In[10]:


print(f"No of columns : {df.shape[1]}")


# ## b. Data types of columns.

# In[11]:


df.dtypes


# ## c. Info & describe of data in dataframe.

# In[12]:


df.info()


# In[13]:


df.describe()


# # 3. Low Level Data Understanding :

# ## a. Find count of unique values in location column.

# In[14]:


df["location"].nunique()


# ## b. Find which continent has maximum frequency using values counts.

# In[15]:


df["continent"].value_counts()


# ## c. Find maximum & mean value in 'total_cases'.

# In[16]:


df["total_cases"].max()


# In[17]:


df["total_cases"].min()


# In[18]:


df["total_cases"].mean()


# ## d. Find 25%,50% & 75% quartile value in 'total_deaths'.

# In[19]:


df["total_deaths"].quantile(0.25)


# In[20]:


df["total_deaths"].quantile(0.5)


# In[21]:


df["total_deaths"].quantile(0.75)


# ## e. Find which continent has maximum 'human_development_index'.

# In[22]:


df.iloc[df["human_development_index"].idxmax()]["continent"]


# ## f. Find which continent has minimum 'gdp_per_capita'.

# In[23]:


df.iloc[df["gdp_per_capita"].idxmin()]["continent"]


# # 4. Filter the dataframe with only this columns ['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index'] and update the data frame.

# In[24]:


df = df[['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index']]


# In[25]:


df.head()


# # 5. Data Cleaning

# ## a. Remove all duplicates observations

# In[26]:


df.drop_duplicates(inplace=True)


# In[27]:


df.shape


# ## b. Find missing values in all columns

# In[28]:


df.isnull().sum()


# ## c. Remove all observations where continent column value is missing

# In[29]:


df.dropna(subset="continent",inplace= True)


# In[30]:


df.shape


# ## d. Fill all missing values with 0

# In[31]:


df.fillna(value=0,inplace=True)


# # 6. Date time format :

# ## a. Convert date column in datetime format using pandas.to_datetime

# In[32]:


pd.to_datetime(df["date"])


# ## b. Create new column month after extracting month data from date column.

# In[33]:


df["month"] = pd.to_datetime(df["date"]).dt.month


# In[34]:


df.head()


# # 7. Data Aggregation:

# ## a. Find max value in all columns using groupby function on 'continent' column

# In[35]:


df.groupby("continent").max().reset_index()


# ## b. Store the result in a new dataframe named 'df_groupby'.

# In[36]:


df_groupby = df.groupby("continent").max().reset_index()


# In[37]:


df_groupby.head()


# # 8. Feature Engineering :

# ## a. Create a new feature 'total_deaths_to_total_cases' by ratio of 'total_deaths' column to 'total_cases'

# In[38]:


df_groupby["total_deaths"]/df_groupby["total_cases"]


# In[39]:


df_groupby["total_deaths_to_total_cases"] = df_groupby["total_deaths"]/df_groupby["total_cases"]


# In[40]:


df_groupby


# # 9. Data Visualization :

# ## a. Perform Univariate analysis on 'gdp_per_capita' column by plotting histogram using seaborn dist plot.

# In[56]:


sns.set_theme()


# In[57]:


sns.distplot(df_groupby["gdp_per_capita"],bins=6,rug=True)


# ## b. Plot a scatter plot of 'total_cases' & 'gdp_per_capita'

# In[48]:


sns.scatterplot(data=df_groupby,x="total_cases",y="gdp_per_capita",hue="continent",legend=False)


# ## c. Plot Pairplot on df_groupby dataset.

# In[51]:


sns.pairplot(df_groupby,diag_kind="kde")


# ## d. Plot a bar plot of 'continent' column with 'total_cases' .

# In[52]:


sns.catplot(data=df_groupby,x="continent",y="total_cases",kind="bar",palette="coolwarm")


# # 10.Save the df_groupby dataframe in your local drive using pandas.to_csv function .

# In[53]:


df_groupby.to_csv("Data_groupby")


# In[ ]:





# In[ ]:




