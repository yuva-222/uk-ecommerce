#!/usr/bin/env python
# coding: utf-8

# # Importing Libararies

# In[4]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
import gc
import datetime


# # Importing Dataset

# In[5]:


df = pd.read_csv("C:/Users/yuvak/OneDrive/Desktop/Ecommerce - UK Retailer.csv",encoding='unicode_escape')


# In[6]:


print(os.listdir())


# In[7]:


df.head()


# # Basic information about the data-EDA

# In[8]:


df.info()


# In[9]:


df.describe()


# # Finding the duplicate values

# In[10]:


df.duplicated().sum()


# In[11]:


df.drop_duplicates()


# # Identification of the unique values in the Represented columns.

# In[12]:


df['InvoiceNo'].unique()
df['CustomerID'].unique()
df['InvoiceDate'].unique()
df['Country'].unique()


# In[13]:


df.dtypes


# In[14]:


df.isnull().sum().sort_values(ascending=False)


# In[15]:


numeric_col =["Quantity","UnitPrice","CustomerID"]
categ_col = ["StockCode","Description","Country"]


# # 1. Perform Basic EDA
# 
# a. Boxplot – All Numeric Variables.

# In[16]:


for i in numeric_col:
    plt.boxplot(df[i])
    plt.title(i)
    plt.show()


# b. Histogram – All Numeric Variables.

# In[17]:


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.hist(df["UnitPrice"])
plt.title("HISTOGRAM FOR UNIT_PRICE")
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(2,1,2)
plt.hist(df["Quantity"])
plt.title("HISTOGRAM FOR QUANTITY")
plt.show()


# c. Distribution Plot – All Numeric Variables.

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.displot(df["UnitPrice"])
plt.title("DISTRIBUTION FOR UNIT_PRICE")
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(2,1,2)
sns.displot(df["Quantity"])
plt.title("DISTRIBUTION FOR QUANTITY")
plt.show()


# d. Aggregation for all numerical Columns.

# In[18]:


for i in numeric_col:
    print(i,sum(df[i]))


# e. Unique Values across all columns.

# In[19]:


col=df.columns
for i in col:
    print(i,df[i].nunique())


# f. Duplicate values across all columns.

# In[20]:


for i in col:
    print(i,df[i].duplicated().sum())


# g. Correlation – Heatmap - All Numeric Variables.

# In[21]:


sns.heatmap(df.corr(), annot = True)
plt.show()


# h. Regression Plot - All Numeric Variables.

# In[19]:


plt.figure(figsize=(10,10))
sns.regplot(data = df, x= "Quantity", y ="UnitPrice")
plt.title("PRICE VS QUANTITY RELATION")


# i. Bar Plot – Every Categorical Variable vs every Numerical Variable.

# In[25]:


plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
sns.barplot(data = df, x="Country", y="UnitPrice")
plt.xticks(rotation=90)
plt.title("COUNTRY VS PRICE")
plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,1,2)
sns.barplot(data = df, x="Country", y="Quantity")
plt.xticks(rotation=90)
plt.title("COUNTRY VS QUANTITY")


# j. Pair plot - All Numeric Variables.

# In[76]:


sns.pairplot(df,vars=["Quantity","UnitPrice","Month","Day"])
plt.show()


# k. Line chart to show the trend of data - All Numeric/Date Variables.

# In[85]:


plt.plot(df['Quantity'],df['InvoiceNo'])
plt.xlabel('Quantity')
plt.ylabel('InvoiceNo')
plt.show()


# In[ ]:


sales_per_hour = df.groupby("hh")["price"].sum().reset_index()
sales_per_day = df.groupby("day")["price"].sum().reset_index()
sales_per_month = df.groupby("month")["price"].sum().reset_index()


# In[ ]:


plt.figure(figsize=(40,8))
plt.subplot(1,3,1)
sns.lineplot(data=sales_per_month, x="month", y="price", marker = True, color = "red")
plt.title("SALES TRENDS ACROSS MONTH")

plt.figure(figsize=(40,8))
plt.subplot(1,3,2)
sns.lineplot(data=sales_per_day, x="day", y="price", marker = True, color = "red")
plt.title("SALES TRENDS ACROSS DAY")

plt.figure(figsize=(40,8))
plt.subplot(1,3,3)
sns.lineplot(data=sales_per_hour, x="hh", y="price", marker = True, color = "red")
plt.title("SALES TRENDS ACROSS HOUR");


# In[ ]:


quan_per_hour = df.groupby("hh")["Quantity"].sum().reset_index()
quan_per_day = df.groupby("day")["Quantity"].sum().reset_index()
quan_per_month = df.groupby("month")["Quantity"].sum().reset_index()


# In[ ]:


plt.figure(figsize=(40,8))
plt.subplot(1,3,1)
sns.lineplot(data=quan_per_month, x="month", y="Quantity", marker = True, color = "red")
plt.title("QUANTITY TRENDS ACROSS MONTH")

plt.figure(figsize=(40,8))
plt.subplot(1,3,2)
sns.lineplot(data=quan_per_day, x="day", y="Quantity", marker = True, color = "red")
plt.title("QUANTITY TRENDS ACROSS DAY")

plt.figure(figsize=(40,8))
plt.subplot(1,3,3)
sns.lineplot(data=quan_per_hour, x="hh", y="Quantity", marker = True, color = "red")
plt.title("QUANTITY TRENDS ACROSS HOUR");


# # i. Plot the skewness - All Numeric Variables.

# In[22]:


df['Skewed Data'] = pd.DataFrame(df.skew(axis=1,skipna=True))


# In[23]:


sns.histplot(df['Skewed Data'],bins=10);


# In[24]:


sns.distplot(df['Skewed Data'].head(), bins=10)


# # 2. Check for missing values in all columns and replace them with the appropriate metric.
# (Mean/Median/Mode)

# In[25]:


df.isnull().sum().sort_values(ascending=False)


# In[26]:


df['CustomerID'].fillna(df['CustomerID'].mode()[0],inplace=True)


# In[27]:


df['Description'].fillna(df['Description'].mode()[0],inplace=True)


# In[28]:


df[df["Description"].isna()]


# In[29]:


df.isnull().sum().sort_values(ascending=False)


# # 3. Remove duplicate rows.

# In[30]:


df.duplicated().sum()


# In[31]:


df.drop_duplicates(inplace = True)


# # 4. Remove rows which have negative values in Quantity column.

# In[32]:


temp=df[df["Quantity"]<0].index


# In[33]:


temp


# In[34]:


df.drop(labels = temp, inplace = True)


# In[35]:


df[df["Quantity"]<0]


# # 5. Add the columns - Month, Day and Hour for the invoice.

# In[36]:


df["Invoicedate"] = pd.to_datetime(df['InvoiceDate'])


# In[37]:


df.dtypes


# In[38]:


df["Day"]= pd.DatetimeIndex(df["InvoiceDate"]).day
df["Month"]= pd.DatetimeIndex(df["InvoiceDate"]).month
df["Year"]= pd.DatetimeIndex(df["InvoiceDate"]).year


# In[39]:


df.head()


# # 6. How many orders made by the customers?

# In[40]:


df.groupby(by=["CustomerID","Country"])['InvoiceNo'].count()


# # 7. TOP 5 customers with higher number of orders.

# In[41]:


df.groupby(by=["CustomerID","Country"])['InvoiceNo'].count().sort_values(ascending = False).head(5)


# # 8. How much money spent by the customers?

# In[42]:


df["amount"] = df["Quantity"]*df["UnitPrice"]
df["amount"].astype(int)


# In[43]:


total_spent = df.groupby(by=["CustomerID","Country"], as_index=False)['amount'].sum()
total_spent


# In[44]:


df["amount"].sum() 


# # 9. TOP 5 customers with highest money spent.

# In[45]:


total_spent.sort_values(by='amount', ascending=False).head()


# # 10. How many orders per month?

# In[46]:


order_pr_month = df.groupby(by="Month")["InvoiceNo"].count()
order_pr_month


# In[47]:


month=[x for x in range(1,13)]
data=order_pr_month.values


# In[48]:


data


# In[49]:


plt.pie(data,labels=month,autopct = '%1.1f%%')
plt.show()


# # 11. How many orders per day?

# In[50]:


df["hour"]= pd.DatetimeIndex(df["InvoiceDate"]).hour


# In[51]:


order_pr_day = df.groupby(by="Day")["InvoiceNo"].count()
order_pr_day


# # 13. How many orders for each country?

# In[52]:


df


# In[53]:


orderbycountry=df.groupby(by="Country")['InvoiceNo'].count()
orderbycountry


# In[54]:


df["Country"].nunique()


# # 14. Orders trend across months

# In[55]:


ax = df.groupby('InvoiceNo')['Month'].unique().value_counts().sort_index().plot(kind='bar',figsize=(16,4))
ax.set_xlabel('Month',fontsize=16)
ax.set_ylabel('Number of Orders',fontsize=16)
ax.set_title('Number of orders for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=12)
plt.show()


# # 15. How much money spent by each country?

# In[56]:


spent_country = df.groupby(by="Country")['amount'].sum()
spent_country


# In[59]:


spent_country =df.groupby(by="Country")['amount'].sum().reset_index(drop=True)
spent_country


# In[63]:


spent_country= spent_country.head()
spent_country


# In[ ]:


plt.figure(figsize=(40,30))
plt.plot(df['amount'],df['Country'])
plt.xlabel('amount')
plt.ylabel('Country')
plt.show()


# # Result for Ecommerce-uk-Retailer.
# 
# Top 10 orders made by the customers are UK, FINLAND, ITALY, NORWAY. Top 5 customers with higher number of orders are from UK, EIRE, UK,UK,UK (i.e, 17841, 14911, 14096, 12748, 14606)@ total = 5. Total amount spent by the countries is $9747747.933999998.
# Top 5 customers with highest money spent NETHERLANDS@279489.02, UK@256438.49, UK@187482.17, EIRE@132572.62, AUSTRALIA@123725.45.
# Here 38 unique countries have same orders. AND finally, the orders were increased in the 11th month @15% i.e, upto 3500 orders. 
# 
