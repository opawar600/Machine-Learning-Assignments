#!/usr/bin/env python
# coding: utf-8

# # ML Assignment 01 : Histograms, Boxplots, kNN

# In[94]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        import math
from numpy import linalg as LA


# ### Write a Python program to calculate the density estimator of a histogram.  Use the field x in the NormalSample.csv file. 

# In[95]:


df = pd.read_csv("/kaggle/input/normal/NormalSample.csv")
df.head(10)


# In[96]:


#max value of x
df.describe()


# ### According to Izenman (1991) method, what is the recommended bin-width for the histogram of x? 

# In[97]:


from scipy.stats import iqr
inter_quartile_range = iqr(df.x)
N = df.x.count()
bin_width = 2*inter_quartile_range*(pow(N,-1/3)) #bin width = 2(IQR)N^(-1/3)
print(bin_width)


# ### What are the minimum and the maximum values of the field x? 

# In[98]:


n = len(df.x)
min_x = df.x.min()
max_x = df.x.max()
print("Minimum = ",min_x,"\nMaximum = ",max_x)


# ### Use h = 2, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools. 

# In[99]:


#y = [0.4, 0.6, 0.7, 1.9, 2.4, 6.1, 6.2,7.3]
def get_midpoints(dfx,binw):
    x = dfx


    mini = math.floor(x.min())
    maxi = math.ceil(x.max())
    

    binw = binw     #Bin width

    midpoints = mini+binw/2

    no_of_bins = int((maxi-mini)/binw)
    print("Number of bins =",no_of_bins)
    start = mini+binw/2
    midps = [start]

    for i in range(0,no_of_bins-1):
        m = start+binw
        start = m
        midps.append(m)

    middf = pd.DataFrame(midps,columns = {"mi"}) #We get 100 midpoints and store it in a datframe
    return middf    #Midpoints


# In[100]:


def density_estimate(midpoint):
    m1 = midpoint
    density = 0
    for i in df.x:
        u = (i-m1)/binw
        if (u>-0.5 and u<=0.5):
            density+=1
    return density/(n*binw)


# In[101]:


def answer(df,binwidth):
    x = df.x
    binw = binwidth
    middf = get_midpoints(df.x,binw)
    lis = []
    for m in middf.mi:
        lis.append(density_estimate(m))

    middf["p(mi)"] = np.array(lis)
    #plt.step(middf.mi,middf["p(mi)"])
    return middf


# ### Use h = 2, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools. 

# In[102]:


binw = 2
ans_df2 = answer(df,binw)

plt.hist(x=df.x,range=(26,36), bins = 5,color = "dimgray") # h = 2
plt.title("Bin Width = 2")
plt.grid(axis='y', alpha=0.2)
plt.savefig("h2.png")
ans_df2


# ### Use h = 1, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools. 

# In[103]:


binw = 1
ans_df1 = answer(df,binw)
print(ans_df1)
plt.hist(x=df.x,range=(26,36), bins = 10,color = "dimgray") # h = 1
plt.title("Bin Width = 1")
plt.grid(axis='y', alpha=0.2)
plt.savefig("h1.png")
ans_df1


# ### Use h = 0.5, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools. 

# In[104]:


binw = 0.5
ans_df05 = answer (df,binw)
print("Co-ordinates of density \n",ans_df05)
plt.hist(x=df.x,range=(26,36), bins = 20,color = 'dimgray') # h = 0.5
plt.title("Bin Width = 0.5")
plt.grid(axis='y', alpha=0.2)
plt.savefig("h05.png")
ans_df05


# ### Use h = 0.1, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools. 

# In[105]:


binw = 0.1
ans_df01 = answer (df,binw)
ans_df01
plt.hist(x=df.x,range=(26,36), bins = 100,color = "dimgrey") # h = 2
plt.title("Bin Width = 0.1")
plt.grid(axis='y', alpha=0.2)
plt.savefig("h01.png")
ans_df01


# ### c)	(5 points) Draw a boxplot of x (without the group) using the Python boxplot function.  Can you tell if the Python’s boxplot has displayed the 1.5 IQR whiskers correctly?

# In[148]:


import seaborn as sns
plt.boxplot(df.x,vert = False)
plt.title("Boxplot of X")
plt.savefig("boxplot_x.png")


# ### Use in the NormalSample.csv to generate box-plots for answering the following questions.
# 
# ### a)	(5 points) What is the five-number summary of x?  What are the values of the 1.5 IQR whiskers?
# 

# In[109]:


from scipy.stats import iqr
inter_quartile_range = iqr(df.x)
q1 = np.percentile(df.x,25)
q3 = np.percentile(df.x,75)
l_whisker = q1 - 1.5*inter_quartile_range
u_whisker = q3 + 1.5*inter_quartile_range
print ("Lower Whisker = ",l_whisker,"\nUpper Whisker = ",u_whisker)


# ### b)	5 points) What is the five-number summary of x for each category of the group? What are the values of the 1.5 IQR whiskers for each category of the group?

# In[112]:


zero = []
ones = []
for i in range(0,df.x.count()):
    if df.group[i] == 0:
        zero.append(df.x[i])
    else:
        ones.append(df.x[i])
zeros = np.array(zero)
ones = np.array(ones)


# In[113]:


from scipy.stats import iqr
inter_quartile_range = iqr(zeros)
q1 = np.percentile(zeros,25)
q3 = np.percentile(zeros,75)
l_whisker = q1 - 1.5*inter_quartile_range
u_whisker = q3 + 1.5*inter_quartile_range
print ("Lower Whisker = ",l_whisker,"\nUpper Whisker = ",u_whisker)


# In[114]:


from scipy.stats import iqr
inter_quartile_range = iqr(ones)
q1 = np.percentile(ones,25)
q3 = np.percentile(ones,75)
l_whisker = q1 - 1.5*inter_quartile_range
u_whisker = q3 + 1.5*inter_quartile_range
print ("Lower Whisker = ",l_whisker,"\nUpper Whisker = ",u_whisker)


# In[115]:


plt.boxplot([ones,zero,df.x],widths = 0.4)
plt.xticks(np.arange(1,4),("Ones","Zeros","X"))
plt.title("BoxPlot For Each Category")
plt.savefig("CategoryBoxplot.png")


# In[116]:


df0 = df[df['group']==0]
df0.describe()


# In[117]:


df1 = df[df['group']==1]
df1.describe()


# In[118]:


fraud_df = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv")


# In[119]:


fraud_df.head()


# ### a)	(5 points) What percent of investigations are found to be fraudulent?  Please give your answer up to 4 decimal places.

# In[120]:


num_no_frauds = fraud_df.FRAUD.value_counts()[0]
num_of_frauds = fraud_df.FRAUD.value_counts()[1]
count = fraud_df.FRAUD.count()


# In[121]:


fraud_percent = num_of_frauds/count*100
print(round(fraud_percent,4))


# In[122]:


fraud_df.columns


# ### b)	(5 points) Use the BOXPLOT function to produce horizontal box-plots.  For each interval variable, one box-plot for the fraudulent observations, and another box-plot for the non-fraudulent observations.  These two box-plots must appear in the same graph for each interval variable.

# In[123]:


sns.boxplot(data = fraud_df , x = 'TOTAL_SPEND' , y = 'FRAUD',orient = 'h')
plt.title("Total Spent")
plt.savefig("TotalSpent.png")


# In[124]:


sns.boxplot(data = fraud_df , x = 'NUM_MEMBERS' , y = 'FRAUD',orient = 'h')
plt.title("Number of members covered")
plt.savefig("Num_members.png")


# In[125]:


sns.boxplot(data = fraud_df , x = 'MEMBER_DURATION' , y = 'FRAUD', orient = 'h')
plt.title("Member Duration")
plt.savefig("Mem_duration.png")


# In[126]:


sns.boxplot(data = fraud_df , x = 'OPTOM_PRESC' , y = 'FRAUD' , orient = 'h')
plt.title("Number of optical examinations")
plt.savefig("Optom_Presc.png")


# In[127]:


sns.boxplot(data = fraud_df , x = 'DOCTOR_VISITS' , y = 'FRAUD' ,orient = 'h')
plt.title("Number of visits to a doctor  ")
plt.savefig("Doctor_Visit.png")


# In[128]:


sns.boxplot(data = fraud_df , x = 'NUM_CLAIMS' , y = 'FRAUD',orient = 'h')
plt.title("Number of claims made recently ")
plt.savefig("Num_claims.png")


# In[129]:


fraud_df.head()


# ### c)	(10 points) Orthonormalize interval variables and use the resulting variables for the nearest neighbor analysis. Use only the dimensions whose corresponding eigenvalues are greater than one.
# ### i.	(5 points) How many dimensions are used?
# ### ii.	(5 points) Please provide the transformation matrix?  You must provide proof that the resulting variables are actually orthonormal.
# 

# In[130]:


fraud_df = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv",usecols=["TOTAL_SPEND","DOCTOR_VISITS","NUM_CLAIMS","MEMBER_DURATION","OPTOM_PRESC","NUM_MEMBERS"])


# In[131]:


x = np.matrix(fraud_df)


# In[132]:


xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)


# In[133]:


# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)

# Orthonormalize using the orth function 
import scipy
from scipy import linalg as LA2

orthx = LA2.orth(x)
print("The orthonormalize x = \n", orthx)

# Check columns of the ORTH function
check = orthx.transpose().dot(orthx)
print("Also Expect an Identity Matrix = \n", check)


# In[134]:


from sklearn.neighbors import KNeighborsClassifier


# In[135]:


target = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv",usecols=["FRAUD"])


# ### 10 points) Use the NearestNeighbors module to execute the Nearest Neighbors algorithm using exactly five neighbors and the resulting variables you have chosen in c).  The KNeighborsClassifier module has a score function.
# 
# ### i.	(5 points) Run the score function, provide the function return value
# ### ii.	(5 points) Explain the meaning of the score function return value.
# 

# In[136]:


from sklearn import metrics
knn = KNeighborsClassifier(n_neighbors=5, metric = "euclidean")
knn.fit(transf_x,target)
predictions = knn.predict(transf_x)
print(metrics.accuracy_score(target,predictions))


# ### e)	(5 points) For the observation which has these input variable values: TOTAL_SPEND = 7500, DOCTOR_VISITS = 15, NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2, find its five neighbors.  Please list their input variable values and the target values. Reminder: transform the input observation using the results in c) before finding the neighbors.

# In[137]:


test = [[7500,15,3,127,2,2]] * transf;
test


# In[138]:


neighs = knn.kneighbors(test,return_distance=False)
neighs


# In[139]:


fraudulent_df = pd.read_csv("/kaggle/input/fraudknn/Fraud.csv")
fraudulent_df.iloc[neighs[0][0:]]

