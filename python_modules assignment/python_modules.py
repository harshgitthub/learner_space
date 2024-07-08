# -*- coding: utf-8 -*-
"""Python Modules.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10Mj-ju7t82lhKXkpO5gr9mwxzJ8MtrHV

# Numpy Implementation
"""

import numpy as np
# Create a 2D Numpy array of size 1x3 with elements of your choice
arr1=np.array([[1,2,3]])

# Create a Numpy array of length 50 with zeroes as its elements
arr2=np.zeros(50)

#Create a Numpy array of length 3x2 with elements of your choice
arr3=np.array([[1,2],[3,4],[5,6]])


#Multiply arr1 and arr3 using Numpy functions
arr4 = np.dot(arr1, arr3)

#Change 5th element of arr2 to a different number
#Your code here
arr2[4] = 4

if np.shape(arr4)==(1,2) and arr2[4]!=0:
  print("Passed")
else:
  print("Fail")

import numpy as np

#Task: Perform the dot product of I and 9I+1 using numpy, here I is referred to as an 3x3 Identity matrix.

"""# Pandas Implementation"""

import pandas as pd

## Create a DataFrame from a dictionary
data = {
    'Name': ['Ramesh', 'Mahesh', 'Suresh'],
    'Age': [25, 30, 35],
    'City': ['Bangalore', 'Mumbai', 'Delhi']
}
#Your code here
df = pd.DataFrame(data)
#Display the first 2 rows of the data frame
print(df.head(2))
#Your code here

#Print the age column
print(df['Age'])
#Your code here

#Filter rows where age is greater than 26
print(df[df['Age']>26])
#Your code here

#Add a new column 'Country' with the value 'India' for all rows
df['Country'] = 'India'
print(df.head())

data1 = {
    'Name': ['Ramesh', 'Mahesh', 'Suresh'],
    'Age': [25, None, 35],
    'City': ['Bangalore', 'Mumbai', 'Delhi']
}

df2= pd.DataFrame(data1)

# Fill missing values in the 'Age' column with the mean age
print(df2['Age'].mean())
df2['Age'].fillna(df2['Age'].mean(), inplace=True)
#Your code here

"""# Matplotlib Implementation"""

import matplotlib.pyplot as plt
import numpy as np


xpoints=np.array([1,2,3,4])
ypoints=np.array([2,4,6,8])

#Plot these points without drawing a line
plt.plot(xpoints , ypoints ,'o')
plt.show()
#Your code here

#Plotting with marker: Plot these points with a marker(Star marker)
plt.plot(xpoints , ypoints ,'*')
plt.show()
#Your code here

#Using fmt format, add circular marker,red color and Dashed line
plt.plot(xpoints , ypoints ,'o--r')
plt.show()
#Your code here

#Add xlabel,ylabel and title for the plot.
plt.plot(xpoints , ypoints )
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('My Plot')
plt.show()
#Your code here

colors  = np.array(["red", "green", "blue", "yellow"])
plt.scatter(xpoints , ypoints , c=colors)
plt.show()
#Your code here
#Create a scatter plot for xpoints and ypoints
#Your code here

#Set color to the scatter plot. Blue,Green,Red and yellow color for each point respectively

"""# Miscellaneous Modules Implementation"""



import random
import numpy as np

#Set the seed of random to 20
np.random.seed(20)

#Generate a random number between 0 and 1
np.random.random()

#Your code here

arr1=np.array([1,24,31,45,73,81,94,25])

#Using the random module pick 4 different random numbers from arr1 and return their sum.
np.random.choice(arr1,4)
#Your code here