import pandas as pd
from math import sqrt,exp,pi

#%%
filename='C:/Users/User/Desktop/iris_data.csv'
#filename='./iris_data.csv'
iris = pd.read_csv(filename)
        
#%%
# Only choose two classes
iris = iris.iloc[50:150]
iris.reset_index(inplace=True,drop=True)

# Only choose two features
iris = iris.drop(['sepal length (cm)','petal width (cm)'],axis=1)

#%%
unique = set(iris['class'])

separated = dict()
for i in unique:
    group = iris.groupby(iris['class'])
    separated[i] = group.get_group(i).drop('class',axis=1).values

#%%
# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent


#%%
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	return summaries

summaries = dict()
for class_value, rows in separated.items():
	summaries[class_value] = summarize_dataset(rows)
    
#%%
test = [3.4, 5.2]

total_rows = sum([summaries[label][0][2] for label in summaries])
probabilities = dict()

for class_value, class_summaries in summaries.items():
    # calculate prior
	probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        
    # calculate prior*likelihood
	for i in range(len(class_summaries)):
		m, std, _ = class_summaries[i]
		probabilities[class_value] *= calculate_probability(test[i], m, std)

print(probabilities)

