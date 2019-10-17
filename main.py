import pandas, numpy, sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle

# sep by ; due to csv file
data = pandas.read_csv("student-mat.csv", sep=";")

# trim data to parameters that we want
data = data[['G1', 'G2', 'G3', 'studytime', 'freetime', 'failures', 'absences']]
print(data.head())

# we will try to predict G3
predict = "G3"

x = numpy.array(data.drop(columns=[predict]))
y = numpy.array(data[predict])

# # create training and test data
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0
for _ in range(100):
    # create training and test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # instantiate linear regression model
    linear = linear_model.LinearRegression()

    # find best fit line on linear regression model on training data (train the model, basically)
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)  # Accuracy
    # print('Coeff: \n', linear.coef_)  # Coefficient
    # print('Intercept: \n', linear.intercept_)  # Y-intercept

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as output:
            pickle.dump(linear, output)

print(best)

with open("studentgrades.pickle", "rb") as input_data:
    linear = pickle.load(input_data)

# create predictions and compare to actual data
predictions = linear.predict(x_test)
for a in range(len(predictions)):
    print(predictions[a], x_test[a], y_test[a])
print(linear.score(x_test, y_test))
