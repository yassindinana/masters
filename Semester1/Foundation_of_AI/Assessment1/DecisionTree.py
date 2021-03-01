import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import sklearn.externals.six
from IPython.display import Image
import pydotplus

x1 = pd.read_csv("x_test.csv")
x2 = pd.read_csv("x_train.csv")
y1 = pd.read_csv("y_test.csv")
y2 = pd.read_csv("y_train.csv")

feature_cols = ['air_pressure_9am',
                'air_temp_9am',
                'avg_wind_speed_9am',
                'max_wind_speed_9am',
                'avg_wind_direction_9am',
                'rain_duration_9am',
                'rain_accumul'
                'ation_9am',
                'max_wind_direction_9am']

x_train, y_train = (x2, y2)
x_test, y_test = (x1, y1)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

goal = accuracy_score(y1, tree.predict(x1))


print("Result is  ", goal)

tree_model = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=4, min_samples_leaf=5)
tree_model.fit(x_train, y_train)

tree_prediction = tree.predict(x_test)
print("Predicted values are:")
print(tree_prediction)

print("Confusion Matrix: ",
      confusion_matrix(y_test, tree_prediction))

print ("Accuracy : ",
       accuracy_score(y_test, tree_prediction) * 100)

print("Report : ",
      classification_report(y_test, tree_prediction))

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')
Image(graph.create_png())
