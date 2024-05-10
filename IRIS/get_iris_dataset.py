from sklearn import datasets
import csv
import os

# Download the Iris dataset
iris = datasets.load_iris()

# Get the file directory
file_dir = os.path.dirname(os.path.realpath(__file__))
# Create data directory if it doesn't exist
if not os.path.exists(f"{file_dir}/data"):
    os.makedirs(f"{file_dir}/data")
    
header = iris.feature_names + ["target"]
class_labels = [0 if label == "setosa" else 1 if label == "versicolor" else 2 for label in iris.target_names]
data = [iris.data[i].tolist() + [class_labels[iris.target[i]]] for i in range(len(iris.data))]

data_binary = [row for row in data if row[-1] != 2]


# Write the data to a CSV file
with open(f"{file_dir}/data/iris.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
    
with open(f"{file_dir}/data/iris_binary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_binary)