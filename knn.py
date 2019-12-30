from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataset_loader import DatasetLoader
from imutils import paths
import argparse

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of CPU cores used for classification")
args = vars(ap.parse_args())

# Load images
print("[INFO] Loading images")
image_paths = list(paths.list_images(args["dataset"]))
sdl = DatasetLoader()
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 32*32*3))

# Encode labels as intergers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data.
# training: 75%, testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Evaluate k-NN classifier
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"], weights = 'distance')
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))