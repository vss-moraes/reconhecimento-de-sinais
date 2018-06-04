import cv2
import sys
import numpy as np
from wekapy import *


def create_instance(features):
    instance = Instance()
    instance.add_features([
        Feature(name="feat1", value=features[0], possible_values="numeric"),
        Feature(name="feat2", value=features[1], possible_values="numeric"),
        Feature(name="feat3", value=features[2], possible_values="numeric"),
        Feature(name="feat4", value=features[3], possible_values="numeric"),
        Feature(name="feat5", value=features[4], possible_values="numeric"),
        Feature(name="feat6", value=features[5], possible_values="numeric"),
        Feature(name="feat7", value=features[6], possible_values="numeric"),
        Feature(name="class", value="?", possible_values="{A, B, C, D, E}")
    ])
    return instance


def hsv_thresh(img):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l1 = np.array((5, 38, 30))
    l2 = np.array((50, 250, 242))
    skinMask = cv2.inRange(frame, l1, l2)
    return skinMask

model = Model(classifier_type = "trees.RandomForest", classpath = "/usr/share/java/weka/weka.jar")
model.load_model("random_forest.model")

img = cv2.imread(sys.argv[1])
img = hsv_thresh(img)

hu = cv2.HuMoments(cv2.moments(img)).flatten()
hu = np.array(hu).tolist()

model.add_test_instance(create_instance(hu))
model.test()
print(model.predictions)

predictions = model.predictions
for prediction in predictions:
    print(prediction)
