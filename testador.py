import sys
from wekapy import Model, Instance, Feature
from extraction import extract_features


def create_instance(file):
    features = extract_features(file)
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


def main():
    model = Model(classifier_type = "trees.RandomForest", classpath = "/usr/share/java/weka/weka.jar")
    model.load_model("random_forest.model")

    model.add_test_instance(create_instance(sys.argv[1]))
    model.test()
    print(model.predictions)

    predictions = model.predictions
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    main()
