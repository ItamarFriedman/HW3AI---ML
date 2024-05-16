from ID3.utils import *
from ID3.ID3_experiments import basic_experiment


def main():
    attributes_names, train_dataset, test_dataset = load_data_set("ID3")

    x_train = train_dataset.drop(["diagnosis"], axis=1)
    y_train = train_dataset["diagnosis"]

    x_test = test_dataset.drop(["diagnosis"], axis=1)
    y_test = test_dataset["diagnosis"]

    print("train acc")
    basic_experiment(x_train, y_train, x_train, y_train)
    print("test acc")
    basic_experiment(x_train, y_train, x_test, y_test)


if __name__=="__main__":
    main()