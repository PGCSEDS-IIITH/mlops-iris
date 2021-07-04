from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
clf = GaussianNB()
rf = RandomForestClassifier(n_estimators=450)
model = None
max_acc = 0

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

def load_random_forest():
    X, y = datasets.load_iris(return_X_y=True)
    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf.fit(X_train, y_train)
    global max_acc
    global model
    # calculate the print the accuracy score
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    if acc_rf > max_acc:
        model = rf
    print(f"Random forest Model trained with accuracy: {round(acc_rf, 5)}")

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    global max_acc
    global model
    clf.fit(X_train, y_train)
    # calculate the print the accuracy score
    acc_nb = accuracy_score(y_test, clf.predict(X_test))
    if acc_nb > max_acc:
        model = clf
    print(f"GaussianNB Model trained with accuracy: {round(acc_nb, 5)}")


# function to predict the flower using the model
def predict(query_data):


    x = list(query_data.dict().values())
    print('predicting using model:', model)
    prediction = model.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    model.fit(X, y)
