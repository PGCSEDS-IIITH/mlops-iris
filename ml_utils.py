from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = None

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    global clf
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # define a Gaussain NB classifier
    clf_nb = GaussianNB()
    #deffine a Decsion Tree classifier
    clf_dt = DecisionTreeClassifier(random_state=0)
    
    clf_nb.fit(X_train, y_train)
    clf_dt.fit(X_train, y_train)
    # calculate the print the accuracy score
    acc_nb = accuracy_score(y_test, clf_nb.predict(X_test))
    acc_dt = accuracy_score(y_test, clf_dt.predict(X_test))
    print(f"Model trained using naive-bayes classifier with accuracy: {round(acc_nb, 3)}")
    print(f"Model trained using decsion-tree classifier with accuracy: {round(acc_dt, 3)}")
    if acc_nb>acc_dt:
        print('loading model with naive-bayes classifier')
        clf = clf_nb
    else:
        print('loading model with decision-tree classfier')
        clf = clf_dt
    return clf

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
