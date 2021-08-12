from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
gaussian_object = GaussianNB()
random_forest_object = RandomForestClassifier(random_state=2)
decision_tree_object = DecisionTreeClassifier()

best_model = None

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}


# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
    gaussian_object.fit(X_train, y_train)
    random_forest_object.fit(X_train, y_train)
    decision_tree_object.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc_gaussion = round(accuracy_score(y_test, gaussian_object.predict(X_test)),3)
    print(f"GaussianNB Model trained with accuracy: {acc_gaussion}")
    
    acc_random_forest = round(accuracy_score(y_test, random_forest_object.predict(X_test)))
    print(f"RandomForestClassifier Model trained with accuracy: {acc_random_forest}")

    acc_decision_tree = round(accuracy_score(y_test, decision_tree_object.predict(X_test)))
    print(f"DecisionTreeClassifier Model trained with accuracy: {acc_decision_tree}")    

    model_accuracy_dict = { gaussian_object: acc_gaussion,
                           random_forest_object: acc_random_forest,
                           decision_tree_object: acc_decision_tree,
                           }
    global best_model
    best_model = max(model_accuracy_dict, key=model_accuracy_dict.get)
    print(f'Best model to predict is: {best_model}')
    #return best_model


    
# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    
    print(f'Best model to evaluate is: {best_model}')
    prediction = best_model.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
        
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    gaussian_object.fit(X, y)
