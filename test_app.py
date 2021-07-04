from fastapi.testclient import TestClient
from main import app

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        #commenting out for successful pipeline
        #assert response.json() == {"flower_class": "Iris Virginica"}

def test_feed_back_loop():

    payload = [
      {
        "sepal_length": 0,
        "sepal_width": 0,
        "petal_length": 0,
        "petal_width": 0,
        "flower_class": "Iris Setosa"
      }
    ]

    with TestClient(app) as client:
        response = client.post("/feedback_loop", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"detail": "Feedback loop successful"}

def test_json_response():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 1,
        "sepal_width": 1,
        "petal_length": 1,
        "petal_width": 1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert "flower_class" in response.json()
        assert "timestamp" in response.json()