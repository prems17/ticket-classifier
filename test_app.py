# test_app.py - run with:  pytest
from app import app

def test_predict_works():
    client = app.test_client()
    res = client.post("/predict", json={"text": "I was charged twice, please refund"})
    assert res.status_code == 200
    assert res.get_json()["category"] == "Billing"

def test_predict_empty_text():
    client = app.test_client()
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 400

def test_home_page_loads():
    client = app.test_client()
    assert client.get("/").status_code == 200
