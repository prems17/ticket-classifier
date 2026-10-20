# Support Ticket Classifier

A small web app that reads a customer support ticket and predicts its category
(Billing, Technical Issue, Account Access, Shipping and Delivery).

## How it works
1. `train.py` trains a TF-IDF + Logistic Regression model (scikit-learn) on `data/tickets.csv`.
2. `app.py` is a Flask web app that loads the model, predicts a category for new text,
   and saves every ticket and prediction in a SQLite database.
3. The home page has a text box and a history table. A JSON API is available at `POST /predict`.

## Tech
Python, scikit-learn, pandas, Flask, SQLite, HTML, pytest

## Run it
```
pip install -r requirements.txt
python train.py
python app.py
```
Open http://127.0.0.1:5000

## API example
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"I was charged twice\"}"
```

## Tests
```
pytest
```

## Notes
The dataset is a small hand-written sample of 100 tickets (25 per category), so accuracy is
modest. With a larger real dataset the same code would perform better.
