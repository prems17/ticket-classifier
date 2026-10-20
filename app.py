# app.py - Flask web app: predicts a category and saves it in SQLite

import sqlite3
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
DB_FILE = "tickets.db"
model = joblib.load("model.joblib")


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               text TEXT NOT NULL,
               category TEXT NOT NULL,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.commit()
    conn.close()


def classify_and_save(text):
    category = model.predict([text])[0]
    conn = get_db()
    conn.execute("INSERT INTO tickets (text, category) VALUES (?, ?)", (text, category))
    conn.commit()
    conn.close()
    return category


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            result = classify_and_save(text)
    conn = get_db()
    history = conn.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    return render_template("index.html", result=result, history=history)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please send JSON like {\"text\": \"...\"}"}), 400
    return jsonify({"text": text, "category": classify_and_save(text)})


init_db()

if __name__ == "__main__":
    app.run(debug=True)
