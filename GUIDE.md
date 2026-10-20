# Support Ticket Classifier: Complete Beginner Guide

## Part 1: Theory (read this first)

### The idea in one line
Text goes in, the model predicts a category, the result is saved in a database and shown on a web page.

### Machine learning in simple words
Normal programming: you write the rules.
Machine learning: you show examples and the computer finds the rules itself.

We show it 75 tickets with their correct categories. It notices that words like "refund", "charged", "invoice" appear in Billing tickets, and "tracking", "courier", "parcel" appear in Shipping tickets. Later, when it sees a new ticket, it guesses based on those patterns. This is called **supervised learning** (learning from labelled examples). The JD mentions this exact term.

### The pieces

| Term | Plain meaning |
|---|---|
| **Dataset** | The CSV file of tickets and their categories. |
| **Train/test split** | 75% of data teaches the model; 25% is hidden and used only to check it. |
| **TF-IDF** | Turns text into numbers. A word gets a high score if it is frequent in this ticket but rare across all tickets (so "refund" matters, "the" does not). |
| **Logistic Regression** | A simple model that learns which words push toward which category. Despite the name, it is used for classification. |
| **Accuracy** | Correct predictions divided by total predictions on the hidden test data. |
| **Pipeline** | Chains TF-IDF and the model so one `.fit()` and one `.predict()` do both steps. |
| **joblib / model.joblib** | Saves the trained model to a file so the web app doesn't retrain every time. |
| **Flask** | Python library for a small web server (like your C++ server, but it handles the sockets and parsing for you). |
| **Route** | A URL your server answers, like `/` or `/predict`. |
| **REST API** | A URL that programs call to get data, here `POST /predict` returns JSON. |
| **SQLite** | A database stored in one file (`tickets.db`). |
| **SQL** | The language to talk to a database: `CREATE TABLE`, `INSERT`, `SELECT`. |

### Why Logistic Regression and not a neural network?
Small dataset (100 rows), simple text task. A neural network would overfit and is harder to explain. Logistic Regression is fast, simple, and a standard baseline.

### Walkthrough of `train.py`
1. `pd.read_csv` loads the CSV.
2. `train_test_split` splits into train and test (`stratify` keeps categories balanced, `random_state=42` makes it repeatable).
3. `Pipeline([TfidfVectorizer, LogisticRegression])` builds the model. `ngram_range=(1,2)` also looks at word pairs like "charged twice".
4. `model.fit(X_train, y_train)` is the learning step.
5. `model.predict(X_test)` guesses the test tickets; `accuracy_score` compares guesses with true answers.
6. `joblib.dump` saves the model.

### Walkthrough of `app.py`
1. `joblib.load` loads the saved model once at start.
2. `init_db()` creates the `tickets` table if it doesn't exist.
3. `classify_and_save(text)` predicts, then runs `INSERT INTO tickets ...` to save it. The `?` marks are placeholders, which prevent SQL injection.
4. Route `/` shows the page; on form submit it classifies and shows the result plus the last 20 rows (`SELECT ... ORDER BY id DESC LIMIT 20`).
5. Route `/predict` accepts JSON and returns JSON (the REST API).

---

## Part 2: Run it on your computer

1. Install Python 3.9+ from python.org (tick "Add Python to PATH" on Windows).
2. Open a terminal in the project folder.
3. Create a virtual environment (an isolated space for the libraries):
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `python train.py` prints your **real accuracy**. Use that number in your resume.
6. `python app.py` then open http://127.0.0.1:5000 and try some tickets.
7. `pytest` runs the 3 tests.

My run printed 76%. Yours may differ slightly by library version. **Use your own number.**

---

## Part 3: Upload to GitHub, step by step

### One-time setup
1. Install Git from git-scm.com.
2. Tell Git who you are (once):
   ```
   git config --global user.name "Prem Shankar Upadhyay"
   git config --global user.email "premsupadhyay17@gmail.com"
   ```

### Create the repository
3. Go to github.com and log in (account: prems17).
4. Click the **+** (top right) then **New repository**.
5. Repository name: `ticket-classifier` (must match the link on your resume).
6. Set it to **Public**. Do NOT tick "Add a README" or .gitignore (we already have them).
7. Click **Create repository**.

### Push your code
8. In your terminal, inside the project folder:
   ```
   git init
   git add .
   git commit -m "Add support ticket classifier"
   git branch -M main
   git remote add origin https://github.com/prems17/ticket-classifier.git
   git push -u origin main
   ```
9. If asked to log in, GitHub no longer accepts your password in the terminal. Use a **Personal Access Token**: GitHub, Settings, Developer settings, Personal access tokens, Tokens (classic), Generate new token, tick `repo`, copy it, and paste it when asked for the password. (Or install GitHub Desktop and use the buttons instead.)
10. Refresh the repo page. You should see all files and the README.

### Check before applying
- The link `github.com/prems17/ticket-classifier` opens.
- The README is visible.
- `venv/`, `tickets.db` and `model.joblib` are not uploaded (the `.gitignore` handles this). People run `python train.py` to recreate the model.
- Optional: add a screenshot of the web page to the README.

### Later changes
```
git add .
git commit -m "Describe your change"
git push
```

---

## Part 4: Interview questions and answers

**30-second pitch:** "Support teams get hundreds of tickets. I built a system that reads the ticket text, predicts the category using TF-IDF and Logistic Regression, saves it in a SQLite database and shows it on a Flask web page."

**What is TF-IDF?** It converts text to numbers, giving high weight to words that are important in one document but rare across all documents.

**Why Logistic Regression?** Small dataset, simple, fast and explainable. A neural network would be overkill.

**How did you measure accuracy?** I split the data 75/25, trained on 75%, and measured correct predictions on the hidden 25%.

**Why only about 76%?** The dataset is small (100 hand-written tickets). More and more varied data would improve it. I would also try cross-validation.

**What is overfitting?** When a model memorises the training data and does badly on new data. The hidden test set is how I check for it.

**What is a REST API?** A URL that programs call over HTTP, here `POST /predict` takes JSON text and returns the category as JSON.

**What is SQL injection and how did you avoid it?** Attacking a database by sneaking SQL into input. I used `?` placeholders so input is treated as data, not code.

**How would you scale it?** Move SQLite to PostgreSQL, run with gunicorn, use a larger dataset, and deploy on AWS/Render.

**Hard question you should expect:** "Did you write the dataset yourself?" Answer honestly: yes, a small hand-written sample, and it could be replaced with a larger Kaggle dataset.

---

## Part 5: Resume check
- Replace `[XX]` with your number from `python train.py` (round to whole number).
- The resume bullet about SQLite and the web interface is true because this code does it. Keep the "pytest" idea off the resume unless you want to add "Wrote pytest tests" (the 3 tests are included).
- Read every file once and be able to explain each line before the interview. Type the code yourself if you can, rather than only copy-pasting.
