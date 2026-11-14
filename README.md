# FraudGraph AI — Full-stack website

This project contains a **React frontend** and a **FastAPI backend**.

## Features
- Responsive dashboard with project/team information
- Upload a labeled `creditcard.csv` dataset and train a Logistic Regression model
- Model evaluation metrics: precision, recall, F1, average precision, confusion matrix and classification report
- Upload transaction rows and download fraud predictions as CSV
- Analyze uploaded account-transfer CSVs or a built-in synthetic graph
- Graph visualization, flagged account table, explanations, CSV export
- FastAPI health endpoint and interactive API docs

## Requirements
- Python 3.10+
- Node.js 18+

## Start backend (Terminal 1)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Backend docs: http://127.0.0.1:8000/docs

## Start frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

The frontend expects the API at `http://127.0.0.1:8000` by default. To change it, create `frontend/.env`:
```env
VITE_API_URL=http://127.0.0.1:8000
```
Restart Vite after changing the environment file.

## Dataset
Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Upload it from the **Fraud Prediction** page and click **Train model**.

## Graph CSV format
Use headers `Sender,Receiver,Amount`:
```csv
Sender,Receiver,Amount
A101,A102,18000
A102,A103,16500
A103,A101,15000
```

## Important project limitation
The Kaggle transaction dataset has no account IDs or transfer edges. The graph module is separate and uses synthetic or user-supplied transfer records. Current graph flags are exploratory rules, not proof of fraud. The project does not currently include a trained GNN, authentication, or persistent database. Model state is held in backend memory and resets when the server restarts. No personal/team names are hard-coded into the interface.
