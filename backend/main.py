from __future__ import annotations
from io import BytesIO
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, classification_report
)

app = FastAPI(title="FraudGraph AI API", version="1.0.0",
              description="Transaction fraud classification and graph analysis prototype")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

STATE = {"model": None, "scaler": None, "feature_columns": None, "scale_cols": None,
         "metrics": None, "report": None, "confusion_matrix": None, "dataset_info": None}


class PredictionRow(BaseModel):
    features: dict[str, float]

class GraphTransfer(BaseModel):
    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)
    amount: float = Field(ge=0)

class GraphRequest(BaseModel):
    transfers: list[GraphTransfer]

@app.get("/")
def root():
    return {"name": "FraudGraph AI API", "status": "ok", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok", "model_trained": STATE["model"] is not None}

@app.post("/api/model/train")
async def train_model(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")
    try:
        df = pd.read_csv(BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")
    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise HTTPException(400, "Dataset must contain Time, Amount, and Class columns.")
    if df.empty:
        raise HTTPException(400, "The uploaded dataset is empty.")
    if df["Class"].isna().any() or not set(df["Class"].unique()).issubset({0, 1}):
        raise HTTPException(400, "Class must contain only 0/1 labels without missing values.")
    if df[["Time", "Amount", "Class"]].isna().any().any():
        raise HTTPException(400, "Time, Amount, and Class cannot contain missing values.")
    if df["Class"].nunique() < 2 or df["Class"].value_counts().min() < 2:
        raise HTTPException(400, "At least two examples of each class are needed for a stratified split.")

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_cols = [c for c in ["Time", "Amount"] if c in X.columns]
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
    if scale_cols:
        X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                               solver="liblinear", random_state=42)
    try:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        score = model.predict_proba(X_test_scaled)[:, 1]
    except Exception as exc:
        raise HTTPException(400, f"Model training failed: {exc}")

    cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    report = classification_report(y_test, pred, labels=[0, 1],
        target_names=["Legitimate", "Fraud"], output_dict=True, zero_division=0)
    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, score)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "fraud_train": int(y_train.sum()), "fraud_test": int(y_test.sum()),
        "legitimate_train": int((y_train == 0).sum()),
        "legitimate_test": int((y_test == 0).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
    }
    STATE.update(model=model, scaler=scaler, feature_columns=list(X.columns),
                 scale_cols=scale_cols, metrics=metrics, report=report,
                 confusion_matrix=cm,
                 dataset_info={"filename": file.filename, "shape": list(df.shape),
                               "class_counts": {"legitimate": int((y == 0).sum()), "fraud": int(y.sum())}})
    return {"message": "Model trained successfully", "metrics": metrics,
            "confusion_matrix": cm, "report": report,
            "feature_columns": list(X.columns), "dataset_info": STATE["dataset_info"]}

@app.get("/api/model/status")
def model_status():
    return {"trained": STATE["model"] is not None,
            "feature_columns": STATE["feature_columns"],
            "metrics": STATE["metrics"], "dataset_info": STATE["dataset_info"],
            "report": STATE["report"], "confusion_matrix": STATE["confusion_matrix"]}

@app.post("/api/model/predict")
def predict(payload: PredictionRow):
    if STATE["model"] is None:
        raise HTTPException(409, "Train the model first by uploading the labeled dataset.")
    expected = STATE["feature_columns"]
    missing = set(expected) - set(payload.features)
    extra = set(payload.features) - set(expected)
    if missing or extra:
        raise HTTPException(400, f"Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}")
    try:
        row = pd.DataFrame([[payload.features[c] for c in expected]], columns=expected)
        scaled = row.copy()
        if STATE["scale_cols"]:
            scaled[STATE["scale_cols"]] = STATE["scaler"].transform(row[STATE["scale_cols"]])
        probability = float(STATE["model"].predict_proba(scaled)[0, 1])
        pred = int(probability >= 0.5)
        return {"predicted_class": pred, "label": "Potential fraud" if pred else "Likely legitimate",
                "fraud_probability": probability, "threshold": 0.5}
    except Exception as exc:
        raise HTTPException(400, f"Prediction failed: {exc}")

@app.post("/api/model/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if STATE["model"] is None:
        raise HTTPException(409, "Train the model first.")
    try:
        df = pd.read_csv(BytesIO(await file.read())).drop(columns=["Class"], errors="ignore")
        expected = STATE["feature_columns"]
        missing, extra = set(expected) - set(df.columns), set(df.columns) - set(expected)
        if missing or extra:
            raise HTTPException(400, f"Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}")
        df = df[expected]
        if df.isna().any().any():
            raise HTTPException(400, "Prediction CSV contains missing values.")
        scaled = df.copy()
        if STATE["scale_cols"]:
            scaled[STATE["scale_cols"]] = STATE["scaler"].transform(df[STATE["scale_cols"]])
        scores = STATE["model"].predict_proba(scaled)[:, 1]
        preds = (scores >= 0.5).astype(int)
        result = df.copy()
        result["predicted_class"] = preds
        result["fraud_probability"] = scores
        result["prediction"] = np.where(preds == 1, "Potential fraud", "Likely legitimate")
        return {"rows": result.head(500).replace({np.nan: None}).to_dict(orient="records"),
                "total_rows": int(len(result)), "truncated": len(result) > 500}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not score CSV: {exc}")

@app.get("/api/graph/demo")
def graph_demo():
    transfers = [
        ("A101","A102",18000), ("A102","A103",16500), ("A103","A104",15200),
        ("A104","A101",14100), ("A101","A105",800), ("A106","A107",1200),
        ("A107","A108",450), ("A109","A110",700), ("A111","A112",950),
        ("A112","A113",500), ("A114","A115",300), ("A115","A116",250),
        ("A103","A105",600), ("A117","A118",400), ("A118","A119",300)
    ]
    payload = GraphRequest(transfers=[GraphTransfer(sender=a, receiver=b, amount=c) for a, b, c in transfers])
    return analyze_graph(payload)

@app.post("/api/graph/analyze")
def analyze_graph(payload: GraphRequest):
    if not payload.transfers:
        raise HTTPException(400, "Add at least one transfer.")
    G = nx.DiGraph()
    clean = []
    for t in payload.transfers:
        if t.sender.strip() == t.receiver.strip():
            continue
        s, r = t.sender.strip(), t.receiver.strip()
        amount = float(t.amount)
        clean.append({"sender": s, "receiver": r, "amount": amount})
        if G.has_edge(s, r):
            G[s][r]["amount"] += amount
            G[s][r]["transactions"] += 1
        else:
            G.add_edge(s, r, amount=amount, transactions=1)
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "No valid transfer edges found.")
    cycle_nodes = set()
    for comp in nx.strongly_connected_components(G):
        if len(comp) > 1:
            cycle_nodes.update(comp)
    outgoing = {n: sum(G[n][v]["amount"] for v in G.successors(n)) for n in G.nodes()}
    incoming = {n: sum(G[u][n]["amount"] for u in G.predecessors(n)) for n in G.nodes()}
    values = np.array(list(outgoing.values()), dtype=float)
    threshold = float(np.quantile(values, 0.75)) if len(values) else 0
    records = []
    for n in G.nodes():
        reasons = []
        if n in cycle_nodes: reasons.append("Part of a directed cycle")
        if outgoing[n] >= threshold and outgoing[n] > 0:
            reasons.append("Outgoing transfer volume in top quartile")
        records.append({
            "id": n, "incoming_transfers": int(G.in_degree(n)),
            "outgoing_transfers": int(G.out_degree(n)),
            "incoming_amount": float(incoming[n]), "outgoing_amount": float(outgoing[n]),
            "flagged": bool(reasons), "reason": "; ".join(reasons) if reasons else "No demo rule triggered"
        })
    flagged = [r["id"] for r in records if r["flagged"]]
    nodes = [{"id": n, "flagged": n in flagged, "incoming": incoming[n], "outgoing": outgoing[n]} for n in G.nodes()]
    edges = [{"source": u, "target": v, "amount": float(d["amount"])} for u,v,d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "accounts": records, "transfers": clean,
            "summary": {"node_count": G.number_of_nodes(), "edge_count": G.number_of_edges(),
                        "flagged_count": len(flagged), "threshold": threshold},
            "note": "Exploratory rules on synthetic or user-supplied transfers; flags are not proof of fraud."}
