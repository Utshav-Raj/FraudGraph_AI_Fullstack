
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import math

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="FraudGraph AI API",
    version="1.0.0",
    description="Transaction fraud classification and graph analysis prototype"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Keep training data small for Render free tier
MAX_TRAIN_ROWS = 10000
MAX_PREDICT_ROWS = 10000
MAX_RESPONSE_ROWS = 500

# --------------------------------------------------
# APPLICATION STATE
# --------------------------------------------------

STATE = {
    "model": None,
    "scaler": None,
    "feature_columns": None,
    "scale_cols": None,
    "metrics": None,
    "report": None,
    "confusion_matrix": None,
    "dataset_info": None
}

# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class PredictionRow(BaseModel):
    features: dict[str, float]


class GraphTransfer(BaseModel):
    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)
    amount: float = Field(ge=0)


class GraphRequest(BaseModel):
    transfers: list[GraphTransfer]


# --------------------------------------------------
# BASIC ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "FraudGraph AI API",
        "status": "ok",
        "docs": "/docs"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_trained": STATE["model"] is not None
    }


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

@app.post("/api/model/train")
async def train_model(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            400,
            "Please upload a CSV file."
        )

    try:
        import pandas as pd

        file.file.seek(0)

        # Read only a limited number of rows
        df = pd.read_csv(
            file.file,
            nrows=MAX_TRAIN_ROWS
        )

    except Exception as exc:
        raise HTTPException(
            400,
            f"Could not read CSV: {exc}"
        )

    required = {"Time", "Amount", "Class"}

    if not required.issubset(df.columns):
        raise HTTPException(
            400,
            "Dataset must contain Time, Amount, and Class columns."
        )

    if df.empty:
        raise HTTPException(
            400,
            "Dataset is empty."
        )

    if df["Class"].isna().any():
        raise HTTPException(
            400,
            "Class contains missing values."
        )

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise HTTPException(
            400,
            "Class must contain only 0 and 1."
        )

    if df[["Time", "Amount", "Class"]].isna().any().any():
        raise HTTPException(
            400,
            "Time, Amount and Class cannot contain missing values."
        )

    if df["Class"].nunique() < 2:
        raise HTTPException(
            400,
            "Dataset must contain both legitimate and fraud transactions."
        )

    if df["Class"].value_counts().min() < 2:
        raise HTTPException(
            400,
            "At least two examples of each class are required."
        )

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            confusion_matrix,
            classification_report
        )

        # Separate features and labels
        y = df.pop("Class").astype("int8")
        X = df

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scale Time and Amount
        scale_cols = [
            c for c in ["Time", "Amount"]
            if c in X.columns
        ]

        scaler = StandardScaler()

        if scale_cols:
            X_train.loc[:, scale_cols] = scaler.fit_transform(
                X_train[scale_cols]
            )

            X_test.loc[:, scale_cols] = scaler.transform(
                X_test[scale_cols]
            )

        # Train classification model
        model = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).tolist()

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "average_precision": float(
                average_precision_score(y_test, scores)
            ),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "fraud_train": int(y_train.sum()),
            "fraud_test": int(y_test.sum()),
            "legitimate_train": int((y_train == 0).sum()),
            "legitimate_test": int((y_test == 0).sum()),
            "missing_values": 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1] + 1)
        }

        STATE.update({
            "model": model,
            "scaler": scaler,
            "feature_columns": list(X.columns),
            "scale_cols": scale_cols,
            "metrics": metrics,
            "report": report,
            "confusion_matrix": cm,
            "dataset_info": {
                "filename": file.filename,
                "shape": [
                    int(df.shape[0]),
                    int(df.shape[1] + 1)
                ],
                "class_counts": {
                    "legitimate": int((y == 0).sum()),
                    "fraud": int(y.sum())
                }
            }
        })

        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "confusion_matrix": cm,
            "report": report,
            "feature_columns": list(X.columns),
            "dataset_info": STATE["dataset_info"]
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            400,
            f"Model training failed: {exc}"
        )


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

@app.get("/api/model/status")
def model_status():

    return {
        "trained": STATE["model"] is not None,
        "feature_columns": STATE["feature_columns"],
        "metrics": STATE["metrics"],
        "dataset_info": STATE["dataset_info"],
        "report": STATE["report"],
        "confusion_matrix": STATE["confusion_matrix"]
    }


# --------------------------------------------------
# SINGLE TRANSACTION PREDICTION
# --------------------------------------------------

@app.post("/api/model/predict")
def predict(payload: PredictionRow):

    if STATE["model"] is None:
        raise HTTPException(
            409,
            "Train the model first."
        )

    try:
        import pandas as pd

        expected = STATE["feature_columns"]

        missing = set(expected) - set(payload.features)
        extra = set(payload.features) - set(expected)

        if missing or extra:
            raise HTTPException(
                400,
                f"Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}"
            )

        row = pd.DataFrame(
            [[payload.features[c] for c in expected]],
            columns=expected
        )

        if STATE["scale_cols"]:
            row.loc[:, STATE["scale_cols"]] = (
                STATE["scaler"].transform(
                    row[STATE["scale_cols"]]
                )
            )

        probability = float(
            STATE["model"].predict_proba(row)[0, 1]
        )

        prediction = int(probability >= 0.5)

        return {
            "predicted_class": prediction,
            "label": (
                "Potential fraud"
                if prediction
                else "Likely legitimate"
            ),
            "fraud_probability": probability,
            "threshold": 0.5
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            400,
            f"Prediction failed: {exc}"
        )


# --------------------------------------------------
# CSV PREDICTION
# --------------------------------------------------

@app.post("/api/model/predict-csv")
async def predict_csv(file: UploadFile = File(...)):

    if STATE["model"] is None:
        raise HTTPException(
            409,
            "Train the model first."
        )

    try:
        import pandas as pd

        file.file.seek(0)

        df = pd.read_csv(
            file.file,
            nrows=MAX_PREDICT_ROWS
        ).drop(
            columns=["Class"],
            errors="ignore"
        )

        expected = STATE["feature_columns"]

        missing = set(expected) - set(df.columns)
        extra = set(df.columns) - set(expected)

        if missing or extra:
            raise HTTPException(
                400,
                f"Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}"
            )

        df = df[expected]

        if df.isna().any().any():
            raise HTTPException(
                400,
                "Prediction CSV contains missing values."
            )

        if STATE["scale_cols"]:
            df.loc[:, STATE["scale_cols"]] = (
                STATE["scaler"].transform(
                    df[STATE["scale_cols"]]
                )
            )

        scores = STATE["model"].predict_proba(df)[:, 1]

        predictions = [
            int(score >= 0.5) for score in scores
        ]

        rows = []

        for index, (_, record) in enumerate(df.head(MAX_RESPONSE_ROWS).iterrows()):

            item = record.to_dict()
            item["predicted_class"] = predictions[index]
            item["fraud_probability"] = float(scores[index])

            item["prediction"] = (
                "Potential fraud"
                if predictions[index] == 1
                else "Likely legitimate"
            )

            rows.append(item)

        return {
            "rows": rows,
            "total_rows": int(len(df)),
            "truncated": len(df) > MAX_RESPONSE_ROWS
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            400,
            f"Could not score CSV: {exc}"
        )


# --------------------------------------------------
# LIGHTWEIGHT GRAPH ANALYSIS
# --------------------------------------------------

def find_cycle_nodes(graph):

    index = 0
    stack = []
    on_stack = set()

    indices = {}
    lowlink = {}

    cycle_nodes = set()

    def strong_connect(node):

        nonlocal index

        indices[node] = index
        lowlink[node] = index

        index += 1

        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:

            if neighbor not in indices:

                strong_connect(neighbor)

                lowlink[node] = min(
                    lowlink[node],
                    lowlink[neighbor]
                )

            elif neighbor in on_stack:

                lowlink[node] = min(
                    lowlink[node],
                    indices[neighbor]
                )

        if lowlink[node] == indices[node]:

            component = []

            while True:

                current = stack.pop()
                on_stack.remove(current)

                component.append(current)

                if current == node:
                    break

            if len(component) > 1:
                cycle_nodes.update(component)

    for node in graph:

        if node not in indices:
            strong_connect(node)

    return cycle_nodes


def analyze_graph_data(transfers):

    graph = {}
    incoming = {}
    outgoing = {}
    edge_amounts = {}
    clean = []

    # Build adjacency data
    for transfer in transfers:

        sender = transfer["sender"].strip()
        receiver = transfer["receiver"].strip()

        amount = float(transfer["amount"])

        if not sender or not receiver or sender == receiver:
            continue

        clean.append({
            "sender": sender,
            "receiver": receiver,
            "amount": amount
        })

        graph.setdefault(sender, set())
        graph.setdefault(receiver, set())

        graph[sender].add(receiver)

        incoming.setdefault(sender, 0.0)
        incoming.setdefault(receiver, 0.0)

        outgoing.setdefault(sender, 0.0)
        outgoing.setdefault(receiver, 0.0)

        outgoing[sender] += amount
        incoming[receiver] += amount

        edge = (sender, receiver)

        edge_amounts[edge] = (
            edge_amounts.get(edge, 0.0) + amount
        )

    if not graph:
        raise HTTPException(
            400,
            "No valid transfer edges found."
        )

    # Detect directed cycles
    cycle_nodes = find_cycle_nodes(graph)

    # Calculate top quartile threshold
    values = sorted(outgoing.values())

    if values:

        position = max(
            0,
            math.ceil(0.75 * len(values)) - 1
        )

        threshold = float(values[position])

    else:
        threshold = 0.0

    records = []

    for node in graph:

        reasons = []

        if node in cycle_nodes:
            reasons.append("Part of a directed cycle")

        if outgoing[node] >= threshold and outgoing[node] > 0:
            reasons.append(
                "Outgoing transfer volume in top quartile"
            )

        records.append({
            "id": node,
            "incoming_transfers": sum(
                1 for u, v in edge_amounts if v == node
            ),
            "outgoing_transfers": sum(
                1 for u, v in edge_amounts if u == node
            ),
            "incoming_amount": float(incoming[node]),
            "outgoing_amount": float(outgoing[node]),
            "flagged": bool(reasons),
            "reason": (
                "; ".join(reasons)
                if reasons
                else "No demo rule triggered"
            )
        })

    flagged = [
        record["id"]
        for record in records
        if record["flagged"]
    ]

    nodes = [
        {
            "id": node,
            "flagged": node in flagged,
            "incoming": incoming[node],
            "outgoing": outgoing[node]
        }
        for node in graph
    ]

    edges = [
        {
            "source": sender,
            "target": receiver,
            "amount": amount
        }
        for (sender, receiver), amount in edge_amounts.items()
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "accounts": records,
        "transfers": clean,
        "summary": {
            "node_count": len(graph),
            "edge_count": len(edge_amounts),
            "flagged_count": len(flagged),
            "threshold": threshold
        },
        "note": (
            "Exploratory rules on synthetic or user-supplied transfers; "
            "flags are not proof of fraud."
        )
    }


# --------------------------------------------------
# GRAPH DEMO
# --------------------------------------------------

@app.get("/api/graph/demo")
def graph_demo():

    transfers = [
        ("A101", "A102", 18000),
        ("A102", "A103", 16500),
        ("A103", "A104", 15200),
        ("A104", "A101", 14100),
        ("A101", "A105", 800),
        ("A106", "A107", 1200),
        ("A107", "A108", 450),
        ("A109", "A110", 700),
        ("A111", "A112", 950),
        ("A112", "A113", 500),
        ("A114", "A115", 300),
        ("A115", "A116", 250),
        ("A103", "A105", 600),
        ("A117", "A118", 400),
        ("A118", "A119", 300)
    ]

    payload = [
        {
            "sender": sender,
            "receiver": receiver,
            "amount": amount
        }
        for sender, receiver, amount in transfers
    ]

    return analyze_graph_data(payload)


# --------------------------------------------------
# CUSTOM GRAPH ANALYSIS
# --------------------------------------------------

@app.post("/api/graph/analyze")
def analyze_graph(payload: GraphRequest):

    if not payload.transfers:
        raise HTTPException(
            400,
            "Add at least one transfer."
        )

    transfers = [
        {
            "sender": transfer.sender,
            "receiver": transfer.receiver,
            "amount": float(transfer.amount)
        }
        for transfer in payload.transfers
    ]

    return analyze_graph_data(transfers)
