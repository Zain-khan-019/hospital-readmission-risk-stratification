# Step 24-28: cleaning, transformation, 3 models, evaluation
import json, os, joblib, numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications",
       "number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
CAT = ["age","gender","diag_1","insulin","change","diabetesMed","A1Cresult"]

# Step 24: Cleaning
df = pd.read_csv("data/diabetes_datasets.csv").replace("?", np.nan)
df = df.drop(columns=[c for c in ["encounter_id","patient_nbr"] if c in df.columns])  # IDs hatao
df["diag_1"] = df["diag_1"].fillna("Missing").astype(str)
top = df["diag_1"].value_counts().head(15).index.tolist()                 # rare codes -> Other
df["diag_1"] = df["diag_1"].where(df["diag_1"].isin(top), "Other")
for c in CAT: df[c] = df[c].fillna("Unknown").astype(str)

X, y = df[NUM + CAT], df["readmitted"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 25: Transformation (encode + standardize)
def pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

# Step 26-27: Models (lbfgs solver by default multinomial hota hai)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1),
}
fitted, out = {}, {"diag_top": top + ["Other"], "models": {}}
summary = []

for name, m in models.items():
    p = Pipeline([("pre", pre()), ("m", m)]).fit(Xtr, ytr)
    pred = p.predict(Xte); classes = list(p.classes_)
    out["classes"] = classes

    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, average="macro")
    wf1 = f1_score(yte, pred, average="weighted")

    out["models"][name] = {
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "report": classification_report(yte, pred, output_dict=True, zero_division=0),
        "cm": confusion_matrix(yte, pred, labels=classes).tolist()}
    fitted[name] = p
    summary.append([name, round(acc, 3), round(mf1, 3), round(wf1, 3)])

    print("=" * 55)
    print(name)
    print(f"Accuracy    : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"Macro F1    : {mf1:.3f}")
    print(f"Weighted F1 : {wf1:.3f}")
    print(classification_report(yte, pred, zero_division=0))

# Teeno models ka comparison table
print("=" * 55)
print("MODEL COMPARISON")
print(pd.DataFrame(summary, columns=["Model", "Accuracy", "Macro F1", "Weighted F1"]).to_string(index=False))

rf = fitted["Random Forest"]                                              # top risk factors
imp = rf.named_steps["m"].feature_importances_
names = rf.named_steps["pre"].get_feature_names_out()
out["top_features"] = [[n.split("__")[1], float(v)] for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:8]]

os.makedirs("model", exist_ok=True)
joblib.dump(fitted, "model/models.joblib")
json.dump(out, open("model/metrics.json", "w"))
print("Saved model/models.joblib and model/metrics.json")

#python train.py
#python app.py



