# Sample dataset banata hai (same columns jo UCI Diabetes 130-US Hospitals mein hain)
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 8000
common = ["250.00","428","414","401","486","427","276","599","786","715","250.01","434"]
rare = [str(c) for c in rng.integers(100, 999, 120)]
diag = np.where(rng.random(n) < 0.85, rng.choice(common, n), rng.choice(rare, n))
df = pd.DataFrame({
    "encounter_id": np.arange(n) + 1000, "patient_nbr": rng.integers(10000, 99999, n),
    "age": rng.choice([f"[{i}-{i+10})" for i in range(0, 100, 10)], n, p=[.01,.01,.02,.04,.09,.17,.22,.26,.15,.03]),
    "gender": rng.choice(["Female","Male"], n),
    "time_in_hospital": rng.integers(1, 15, n), "num_lab_procedures": rng.integers(1, 100, n),
    "num_procedures": rng.integers(0, 7, n), "num_medications": rng.integers(1, 40, n),
    "number_outpatient": rng.poisson(0.4, n), "number_emergency": rng.poisson(0.2, n),
    "number_inpatient": rng.poisson(0.6, n), "number_diagnoses": rng.integers(3, 12, n),
    "diag_1": diag, "insulin": rng.choice(["No","Steady","Up","Down"], n),
    "change": rng.choice(["No","Ch"], n), "diabetesMed": rng.choice(["Yes","No"], n, p=[.77,.23]),
    "A1Cresult": rng.choice(["None","Norm",">7",">8"], n, p=[.8,.05,.05,.1]),
})
score = (0.5*df.number_inpatient + 0.3*df.number_emergency + 0.05*df.time_in_hospital
         + 0.02*df.num_medications + 0.15*df.number_diagnoses/3 + 0.2*(df.insulin.isin(["Up","Down"]))
         + 0.15*(df.change == "Ch") + rng.normal(0, 0.8, n))
r = score.rank(pct=True)
df["readmitted"] = np.where(r > 0.89, "<30", np.where(r > 0.54, ">30", "NO"))
df.loc[rng.random(n) < 0.03, "diag_1"] = "?"          # missing values jaisa asli data mein hota hai
df.to_csv("data/diabetes_datasets.csv", index=False)
print("Saved data/diabetes_datasets.csv", df.shape)
