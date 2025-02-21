import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# 1. Generate Dummy Data for Fraud Detection
# -----------------------------
np.random.seed(42)
num_samples = 2000

data = pd.DataFrame({
    "agent_id": np.random.randint(1, 500, num_samples),
    "sm_id": np.random.randint(1, 300, num_samples),
    "policy_id": np.random.randint(1000, 5000, num_samples),
    "patient_id": np.random.randint(5000, 20000, num_samples),
    "hospital_id": np.random.randint(1, 100, num_samples),
    "claim_amount": np.random.randint(500, 50000, num_samples),
    "num_visits": np.random.randint(1, 15, num_samples),
    "age": np.random.randint(18, 90, num_samples),
    "pre_existing_conditions": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    "icd_code": np.random.choice(["A01", "B02", "C03", "D04", "E05"], num_samples),
    "fraud_flag": np.random.choice([0, 1], num_samples, p=[0.85, 0.15])  # 15% fraud cases
})

# -----------------------------
# 2. Train Fraud Models for Each Entity
# -----------------------------
entities = ["agent_id", "sm_id", "policy_id", "patient_id", "hospital_id"]
models = {}

for entity in entities:
    X = data[["claim_amount", "num_visits", "age", "pre_existing_conditions"]]
    y = data["fraud_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"Fraud Model for {entity}:")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    models[entity] = model

# -----------------------------
# 3. Combined Fraud Model for Claims
# -----------------------------
# Aggregating fraud probabilities from entity models
data["fraud_score"] = np.zeros(num_samples)

for entity in entities:
    entity_data = data[["claim_amount", "num_visits", "age", "pre_existing_conditions"]]
    entity_data_scaled = scaler.transform(entity_data)
    fraud_probs = models[entity].predict_proba(entity_data_scaled)[:, 1]
    data["fraud_score"] += fraud_probs

# Normalize fraud scores
data["fraud_score"] /= len(entities)
data["final_fraud_flag"] = (data["fraud_score"] > 0.5).astype(int)

# -----------------------------
# 4. Network Model for Disease Code (ICD)
# -----------------------------
# Create Graph of ICD Code Co-occurrence
G = nx.Graph()

for icd_code in data["icd_code"].unique():
    G.add_node(icd_code)

# Add edges between frequently co-occurring ICD codes
icd_pairs = data.groupby(["icd_code", "fraud_flag"]).size().reset_index(name="count")

for _, row in icd_pairs.iterrows():
    if row["count"] > 10:  # Threshold for meaningful connections
        G.add_edge(row["icd_code"], row["fraud_flag"], weight=1/row["count"])

# Visualize the Disease Network
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
plt.title("ICD Disease Network")
plt.show()

# -----------------------------
# 5. Anomaly Detection via Shortest Path
# -----------------------------
# Find shortest path anomalies in ICD Codes
shortest_paths = nx.shortest_path_length(G)

anomalies = []
for src, targets in shortest_paths.items():
    for tgt, path_len in targets.items():
        if path_len > 2:  # Threshold for anomaly
            anomalies.append((src, tgt, path_len))

anomalies_df = pd.DataFrame(anomalies, columns=["ICD_Code_1", "ICD_Code_2", "Path_Length"])
print("Anomalous Disease Code Combinations:\n", anomalies_df)
