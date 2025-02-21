# Insurance-Fraud-Statistical-Analysis

![alt text](https://github.com/gaptab/Insurance-Fraud-Statistical-Analysis/blob/main/486.png)

1. Generating Data

To simulate a real-world fraud detection scenario, we create dummy data with key attributes such as:

Entities: Agent, Sales Manager (SM), Policy, Patient, and Hospital

Claim-related Features: Claim amount, number of visits, age, pre-existing conditions

ICD Codes: Standard disease classification codes

Fraud Labels: Whether a transaction was fraudulent (15% fraud cases)

2. Training Fraud Models for Each Entity

For each entity (Agent, SM, Policy, Patient, Hospital), a fraud detection model is trained using Logistic Regression, a widely used classification algorithm.

Features like claim amount, age, pre-existing conditions, and number of visits help in detecting fraud.

The model is trained on 80% of the data and tested on 20% to evaluate performance.

Accuracy and classification reports are generated for each entity's model.

3. Developing a Combined Fraud Model

After training separate fraud models for each entity:

Each model predicts fraud probability for its respective entity.

These fraud probabilities are averaged to create a final fraud score for each claim.

If the fraud score exceeds a threshold (e.g., 0.5), the claim is flagged as fraudulent.
This ensures that fraud detection is multi-layered, covering all involved entities.

4. Creating a Disease Network Model (ICD Codes)

To detect fraudulent claims based on unusual disease (ICD) codes, a graph-based approach is used:

A network graph is created, where nodes represent ICD codes.

Edges are added between codes that frequently appear together in fraudulent claims.

The network is visualized, showing relationships between different ICD codes.

5. Anomaly Detection Using Shortest Path

To detect anomalous ICD code patterns:

The shortest path algorithm is applied to the ICD network.

If two disease codes are connected but have an unusually long path length, it suggests an unexpected or fraudulent link.

A list of anomalous disease code pairs is generated.

For example, if a claim contains an unrelated combination of diseases, it could indicate fraud.
