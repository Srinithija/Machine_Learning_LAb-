import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#from pgmpy.models import BayesianNetwork
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

st.title("Heart Disease Diagnosis using Bayesian Network")

# Load dataset
data = pd.read_csv("heart.csv")

st.subheader("Heart Disease Dataset Sample")
st.dataframe(data.head())

# Convert continuous features to categorical
data['age'] = pd.cut(data['age'], bins=3, labels=[0,1,2])
data['chol'] = pd.cut(data['chol'], bins=3, labels=[0,1,2])
data['trestbps'] = pd.cut(data['trestbps'], bins=3, labels=[0,1,2])

data = data[['age','sex','cp','chol','trestbps','fbs','target']]

# Build Bayesian Network
model = DiscreteBayesianNetwork([
    ('age','target'),
    ('sex','target'),
    ('cp','target'),
    ('chol','target'),
    ('trestbps','target'),
    ('fbs','target')
])

# Train model
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference engine
infer = VariableElimination(model)

# -------------------------------
# Show Bayesian Network Graph
# -------------------------------

st.subheader("Bayesian Network Structure")

G = nx.DiGraph()
G.add_edges_from([
('age','target'),
('sex','target'),
('cp','target'),
('chol','target'),
('trestbps','target'),
('fbs','target')
])

fig, ax = plt.subplots()
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
st.pyplot(fig)

# -------------------------------
# User Input Section
# -------------------------------

st.subheader("Enter Patient Information")

age = st.selectbox("Age Category Low age : 29-45(0), Medium age : 46-60(1), High age : 61-80(2)", [0,1,2])
sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
cp = st.selectbox("Chest Pain Type 0->high, 1->moderate, 2->low, 3->none", [0,1,2,3])
chol = st.selectbox("Cholesterol Level Category0->low, 1->medium, 2->high", [0,1,2])
trestbps = st.selectbox("Blood Pressure Category ", [0,1,2])
fbs = st.selectbox("Fasting Blood Sugar >120 (1=True,0=False)", [0,1])

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Heart Disease"):

    result = infer.query(
        variables=['target'],
        evidence={
            'age':age,
            'sex':sex,
            'cp':cp,
            'chol':chol,
            'trestbps':trestbps,
            'fbs':fbs
        }
    )

    prob_no = result.values[0]
    prob_yes = result.values[1]

    st.subheader("Prediction Result")

    st.write(f"Probability of No Heart Disease: {prob_no:.2f}")
    st.write(f"Probability of Heart Disease: {prob_yes:.2f}")

    if prob_yes > prob_no:
        st.error("Patient likely has Heart Disease")
    else:
        st.success("Patient likely does NOT have Heart Disease")