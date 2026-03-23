# 🏥 Population Health Risk Stratifier

An interactive machine learning application that segments a synthetic patient population by health risk, identifies the key drivers of hospitalization, and surfaces the social determinants that layer onto clinical risk.

**Live Demo:** [View App](#) ← replace with your Streamlit URL

---

## Why I Built This

I spent a decade doing this work inside two of the largest health insurers in America. The goal was almost always the same: find the people headed toward a bad outcome before it happens, understand what is driving it, and give someone the information they need to intervene.

Most portfolio projects in data science show you can run a model. This one shows you understand what the model is actually for.

---

## What It Does

The app generates a synthetic population of 2,000 patients with realistic clinical and social health profiles, runs a clustering algorithm to find behavioral and clinical segments, trains a gradient boosting model to predict hospitalization probability, and presents everything in an interactive dashboard.

You can filter the population by age range, risk tier, and insurance type using the sidebar and watch every chart update in real time.

### Five Tabs

**Risk Segments** — how the population breaks down across four risk tiers (Low, Moderate, High, Critical) and the four segments identified through K-Means clustering. Includes a full segment profile table showing average age, risk score, chronic conditions, and hospitalization rate per segment.

**Key Drivers** — feature importance from the gradient boosting classifier showing what actually predicts hospitalization. Medication adherence, chronic conditions, and ED visit history consistently rank at the top.

**Hospitalization Risk** — predicted hospitalization probability by risk tier and segment, a scatter plot of risk score versus probability, and ED visit patterns broken out by tier.

**Social Determinants** — how food insecurity, housing instability, transportation barriers, and social isolation distribute across risk tiers. Also shows average risk score by insurance type and income level.

**Analysis** — a written explanation of the methodology, what the model is doing, and four key findings grounded in real population health practice.

---

## Features

- Synthetic patient data with realistic clinical and social health distributions
- K-Means clustering to identify population segments
- Gradient Boosting Classifier for hospitalization prediction
- Feature importance analysis
- Interactive filters via sidebar
- Full analysis writeup connecting the model to real-world application

---

## Tech Stack

- **Python** with Streamlit
- **scikit-learn** for clustering and predictive modeling
- **Plotly** for interactive visualizations
- **pandas / numpy** for data processing
- Deployed on **Streamlit Cloud**

---

## Running Locally

```bash
git clone https://github.com/EminaToric/population-health-risk-stratifier.git
cd population-health-risk-stratifier
pip install -r requirements.txt
streamlit run app.py
```

---

## A Note on the Data

All patient data in this application is synthetically generated using numpy random distributions calibrated to mirror real population health patterns. No real patient data is used. The model logic and risk factor relationships are based on published population health research and my own professional experience in healthcare analytics.

---

## About

Built by **Emina Toric** — data professional with a background in computer science, human development, and a decade of healthcare analytics at UnitedHealth Group and Humana.

[Portfolio](https://eminatoric.github.io) · [LinkedIn](https://linkedin.com/in/emina-toric-msc) · [GitHub](https://github.com/EminaToric)
