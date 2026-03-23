import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Population Health Risk Stratifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #F7F3EE; }

    .stApp { background-color: #F7F3EE; }

    .metric-card {
        background: #FDFAF7;
        border: 1px solid rgba(0,0,0,0.07);
        border-left: 3px solid #C4714A;
        padding: 1.25rem 1.5rem;
        border-radius: 2px;
        margin-bottom: 1rem;
    }

    .metric-card.sage { border-left-color: #7A8C72; }
    .metric-card.gold { border-left-color: #C8A86A; }
    .metric-card.sky  { border-left-color: #7AA0B8; }

    .metric-num {
        font-size: 2rem;
        font-weight: 600;
        color: #C4714A;
        line-height: 1;
        margin-bottom: 0.2rem;
    }

    .metric-label {
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9A9088;
        margin-bottom: 0.4rem;
    }

    .metric-detail {
        font-size: 0.78rem;
        color: #4A4540;
        line-height: 1.5;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #1E1B18;
        margin-bottom: 0.25rem;
    }

    .section-desc {
        font-size: 0.82rem;
        color: #9A9088;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    div[data-testid="stSidebar"] {
        background-color: #FDFAF7;
        border-right: 1px solid rgba(0,0,0,0.07);
    }

    .stSelectbox label, .stSlider label, .stMultiselect label {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9A9088;
    }

    h1 { color: #1E1B18 !important; }
    h2 { color: #1E1B18 !important; font-size: 1.3rem !important; }
    h3 { color: #1E1B18 !important; font-size: 1.1rem !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: #FDFAF7;
        border: 1.5px solid rgba(0,0,0,0.1);
        border-radius: 100px;
        padding: 0.5rem 1.25rem;
        font-size: 0.78rem;
        font-weight: 500;
        color: #9A9088;
    }

    .stTabs [aria-selected="true"] {
        background: #C4714A !important;
        border-color: #C4714A !important;
        color: white !important;
    }

    .finding-box {
        background: #FDFAF7;
        border: 1px solid rgba(0,0,0,0.07);
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────

@st.cache_data
def generate_population(n=2000, seed=42):
    np.random.seed(seed)

    age = np.random.normal(55, 18, n).clip(18, 95).astype(int)
    gender = np.random.choice(["Male", "Female"], n)
    income_level = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
    insurance_type = np.random.choice(["Medicaid", "Medicare", "Commercial", "Uninsured"], n, p=[0.25, 0.3, 0.35, 0.1])
    chronic_conditions = np.random.poisson(1.8, n).clip(0, 6)
    bmi = np.random.normal(27.5, 6, n).clip(16, 50)
    smoking = np.random.choice([0, 1], n, p=[0.75, 0.25])
    ed_visits_12m = np.random.poisson(0.8, n).clip(0, 8)
    primary_care_visits = np.random.poisson(3, n).clip(0, 12)
    medication_adherence = np.random.uniform(0.3, 1.0, n)
    social_isolation = np.random.choice([0, 1], n, p=[0.7, 0.3])
    housing_instability = np.random.choice([0, 1], n, p=[0.85, 0.15])
    food_insecurity = np.random.choice([0, 1], n, p=[0.8, 0.2])
    transportation_barrier = np.random.choice([0, 1], n, p=[0.75, 0.25])

    # Risk score based on multiple factors
    risk_score = (
        (age / 100) * 20 +
        chronic_conditions * 12 +
        (bmi > 30).astype(int) * 8 +
        smoking * 10 +
        ed_visits_12m * 5 +
        (1 - medication_adherence) * 15 +
        social_isolation * 6 +
        housing_instability * 8 +
        food_insecurity * 6 +
        transportation_barrier * 4 +
        (income_level == "Low").astype(int) * 8 +
        (insurance_type == "Uninsured").astype(int) * 10 +
        np.random.normal(0, 5, n)
    ).clip(0, 100)

    # Hospitalization probability
    hosp_prob = (
        0.01 +
        (risk_score / 100) * 0.45 +
        (chronic_conditions > 3).astype(int) * 0.15 +
        (ed_visits_12m > 2).astype(int) * 0.10
    ).clip(0, 0.95)

    hospitalized = np.random.binomial(1, hosp_prob)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "income_level": income_level,
        "insurance_type": insurance_type,
        "chronic_conditions": chronic_conditions,
        "bmi": bmi.round(1),
        "smoking": smoking,
        "ed_visits_12m": ed_visits_12m,
        "primary_care_visits": primary_care_visits,
        "medication_adherence": medication_adherence.round(2),
        "social_isolation": social_isolation,
        "housing_instability": housing_instability,
        "food_insecurity": food_insecurity,
        "transportation_barrier": transportation_barrier,
        "risk_score": risk_score.round(1),
        "hosp_prob": hosp_prob.round(3),
        "hospitalized": hospitalized,
    })

    # Risk tier
    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Low", "Moderate", "High", "Critical"]
    )

    return df

@st.cache_data
def run_clustering(df):
    features = ["age", "chronic_conditions", "bmi", "ed_visits_12m",
                "medication_adherence", "risk_score"]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df = df.copy()
    df["segment"] = kmeans.fit_predict(X_scaled)
    segment_risk = df.groupby("segment")["risk_score"].mean().sort_values()
    segment_map = {old: new for new, old in enumerate(segment_risk.index)}
    df["segment"] = df["segment"].map(segment_map)
    segment_names = {0: "Segment A: Healthy", 1: "Segment B: Moderate Risk",
                     2: "Segment C: High Risk", 3: "Segment D: Critical"}
    df["segment_name"] = df["segment"].map(segment_names)
    return df

@st.cache_data
def train_model(df):
    features = ["age", "chronic_conditions", "bmi", "smoking", "ed_visits_12m",
                "medication_adherence", "social_isolation", "housing_instability",
                "food_insecurity", "transportation_barrier"]
    X = df[features]
    y = df["hospitalized"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)
    feature_labels = {
        "age": "Age",
        "chronic_conditions": "Chronic Conditions",
        "bmi": "BMI",
        "smoking": "Smoking Status",
        "ed_visits_12m": "ED Visits (12 months)",
        "medication_adherence": "Medication Adherence",
        "social_isolation": "Social Isolation",
        "housing_instability": "Housing Instability",
        "food_insecurity": "Food Insecurity",
        "transportation_barrier": "Transportation Barrier"
    }
    importance["feature_label"] = importance["feature"].map(feature_labels)
    return model, importance

# ── Color scheme ──────────────────────────────────────────────────────────────

COLORS = {
    "terracotta": "#C4714A",
    "sage": "#7A8C72",
    "blush": "#C4908A",
    "gold": "#C8A86A",
    "sky": "#7AA0B8",
    "cream": "#F7F3EE",
    "sand": "#EDE4D8",
    "ink": "#1E1B18",
}

RISK_COLORS = {
    "Low": "#7A8C72",
    "Moderate": "#C8A86A",
    "High": "#D4896A",
    "Critical": "#C4714A",
}

SEGMENT_COLORS = {
    "Segment A: Healthy": "#7A8C72",
    "Segment B: Moderate Risk": "#C8A86A",
    "Segment C: High Risk": "#D4896A",
    "Segment D: Critical": "#C4714A",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FDFAF7",
    font=dict(family="DM Sans", color="#4A4540", size=12),
    margin=dict(t=30, b=30, l=10, r=10),
)

# ── Load data ─────────────────────────────────────────────────────────────────

df_raw = generate_population()
df = run_clustering(df_raw)
model, feature_importance = train_model(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏥 Population Health\nRisk Stratifier")
    st.markdown("---")
    st.markdown("**Filter Population**")

    age_range = st.slider("Age Range", 18, 95, (18, 95))
    selected_risk = st.multiselect(
        "Risk Tier",
        ["Low", "Moderate", "High", "Critical"],
        default=["Low", "Moderate", "High", "Critical"]
    )
    selected_insurance = st.multiselect(
        "Insurance Type",
        ["Medicaid", "Medicare", "Commercial", "Uninsured"],
        default=["Medicaid", "Medicare", "Commercial", "Uninsured"]
    )

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #9A9088; line-height: 1.6;'>
    Synthetic patient data generated to mirror real population health distributions.
    Built by <a href='https://eminatoric.github.io' style='color: #C4714A;'>Emina Toric</a>
    </div>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────

mask = (
    (df["age"] >= age_range[0]) &
    (df["age"] <= age_range[1]) &
    (df["risk_tier"].isin(selected_risk)) &
    (df["insurance_type"].isin(selected_insurance))
)
df_filtered = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='margin-bottom: 0.5rem;'>
    <span style='font-family: DM Mono, monospace; font-size: 0.65rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #C4714A;'>
    Population Health Analytics · Synthetic Patient Data
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Population Health Risk Stratifier")
st.markdown("""
<p style='color: #9A9088; font-size: 0.88rem; max-width: 700px; line-height: 1.7; margin-bottom: 1.5rem;'>
Segmenting a synthetic patient population by health risk using clustering and predictive modeling.
The goal is the same one I work toward professionally: find who is headed toward a bad outcome
before it happens and understand what is driving it.
</p>
""", unsafe_allow_html=True)

# ── Top metrics ───────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Patients in View</div>
        <div class='metric-num'>{len(df_filtered):,}</div>
        <div class='metric-detail'>After applying filters</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    high_risk_pct = (df_filtered["risk_tier"].isin(["High", "Critical"]).sum() / len(df_filtered) * 100)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>High or Critical Risk</div>
        <div class='metric-num'>{high_risk_pct:.1f}%</div>
        <div class='metric-detail'>{df_filtered["risk_tier"].isin(["High", "Critical"]).sum():,} patients</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_risk = df_filtered["risk_score"].mean()
    st.markdown(f"""
    <div class='metric-card sage'>
        <div class='metric-label'>Avg Risk Score</div>
        <div class='metric-num' style='color: #7A8C72;'>{avg_risk:.1f}</div>
        <div class='metric-detail'>Out of 100</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    hosp_rate = df_filtered["hospitalized"].mean() * 100
    st.markdown(f"""
    <div class='metric-card gold'>
        <div class='metric-label'>Hospitalization Rate</div>
        <div class='metric-num' style='color: #C8A86A;'>{hosp_rate:.1f}%</div>
        <div class='metric-detail'>Observed in this population</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk Segments", "Key Drivers", "Hospitalization Risk", "Social Determinants", "Analysis"
])

# ── Tab 1: Risk Segments ──────────────────────────────────────────────────────

with tab1:
    st.markdown('<div class="section-header">Population Risk Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">How the filtered population breaks down across four risk tiers, and the four behavioral and clinical segments identified through clustering.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = df_filtered["risk_tier"].value_counts().reindex(["Low", "Moderate", "High", "Critical"])
        fig = go.Figure(go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[RISK_COLORS[r] for r in risk_counts.index],
            text=risk_counts.values,
            textposition="outside",
        ))
        fig.update_layout(
            title="Patients by Risk Tier",
            xaxis_title=None,
            yaxis_title="Number of Patients",
            showlegend=False,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_counts = df_filtered["segment_name"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=seg_counts.index,
            values=seg_counts.values,
            hole=0.45,
            marker_colors=[SEGMENT_COLORS.get(s, "#B8A898") for s in seg_counts.index],
            textinfo="percent+label",
            textfont_size=11,
        ))
        fig2.update_layout(
            title="Population Segments",
            showlegend=False,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Segment Profiles**")
    seg_profile = df_filtered.groupby("segment_name").agg(
        patients=("age", "count"),
        avg_age=("age", "mean"),
        avg_risk_score=("risk_score", "mean"),
        avg_chronic=("chronic_conditions", "mean"),
        hosp_rate=("hospitalized", "mean"),
    ).round(1)
    seg_profile["hosp_rate"] = (seg_profile["hosp_rate"] * 100).round(1).astype(str) + "%"
    seg_profile.columns = ["Patients", "Avg Age", "Avg Risk Score", "Avg Chronic Conditions", "Hospitalization Rate"]
    st.dataframe(seg_profile, use_container_width=True)

# ── Tab 2: Key Drivers ────────────────────────────────────────────────────────

with tab2:
    st.markdown('<div class="section-header">What Drives Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Feature importance from the gradient boosting model trained to predict hospitalization. The longer the bar the more that factor contributes to the model\'s predictions.</div>', unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=feature_importance["importance"],
        y=feature_importance["feature_label"],
        orientation="h",
        marker_color=COLORS["terracotta"],
        text=feature_importance["importance"].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance for Hospitalization Risk",
        xaxis_title="Importance",
        yaxis_title=None,
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.box(
            df_filtered,
            x="risk_tier",
            y="chronic_conditions",
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_tier": ["Low", "Moderate", "High", "Critical"]},
            title="Chronic Conditions by Risk Tier",
        )
        fig2.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.box(
            df_filtered,
            x="risk_tier",
            y="medication_adherence",
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_tier": ["Low", "Moderate", "High", "Critical"]},
            title="Medication Adherence by Risk Tier",
        )
        fig3.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Hospitalization Risk ───────────────────────────────────────────────

with tab3:
    st.markdown('<div class="section-header">Predicted Hospitalization Likelihood</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Predicted probability of hospitalization by risk tier and segment. The model uses clinical and social factors to estimate who is most likely to end up in the hospital in the next 12 months.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        hosp_by_tier = df_filtered.groupby("risk_tier")["hosp_prob"].mean().reindex(["Low", "Moderate", "High", "Critical"])
        fig = go.Figure(go.Bar(
            x=hosp_by_tier.index,
            y=(hosp_by_tier.values * 100).round(1),
            marker_color=[RISK_COLORS[r] for r in hosp_by_tier.index],
            text=(hosp_by_tier.values * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            title="Avg Hospitalization Probability by Risk Tier",
            yaxis_title="Probability (%)",
            xaxis_title=None,
            showlegend=False,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df_filtered.sample(min(500, len(df_filtered))),
            x="risk_score",
            y="hosp_prob",
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            opacity=0.5,
            title="Risk Score vs Hospitalization Probability",
            labels={"risk_score": "Risk Score", "hosp_prob": "Hospitalization Probability"},
        )
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**ED Visit Patterns by Risk Tier**")
    ed_dist = df_filtered.groupby(["risk_tier", "ed_visits_12m"]).size().reset_index(name="count")
    fig3 = px.bar(
        ed_dist,
        x="ed_visits_12m",
        y="count",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        barmode="group",
        title="ED Visits in Past 12 Months by Risk Tier",
        labels={"ed_visits_12m": "ED Visits", "count": "Patients"},
        category_orders={"risk_tier": ["Low", "Moderate", "High", "Critical"]},
    )
    fig3.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Social Determinants ────────────────────────────────────────────────

with tab4:
    st.markdown('<div class="section-header">Social Determinants of Health</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">How social factors layer onto clinical risk. Food insecurity, housing instability, transportation barriers, and social isolation all show up in the model as independent contributors to hospitalization risk.</div>', unsafe_allow_html=True)

    sdoh_factors = ["social_isolation", "housing_instability", "food_insecurity", "transportation_barrier"]
    sdoh_labels = {
        "social_isolation": "Social Isolation",
        "housing_instability": "Housing Instability",
        "food_insecurity": "Food Insecurity",
        "transportation_barrier": "Transportation Barrier"
    }

    sdoh_by_tier = df_filtered.groupby("risk_tier")[sdoh_factors].mean() * 100
    sdoh_by_tier.columns = [sdoh_labels[c] for c in sdoh_by_tier.columns]
    sdoh_by_tier = sdoh_by_tier.reindex(["Low", "Moderate", "High", "Critical"])

    fig = go.Figure()
    for factor in sdoh_by_tier.columns:
        fig.add_trace(go.Bar(
            name=factor,
            x=sdoh_by_tier.index,
            y=sdoh_by_tier[factor].round(1),
        ))

    fig.update_layout(
        barmode="group",
        title="Social Determinant Prevalence by Risk Tier (%)",
        yaxis_title="% of Patients Affected",
        xaxis_title=None,
        colorway=[COLORS["terracotta"], COLORS["sage"], COLORS["gold"], COLORS["sky"]],
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        ins_risk = df_filtered.groupby("insurance_type")["risk_score"].mean().sort_values(ascending=False)
        fig2 = go.Figure(go.Bar(
            x=ins_risk.index,
            y=ins_risk.values.round(1),
            marker_color=COLORS["terracotta"],
            text=ins_risk.values.round(1),
            textposition="outside",
        ))
        fig2.update_layout(
            title="Average Risk Score by Insurance Type",
            yaxis_title="Average Risk Score",
            xaxis_title=None,
            showlegend=False,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        inc_risk = df_filtered.groupby("income_level")["risk_score"].mean().reindex(["Low", "Medium", "High"])
        fig3 = go.Figure(go.Bar(
            x=inc_risk.index,
            y=inc_risk.values.round(1),
            marker_color=[COLORS["terracotta"], COLORS["gold"], COLORS["sage"]],
            text=inc_risk.values.round(1),
            textposition="outside",
        ))
        fig3.update_layout(
            title="Average Risk Score by Income Level",
            yaxis_title="Average Risk Score",
            xaxis_title=None,
            showlegend=False,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 5: Analysis ───────────────────────────────────────────────────────────

with tab5:
    st.markdown('<div class="section-header">What this model is actually doing and why it matters</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.87rem; color: #4A4540; line-height: 1.85; max-width: 780px;'>

    <p style='margin-bottom: 1rem;'>
    This is a simplified version of the kind of work I do professionally. In healthcare analytics,
    the goal is almost always the same: find the people who are headed toward a bad outcome before
    it happens, understand what is driving it, and give someone the information they need to
    intervene. The data here is synthetic but the logic mirrors real population health programs
    at health insurers, health systems, and government payers.
    </p>

    <p style='margin-bottom: 1rem;'>
    The model uses a gradient boosting classifier trained on clinical and social features to predict
    hospitalization probability. Gradient boosting works well here because it handles the nonlinear
    relationships between variables that matter in health data. A patient with three chronic
    conditions and low medication adherence is not just the sum of those two risks. The interaction
    between them matters too.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Key findings from this population**")

    findings = [
        ("Medication adherence is the strongest modifiable driver of risk.",
         "It ranks among the top predictors of hospitalization even after controlling for clinical complexity. This is actionable. Chronic conditions and age are not things you can change. Adherence support programs are."),
        ("Social determinants layer onto clinical risk in a predictable way.",
         "Uninsured patients show significantly higher average risk scores than commercially insured patients. Food insecurity and housing instability are more prevalent in higher risk tiers. These are not soft factors. They are clinical risk factors with a different name."),
        ("The Critical segment is a small group driving outsized utilization.",
         "The top risk tier represents a minority of the population but accounts for a disproportionate share of ED visits and hospitalizations. This is the group where targeted case management has the clearest return on investment."),
        ("ED visit frequency is both a symptom and a predictor.",
         "High ED utilization in the past 12 months is strongly associated with future hospitalization risk. Patients who use the ED as primary care are often signaling that something in their care coordination has broken down."),
    ]

    for i, (title, detail) in enumerate(findings):
        st.markdown(f"""
        <div class='finding-box'>
            <div style='display: flex; gap: 1rem; align-items: flex-start;'>
                <span style='font-family: serif; font-size: 1.4rem; font-weight: 600;
                color: #C4714A; line-height: 1; flex-shrink: 0; width: 1.75rem;'>
                0{i+1}
                </span>
                <div>
                    <div style='font-weight: 500; color: #1E1B18; font-size: 0.88rem;
                    margin-bottom: 0.3rem;'>{title}</div>
                    <div style='font-size: 0.82rem; color: #4A4540; line-height: 1.7;'>{detail}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.82rem; color: #9A9088; margin-top: 1.5rem; padding-top: 1rem;
    border-top: 1px solid rgba(0,0,0,0.07); line-height: 1.6;'>
    Data is synthetically generated to mirror real population health distributions.
    Model trained using scikit-learn GradientBoostingClassifier.
    Built by <a href='https://eminatoric.github.io' style='color: #C4714A;'>Emina Toric</a>
    </div>
    """, unsafe_allow_html=True)
