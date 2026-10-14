"""CreditWise - single-file Streamlit loan approval app."""
import streamlit as st

import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAT_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Education_Level", "Employer_Category"]
RAW_NUM_COLS = ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
                "Collateral_Value", "Loan_Amount", "Loan_Term"]
ENGINEERED = ["Total_Income", "EMI_Approx", "EMI_to_Income", "Loan_to_Income"]
FEATURE_ORDER = [
    "Applicant_Income", "Coapplicant_Income", "Employment_Status", "Age",
    "Marital_Status", "Dependents", "Credit_Score", "Existing_Loans", "DTI_Ratio",
    "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term", "Loan_Purpose",
    "Property_Area", "Education_Level", "Employer_Category",
]


def add_features(X):
    """Row-wise engineered features (no leakage)."""
    X = X.copy()
    total_income = (X["Applicant_Income"] + X["Coapplicant_Income"]).replace(0, np.nan)
    X["Total_Income"] = total_income
    X["EMI_Approx"] = X["Loan_Amount"] / X["Loan_Term"].replace(0, np.nan)
    X["EMI_to_Income"] = X["EMI_Approx"] / total_income
    X["Loan_to_Income"] = X["Loan_Amount"] / total_income
    return X


def find_dataset():
    paths = glob.glob(os.path.join(BASE_DIR, "*.csv")) + \
        glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))
    for p in paths:
        try:
            if "Loan_Approved" in pd.read_csv(p, nrows=1).columns:
                return p
        except Exception:
            pass
    raise FileNotFoundError("loan_approval_data.csv not found next to app.py")


def build_pipeline():
    num_cols = RAW_NUM_COLS + ENGINEERED
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, num_cols),
                              ("cat", categorical_pipe, CAT_COLS)])
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("prep", prep),
                     ("model", RandomForestClassifier(random_state=RANDOM_STATE))])


def train_model(csv_path=None):
    df = pd.read_csv(csv_path or find_dataset())
    df = df.dropna(subset=["Loan_Approved"]).drop(columns=["Applicant_ID"]).reset_index(drop=True)
    y = (df["Loan_Approved"] == "Yes").astype(int)
    X = df[FEATURE_ORDER]          # Gender excluded (fairness)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        build_pipeline(),
        {"model__n_estimators": [200, 400],
         "model__max_depth": [None, 10, 20],
         "model__min_samples_leaf": [1, 3],
         "model__class_weight": [None, "balanced"]},
        cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "cv_f1": grid.best_score_,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    names = best.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
    imp.index = [n.split("__", 1)[1] for n in imp.index]
    importances = imp.sort_values(ascending=False).head(12).to_dict()

    bundle = {"pipeline": best, "metrics": metrics, "importances": importances}
    return bundle



# ============================ STREAMLIT UI ============================
import pandas as pd


st.set_page_config(page_title="CreditWise – Loan Approval", page_icon="🏦", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; color: #fff;
       background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 100%); margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .3rem 0 0; opacity: .9;}
.sec {font-weight: 700; font-size: 1.05rem; margin: .2rem 0 .4rem;}
.result {border-radius: 18px; padding: 1.5rem 1.8rem; text-align: center; border: 2px solid;}
.result.ok  {background: rgba(22,163,74,.12);  border-color: #16a34a;}
.result.bad {background: rgba(220,38,38,.12);  border-color: #dc2626;}
.result .big {font-size: 2.4rem; font-weight: 800; letter-spacing: .5px;}
.result.ok .big {color: #16a34a;} .result.bad .big {color: #dc2626;}
.bar {height: 14px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden; margin-top: .6rem;}
.bar > div {height: 100%; border-radius: 99px;}
.chip {display: inline-block; padding: .15rem .65rem; border-radius: 99px; font-size: .8rem; font-weight: 600;}
.chip.g {background: rgba(22,163,74,.18); color: #16a34a;}
.chip.y {background: rgba(217,119,6,.18); color: #d97706;}
.chip.r {background: rgba(220,38,38,.18); color: #dc2626;}
.ind {border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: .8rem 1rem; height: 100%;}
.ind small {opacity: .7;}
.ind .v {font-size: 1.35rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the model (first run only)…")
def get_bundle():
    return train_model()


bundle = get_bundle()
model = bundle["pipeline"]

st.markdown("""
<div class="hero">
  <h1>🏦 CreditWise</h1>
  <p>Smart loan approval prediction – fill in the applicant details and get an instant decision.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ tab 1
with st.container():
    with st.form("loan_form"):
        st.markdown('<div class="sec">👤 Personal Details</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 18, 70, 35)
        marital = c2.selectbox("Marital Status", ["Single", "Married"])
        dependents = c3.number_input("Dependents", 0, 10, 0)
        education = c4.selectbox("Education Level", ["Graduate", "Not Graduate"])

        st.markdown('<div class="sec">💼 Employment & Income</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        employment = c1.selectbox("Employment Status",
                                  ["Salaried", "Self-employed", "Contract", "Unemployed"])
        employer = c2.selectbox("Employer Category",
                                ["Private", "Government", "MNC", "Business", "Unemployed"])
        app_income = c3.number_input("Applicant Income (monthly)", 0, 1_000_000, 9000, step=500)
        co_income = c4.number_input("Coapplicant Income (monthly)", 0, 1_000_000, 2000, step=500)

        st.markdown('<div class="sec">💳 Financial Profile</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        credit = c1.slider("Credit Score", 300, 900, 720)
        dti = c2.slider("DTI Ratio", 0.00, 1.00, 0.25, 0.01)
        existing = c3.number_input("Existing Loans", 0, 20, 1)
        savings = c4.number_input("Savings", 0, 10_000_000, 12000, step=500)
        c1, c2, c3, c4 = st.columns(4)
        collateral = c1.number_input("Collateral Value", 0, 10_000_000, 30000, step=1000)
        property_area = c2.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        st.markdown('<div class="sec">📄 Loan Request</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        loan_amount = c1.number_input("Loan Amount", 500, 10_000_000, 20000, step=500)
        loan_term = c2.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=4)
        purpose = c3.selectbox("Loan Purpose", ["Personal", "Car", "Home", "Business", "Education"])

        submitted = st.form_submit_button("🔍 Check Loan Eligibility",
                                          use_container_width=True, type="primary")

    if submitted:
        applicant = pd.DataFrame([{
            "Applicant_Income": app_income, "Coapplicant_Income": co_income,
            "Employment_Status": employment, "Age": age, "Marital_Status": marital,
            "Dependents": dependents, "Credit_Score": credit, "Existing_Loans": existing,
            "DTI_Ratio": dti, "Savings": savings, "Collateral_Value": collateral,
            "Loan_Amount": loan_amount, "Loan_Term": loan_term, "Loan_Purpose": purpose,
            "Property_Area": property_area, "Education_Level": education,
            "Employer_Category": employer,
        }])[FEATURE_ORDER]

        prob = float(model.predict_proba(applicant)[0, 1])
        approved = prob >= 0.5

        total_income = app_income + co_income
        emi = loan_amount / loan_term
        emi_ratio = emi / total_income if total_income else 1.0
        loan_ratio = loan_amount / total_income if total_income else 99.0

        st.markdown("---")
        left, right = st.columns([1, 1.2])
        with left:
            cls, label, icon = ("ok", "APPROVED", "✅") if approved else ("bad", "REJECTED", "❌")
            color = "#16a34a" if approved else "#dc2626"
            st.markdown(f"""
            <div class="result {cls}">
              <div style="opacity:.8">Loan Decision</div>
              <div class="big">{icon} {label}</div>
              <div style="margin-top:.4rem">Approval probability: <b>{prob:.1%}</b></div>
              <div class="bar"><div style="width:{prob*100:.0f}%; background:{color}"></div></div>
            </div>""", unsafe_allow_html=True)
            if approved:
                st.success("The profile looks strong for this loan request.")
            else:
                st.error("The profile does not meet the approval pattern for this request.")

        with right:
            def chip(level, text):
                return f'<span class="chip {level}">{text}</span>'

            def ind(title, value, level, text):
                return (f'<div class="ind"><small>{title}</small><div class="v">{value}</div>'
                        f'{chip(level, text)}</div>')

            cs = ("g", "Strong") if credit >= 700 else ("y", "Fair") if credit >= 650 else ("r", "Weak")
            ds = ("g", "Healthy") if dti <= 0.35 else ("y", "Moderate") if dti <= 0.45 else ("r", "High")
            es = ("g", "Comfortable") if emi_ratio <= 0.10 else ("y", "Manageable") if emi_ratio <= 0.20 else ("r", "Heavy")
            ls = ("g", "Low") if loan_ratio <= 2 else ("y", "Medium") if loan_ratio <= 4 else ("r", "High")

            st.markdown("**Key indicators**")
            r1, r2 = st.columns(2)
            r1.markdown(ind("Credit Score", credit, *cs), unsafe_allow_html=True)
            r2.markdown(ind("DTI Ratio", f"{dti:.2f}", *ds), unsafe_allow_html=True)
            st.write("")
            r3, r4 = st.columns(2)
            r3.markdown(ind("EMI / Income", f"{emi_ratio:.1%}", *es), unsafe_allow_html=True)
            r4.markdown(ind("Loan / Income", f"{loan_ratio:.2f}x", *ls), unsafe_allow_html=True)

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Monthly Income", f"{total_income:,.0f}")
        m2.metric("Approx. Monthly EMI", f"{emi:,.0f}")
        m3.metric("Collateral / Loan", f"{collateral / loan_amount:.2f}x")

st.caption("⚠️ This tool is a decision-support demo built on a small dataset. "
           "Final loan decisions should involve a human review.")