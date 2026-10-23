import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE = Path(__file__).parent
BUNDLE_PATH = BASE / "model_bundle.pkl"
DATA_PATH = BASE / "clustered_customers.csv"

st.set_page_config(page_title="SmartCart Customer Segmentation",
                   page_icon="🛒", layout="wide")

SEG_ORDER = ["Premium Spenders", "Loyal Deal Seekers", "Light Spenders", "Budget Browsers"]
COLORS = {"Premium Spenders": "#1D9E75", "Loyal Deal Seekers": "#378ADD",
          "Light Spenders": "#EF9F27", "Budget Browsers": "#D85A30"}
ICONS = {"Premium Spenders": "💎", "Loyal Deal Seekers": "🏷️",
         "Light Spenders": "🛍️", "Budget Browsers": "🔍"}

# ---------------- styling ----------------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {
    background: linear-gradient(135deg, #1D9E75 0%, #378ADD 100%);
    padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;
}
.hero h2 {margin: 0; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92;}
.result-card {
    padding: 1.6rem 2rem; border-radius: 16px; color: white; margin: 1rem 0;
    box-shadow: 0 4px 14px rgba(0,0,0,.15);
}
.result-card h1 {margin: 0; color: white; font-size: 2.1rem;}
.result-card p {margin: .5rem 0 0 0; font-size: 1.02rem; opacity: .95;}
.section-title {font-weight: 600; font-size: 1.05rem; margin-bottom: .4rem;}
</style>
""", unsafe_allow_html=True)


# ---------------- training (runs automatically if model files are missing) ----------------
def train_and_save():
    src = None
    for name in ["smartcart_customers.csv", "smartcart_customers.xls"]:
        if (BASE / name).exists():
            src = BASE / name
            break
    if src is None:
        st.error("Training data not found. Place the dataset next to app.py and reload.")
        st.stop()

    df = pd.read_csv(src)  # the .xls is actually a CSV file

    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days

    spend = ["MntWines", "MntFruits", "MntMeatProducts",
             "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spending"] = df[spend].sum(axis=1)
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "PostGraduate", "PhD": "PostGraduate"})
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner", "Single": "Alone",
        "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"})

    df = df.drop(columns=["ID", "Year_Birth", "Marital_Status",
                          "Kidhome", "Teenhome", "Dt_Customer"] + spend)
    df = df[(df["Age"] < 90) & (df["Income"] < 600_000)].reset_index(drop=True)

    df["Education_Graduate"] = (df["Education"] == "Graduate").astype(int)
    df["Education_PostGraduate"] = (df["Education"] == "PostGraduate").astype(int)
    df["Education_Undergraduate"] = (df["Education"] == "Undergraduate").astype(int)
    df["Living_With_Partner"] = (df["Living_With"] == "Partner").astype(int)

    raw = df.drop(columns=["Education", "Living_With"])
    features = list(raw.columns)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(raw)
    pca = PCA(n_components=3, random_state=42)
    Xp = pca.fit_transform(Xs)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xp)

    raw["Cluster"] = km.labels_
    raw["PC1"], raw["PC2"], raw["PC3"] = Xp[:, 0], Xp[:, 1], Xp[:, 2]

    order = raw.groupby("Cluster")["Total_Spending"].mean().sort_values().index.tolist()
    names = ["Budget Browsers", "Light Spenders", "Loyal Deal Seekers", "Premium Spenders"]
    tips = {
        "Budget Browsers": "Low income, low spend, many web visits. Use entry-level bundles, coupons and free-shipping thresholds.",
        "Light Spenders": "Small baskets, highly educated. Nudge with personalised recommendations and loyalty points.",
        "Loyal Deal Seekers": "Longest tenure, heavy web and deal purchases. Reward with flash sales, early access and subscriptions.",
        "Premium Spenders": "High income, few kids, big catalog and store buyers. Offer premium ranges and VIP programmes.",
    }
    name_map = {int(c): names[i] for i, c in enumerate(order)}
    raw["Segment"] = raw["Cluster"].map(name_map)

    joblib.dump({"kmeans": km, "pca": pca, "scaler": scaler, "features": features,
                 "name_map": name_map, "tips": tips}, BUNDLE_PATH)
    raw.to_csv(DATA_PATH, index=False)


if not (BUNDLE_PATH.exists() and DATA_PATH.exists()):
    with st.spinner("Setting up the model for the first time..."):
        train_and_save()


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


b, df = load_bundle(), load_data()

st.title("🛒 SmartCart Customer Segmentation")
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧩 Segment Profiles", "🔮 Predict a Customer"])

# =====================================================================
# Overview (aggregated charts only, no customer-level data)
# =====================================================================
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Segments", df.Segment.nunique())
    c2.metric("Avg customer spend", f"{df.Total_Spending.mean():,.0f}")
    c3.metric("Avg customer income", f"{df.Income.mean():,.0f}")

    left, right = st.columns(2)
    size = df.Segment.value_counts().reindex(SEG_ORDER).reset_index()
    size.columns = ["Segment", "Customers"]
    left.plotly_chart(
        px.pie(size, names="Segment", values="Customers", hole=0.5,
               color="Segment", color_discrete_map=COLORS, title="Segment share"),
        use_container_width=True)

    rev = df.groupby("Segment").Total_Spending.sum().reindex(SEG_ORDER).reset_index()
    right.plotly_chart(
        px.bar(rev, x="Segment", y="Total_Spending", color="Segment",
               color_discrete_map=COLORS, title="Revenue by segment")
        .update_layout(showlegend=False),
        use_container_width=True)

    st.plotly_chart(
        px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="Segment",
                      color_discrete_map=COLORS, opacity=0.7, title="Customer clusters"),
        use_container_width=True)

# =====================================================================
# Segment Profiles (aggregated only)
# =====================================================================
with tab2:
    seg = st.selectbox("Choose a segment", SEG_ORDER,
                       format_func=lambda s: f"{ICONS[s]}  {s}")
    sub = df[df.Segment == seg]
    st.subheader(f"{ICONS[seg]} {seg}")
    st.info(b["tips"][seg])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Share of customers", f"{len(sub) / len(df):.0%}")
    m2.metric("Avg income", f"{sub.Income.mean():,.0f}")
    m3.metric("Avg spend", f"{sub.Total_Spending.mean():,.0f}")
    m4.metric("Campaign response", f"{sub.Response.mean():.0%}")

    channels = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
                "NumDealsPurchases", "NumWebVisitsMonth"]
    ch = (df.groupby("Segment")[channels].mean()
          .reindex(SEG_ORDER).reset_index().melt("Segment"))
    st.plotly_chart(
        px.bar(ch, x="variable", y="value", color="Segment", barmode="group",
               color_discrete_map=COLORS, title="Channel behaviour by segment",
               labels={"variable": "", "value": "Average per customer"}),
        use_container_width=True)

    cols = ["Income", "Age", "Total_Spending", "Total_Children", "Recency",
            "Customer_Tenure_Days", "Response"] + channels
    st.markdown("**Segment averages**")
    st.dataframe(df.groupby("Segment")[cols].mean().reindex(SEG_ORDER).round(1).T,
                 use_container_width=True)

# =====================================================================
# Predict a Customer (enhanced)
# =====================================================================
with tab3:
    st.markdown("""
    <div class="hero">
        <h2>🔮 Customer Segment Predictor</h2>
        <p>Fill in the customer's details to find the segment they belong to and the best way to market to them.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            with st.container(border=True):
                st.markdown('<div class="section-title">👤 Customer profile</div>', unsafe_allow_html=True)
                income = st.number_input("Annual income", 0, 600000, 50000, step=1000)
                age = st.slider("Age", 18, 90, 40)
                children = st.slider("Children at home (kids + teens)", 0, 5, 1)
                education = st.selectbox("Education", ["Undergraduate", "Graduate", "PostGraduate"], index=1)
                living = st.radio("Living situation", ["Partner", "Alone"], horizontal=True)

            with st.container(border=True):
                st.markdown('<div class="section-title">🕒 Relationship with store</div>', unsafe_allow_html=True)
                total_spend = st.number_input("Total spending (all categories)", 0, 3000, 500, step=10)
                recency = st.slider("Days since last purchase", 0, 100, 50)
                tenure = st.slider("Customer tenure (days)", 0, 700, 350)
                response = st.radio("Responded to last campaign?", ["No", "Yes"], horizontal=True)

        with col_b:
            with st.container(border=True):
                st.markdown('<div class="section-title">🛒 Purchase channels</div>', unsafe_allow_html=True)
                web = st.slider("Web purchases", 0, 30, 4)
                cat = st.slider("Catalog purchases", 0, 30, 2)
                store = st.slider("Store purchases", 0, 30, 5)
                deals = st.slider("Purchases made with a deal", 0, 20, 2)
                visits = st.slider("Website visits per month", 0, 20, 5)

        go_btn = st.form_submit_button("🔍  Predict segment", type="primary",
                                       use_container_width=True)

    if go_btn:
        row = {
            "Income": income, "Recency": recency, "NumDealsPurchases": deals,
            "NumWebPurchases": web, "NumCatalogPurchases": cat,
            "NumStorePurchases": store, "NumWebVisitsMonth": visits,
            "Complain": 0, "Response": int(response == "Yes"),
            "Age": age, "Customer_Tenure_Days": tenure,
            "Total_Spending": total_spend, "Total_Children": children,
            "Education_Graduate": int(education == "Graduate"),
            "Education_PostGraduate": int(education == "PostGraduate"),
            "Education_Undergraduate": int(education == "Undergraduate"),
            "Living_With_Partner": int(living == "Partner"),
        }
        X = pd.DataFrame([row])[b["features"]]
        Xp = b["pca"].transform(b["scaler"].transform(X))
        label = int(b["kmeans"].predict(Xp)[0])
        name = b["name_map"][label]

        # similarity to each segment (closer centroid = higher share)
        dist = b["kmeans"].transform(Xp)[0]
        weights = 1.0 / (dist + 1e-6)
        shares = weights / weights.sum()
        sim = (pd.DataFrame({"Segment": [b["name_map"][i] for i in range(len(dist))],
                             "Match": shares})
               .set_index("Segment").reindex(SEG_ORDER).reset_index())

        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(135deg, {COLORS[name]}, {COLORS[name]}cc);">
                <div style="opacity:.85;">Predicted segment</div>
                <h1>{ICONS[name]} {name}</h1>
                <p>{b["tips"][name]}</p>
            </div>
            """, unsafe_allow_html=True)

        r1, r2 = st.columns(2, gap="large")

        with r1:
            fig = px.bar(sim, x="Match", y="Segment", orientation="h", color="Segment",
                         color_discrete_map=COLORS, title="How close is this customer to each segment?",
                         text=sim["Match"].map(lambda v: f"{v:.0%}"))
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%",
                              xaxis_title="", yaxis_title="", height=340)
            fig.update_yaxes(categoryorder="array", categoryarray=SEG_ORDER[::-1])
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            seg_avg = df[df.Segment == name]
            labels = ["Web", "Catalog", "Store", "Deals", "Web visits"]
            cust_vals = [web, cat, store, deals, visits]
            seg_vals = [seg_avg.NumWebPurchases.mean(), seg_avg.NumCatalogPurchases.mean(),
                        seg_avg.NumStorePurchases.mean(), seg_avg.NumDealsPurchases.mean(),
                        seg_avg.NumWebVisitsMonth.mean()]
            fig2 = go.Figure()
            fig2.add_bar(name="This customer", x=labels, y=cust_vals, marker_color=COLORS[name])
            fig2.add_bar(name=f"{name} average", x=labels, y=[round(v, 1) for v in seg_vals],
                         marker_color="#B4B2A9")
            fig2.update_layout(barmode="group", title="Customer vs segment average",
                               height=340, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig2, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Income vs segment avg", f"{income:,.0f}",
                  f"{income - seg_avg.Income.mean():+,.0f}")
        k2.metric("Spending vs segment avg", f"{total_spend:,.0f}",
                  f"{total_spend - seg_avg.Total_Spending.mean():+,.0f}")
        k3.metric("Age vs segment avg", f"{age}",
                  f"{age - seg_avg.Age.mean():+.0f}")

        st.caption("This is a model-based estimate. Treat it as a guide, not a guarantee.")