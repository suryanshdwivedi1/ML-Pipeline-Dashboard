import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error,
    r2_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb ML Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main { background: #0d0f14; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1d27 0%, #13151f 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #7c85a2 !important;
    font-size: 0.75rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8eaf2 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #5a5f7a !important;
    letter-spacing: 0.03em;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #7c6ef5 !important;
    border-bottom-color: #7c6ef5 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c6ef5 0%, #5b4ed4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124, 110, 245, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 110, 245, 0.45) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #1e2132;
}
section[data-testid="stSidebar"] * {
    color: #c8ccde !important;
}

/* Select/Slider labels */
label[data-testid="stWidgetLabel"] p {
    font-family: 'DM Sans', sans-serif !important;
    color: #8a8fa8 !important;
    font-size: 0.82rem !important;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Info/warning/success boxes */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #c0c5de;
    letter-spacing: 0.02em;
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2132;
}

.badge {
    display: inline-block;
    background: rgba(124,110,245,0.15);
    color: #7c6ef5;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(124,110,245,0.3);
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 1rem;">
    <h1 style="font-size:2.4rem; font-weight:800; color:#e8eaf2; margin:0; letter-spacing:-0.02em;">
        🏙️ NYC Airbnb <span style="color:#7c6ef5;">ML Dashboard</span>
    </h1>
    <p style="color:#6b7194; font-size:0.95rem; margin-top:0.4rem; font-weight:300;">
        End-to-end machine learning pipeline · Explore · Preprocess · Train · Evaluate
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in [("results", None), ("cleaned_df", None), ("model_obj", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#4a4f6a; line-height:1.6;'>"
        "Upload the AB_NYC_2019 dataset or any compatible CSV to begin.<br><br>"
        "Navigate through the tabs in order: Ingest → EDA → Preprocess → Train → Evaluate."
        "</p>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,15,20,0.6)",
    font_family="DM Sans",
    font_color="#c8ccde",
    colorway=["#7c6ef5", "#f5836e", "#5cc8ff", "#60e8a0", "#f5c46e", "#e87cd4"],
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe)
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def remove_outliers(df_in, columns, method, threshold):
    df_out = df_in.copy()
    for col in columns:
        if col not in df_out.columns or not pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        if method == "Z-Score":
            std = df_out[col].std()
            if pd.isna(std) or std == 0:
                continue
            mean = df_out[col].mean()
            df_out = df_out[(df_out[col] >= mean - threshold * std) & (df_out[col] <= mean + threshold * std)]
        elif method == "IQR":
            q1, q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            df_out = df_out[(df_out[col] >= q1 - threshold * iqr) & (df_out[col] <= q3 + threshold * iqr)]
    return df_out

def styled_plotly(fig):
    fig.update_layout(**PLOTLY_THEME)
    fig.update_xaxes(gridcolor="#1e2132", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2132", zeroline=False)
    return fig

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
if uploaded_file is not None:
    df = load_data(uploaded_file)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ Data Ingestion",
        "📊 EDA",
        "🛠️ Preprocessing",
        "🧠 Model Training",
        "📈 Evaluation"
    ])

    # ──────────────────────────────
    # TAB 1 — DATA INGESTION
    # ──────────────────────────────
    with tab1:
        st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

        total_missing = int(df.isna().sum().sum())
        dup_rows = int(df.duplicated().sum())
        num_cols_count = len(df.select_dtypes(include=np.number).columns)
        cat_cols_count = len(df.select_dtypes(exclude=np.number).columns)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", f"{total_missing:,}")
        c4.metric("Duplicate Rows", dup_rows)
        c5.metric("Numeric / Categorical", f"{num_cols_count} / {cat_cols_count}")

        st.markdown('<p class="section-header">Raw Data Preview</p>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown('<p class="section-header">Column Summary</p>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non_null": df.notnull().sum(),
            "null_%": (df.isnull().mean() * 100).round(2),
            "unique": df.nunique(),
            "sample": df.iloc[0]
        })
        st.dataframe(summary, use_container_width=True)

    # ──────────────────────────────
    # TAB 2 — EDA
    # ──────────────────────────────
    with tab2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_list = df.select_dtypes(exclude=np.number).columns.tolist()

        # — Price Distribution
        if "price" in df.columns:
            st.markdown('<p class="section-header">Price Distribution</p>', unsafe_allow_html=True)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                df_plot = df[df["price"] < df["price"].quantile(0.99)]
                fig = px.histogram(df_plot, x="price", nbins=60,
                                   title="Price Distribution (clipped 99th pct)",
                                   color_discrete_sequence=["#7c6ef5"])
                st.plotly_chart(styled_plotly(fig), use_container_width=True)
            with col_p2:
                fig2 = px.box(df[df["price"] < df["price"].quantile(0.99)],
                              x="neighbourhood_group", y="price",
                              title="Price by Borough",
                              color="neighbourhood_group",
                              color_discrete_sequence=["#7c6ef5","#f5836e","#5cc8ff","#60e8a0","#f5c46e"])
                st.plotly_chart(styled_plotly(fig2), use_container_width=True)

        # — Room Type & Borough
        if "room_type" in df.columns and "neighbourhood_group" in df.columns:
            st.markdown('<p class="section-header">Categorical Breakdown</p>', unsafe_allow_html=True)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                rt_counts = df["room_type"].value_counts().reset_index()
                rt_counts.columns = ["room_type", "count"]
                fig3 = px.pie(rt_counts, values="count", names="room_type",
                              title="Room Type Distribution", hole=0.45,
                              color_discrete_sequence=["#7c6ef5","#f5836e","#5cc8ff"])
                st.plotly_chart(styled_plotly(fig3), use_container_width=True)
            with col_c2:
                ng_counts = df["neighbourhood_group"].value_counts().reset_index()
                ng_counts.columns = ["borough", "count"]
                fig4 = px.bar(ng_counts, x="borough", y="count",
                              title="Listings per Borough",
                              color="borough",
                              color_discrete_sequence=["#7c6ef5","#f5836e","#5cc8ff","#60e8a0","#f5c46e"])
                st.plotly_chart(styled_plotly(fig4), use_container_width=True)

        # — Geo Map
        if "latitude" in df.columns and "longitude" in df.columns:
            st.markdown('<p class="section-header">Geographic Spread</p>', unsafe_allow_html=True)
            sample_map = df[df["price"] < df["price"].quantile(0.99)].sample(min(5000, len(df)), random_state=42)
            fig_map = px.scatter_mapbox(
                sample_map, lat="latitude", lon="longitude",
                color="neighbourhood_group" if "neighbourhood_group" in df.columns else None,
                size="price", size_max=10, opacity=0.6,
                zoom=10, height=420,
                mapbox_style="carto-darkmatter",
                title="Listing Locations (sample of 5,000)",
                color_discrete_sequence=["#7c6ef5","#f5836e","#5cc8ff","#60e8a0","#f5c46e"]
            )
            fig_map.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_map, use_container_width=True)

        # — Correlation Heatmap
        if len(numeric_cols) > 1:
            st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                 color_continuous_scale="RdBu_r",
                                 title="Pearson Correlation Matrix")
            st.plotly_chart(styled_plotly(fig_corr), use_container_width=True)

        # — Custom Distribution Explorer
        st.markdown('<p class="section-header">Column Explorer</p>', unsafe_allow_html=True)
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            dist_col = st.selectbox("Distribution of:", numeric_cols, key="eda_dist")
            fig_d = px.histogram(df, x=dist_col, marginal="violin",
                                 color_discrete_sequence=["#5cc8ff"])
            st.plotly_chart(styled_plotly(fig_d), use_container_width=True)
        with col_e2:
            if len(numeric_cols) >= 2:
                scatter_x = st.selectbox("Scatter X:", numeric_cols, key="sc_x")
                scatter_y = st.selectbox("Scatter Y:", numeric_cols,
                                         index=min(1, len(numeric_cols)-1), key="sc_y")
                sample_s = df.sample(min(3000, len(df)), random_state=1)
                fig_s = px.scatter(sample_s, x=scatter_x, y=scatter_y,
                                   color="room_type" if "room_type" in df.columns else None,
                                   opacity=0.5,
                                   color_discrete_sequence=["#7c6ef5","#f5836e","#5cc8ff"])
                st.plotly_chart(styled_plotly(fig_s), use_container_width=True)

    # ──────────────────────────────
    # TAB 3 — PREPROCESSING
    # ──────────────────────────────
    with tab3:
        st.markdown('<p class="section-header">Feature Engineering & Cleaning</p>', unsafe_allow_html=True)

        default_drop = [c for c in ["id", "host_id", "name", "host_name", "last_review"] if c in df.columns]
        cols_to_drop = st.multiselect("Drop columns:", df.columns.tolist(), default=default_drop)
        df_clean = df.drop(columns=cols_to_drop, errors="ignore").copy()

        if "price" in df_clean.columns:
            df_clean = df_clean[df_clean["price"] > 0]
            if st.checkbox("Clip price at 99th percentile", value=True):
                upper_cap = df_clean["price"].quantile(0.99)
                df_clean["price"] = df_clean["price"].clip(upper=upper_cap)
            if st.checkbox("Add log_price feature (helps regression)", value=True):
                df_clean["log_price"] = np.log1p(df_clean["price"])

        if st.checkbox("Drop rows with any missing values", value=False):
            df_clean = df_clean.dropna()

        st.markdown('<p class="section-header">Outlier Removal</p>', unsafe_allow_html=True)
        numeric_clean = df_clean.select_dtypes(include=np.number).columns.tolist()
        default_outlier_cols = [c for c in ["minimum_nights","number_of_reviews",
                                             "reviews_per_month","calculated_host_listings_count"]
                                if c in numeric_clean]
        outlier_cols = st.multiselect("Columns for outlier removal:", numeric_clean,
                                      default=default_outlier_cols)
        
        # PRE-SELECTED IQR FOR SKEWED AIRBNB DATA
        outlier_method = st.radio("Method:", ["IQR", "Z-Score", "None"], index=0, horizontal=True)
        threshold = 1.5
        if outlier_method == "Z-Score":
            threshold = st.slider("Z-Score threshold", 2.0, 4.0, 3.0, 0.1)
        elif outlier_method == "IQR":
            threshold = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)

        if st.button("✅ Apply Outlier Removal"):
            if outlier_method != "None" and outlier_cols:
                before = len(df_clean)
                df_clean = remove_outliers(df_clean, outlier_cols, outlier_method, threshold)
                removed = before - len(df_clean)
                st.success(f"Removed {removed:,} outlier rows → New shape: {df_clean.shape}")
            else:
                st.info("No outlier removal applied.")

        st.session_state.cleaned_df = df_clean.copy()

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Remaining Rows", f"{len(df_clean):,}")
        col_s2.metric("Features", df_clean.shape[1])
        col_s3.metric("Rows Removed", f"{len(df) - len(df_clean):,}")
        st.dataframe(df_clean.head(10), use_container_width=True)

    # ──────────────────────────────
    # TAB 4 — MODEL TRAINING
    # ──────────────────────────────
    with tab4:
        st.markdown('<p class="section-header">Configure & Train</p>', unsafe_allow_html=True)

        df_train = (st.session_state.cleaned_df.copy()
                    if st.session_state.cleaned_df is not None else df.copy())

        if df_train.empty:
            st.warning("Cleaned dataset is empty — adjust preprocessing settings.")
        else:
            # Target selection
            default_target = next((c for c in ["price", "log_price", "room_type"]
                                    if c in df_train.columns), df_train.columns[0])
            target_col = st.selectbox("Target Variable (Y):", df_train.columns.tolist(),
                                       index=df_train.columns.tolist().index(default_target))

            model_df = df_train.dropna(subset=[target_col]).copy()
            is_classification = (
                target_col in ["room_type"]
                or model_df[target_col].dtype == "object"
                or str(model_df[target_col].dtype).startswith("category")
                or (model_df[target_col].nunique() <= 10 and model_df[target_col].dtype != "float64")
            )

            task_badge = "CLASSIFICATION" if is_classification else "REGRESSION"
            st.markdown(f"**Detected Task:** <span class='badge'>{task_badge}</span>", unsafe_allow_html=True)

            # Features
            drop_always = ["id","name","host_id","host_name","last_review"]
            features = model_df.drop(columns=[target_col] + drop_always, errors="ignore")
            if not is_classification and "log_price" in features.columns and target_col == "price":
                features = features.drop(columns=["log_price"], errors="ignore")
            if not is_classification and "price" in features.columns and target_col == "log_price":
                features = features.drop(columns=["price"], errors="ignore")

            X = features.copy()

            st.markdown('<p class="section-header">Algorithm & Hyperparameters</p>', unsafe_allow_html=True)

            if is_classification:
                y = model_df[target_col].astype(str)
                pre = build_preprocessor(X)

                algo = st.selectbox("Algorithm:", [
                    "Random Forest Classifier", "KNN Classifier", "Logistic Regression"
                ], index=2) # Defaulting to Logistic for classification baseline
                c1, c2 = st.columns(2)

                if algo == "Random Forest Classifier":
                    n_est = c1.slider("Trees", 50, 500, 200, 50)
                    max_d = c2.slider("Max Depth", 2, 30, 15)
                    min_samp = c1.slider("Min Samples Leaf", 1, 20, 2)
                    base_model = RandomForestClassifier(
                        n_estimators=n_est, max_depth=max_d,
                        min_samples_leaf=min_samp,
                        random_state=42, n_jobs=-1, class_weight="balanced"
                    )
                elif algo == "KNN Classifier":
                    n_nb = c1.slider("Neighbours (k)", 1, 30, 7)
                    weights = c2.selectbox("Weights", ["distance", "uniform"])
                    metric = c1.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"])
                    base_model = KNeighborsClassifier(n_neighbors=n_nb, weights=weights, metric=metric)
                else:
                    c_val = c1.select_slider("C (regularisation)", [0.001, 0.01, 0.1, 1, 10, 100], value=1)
                    solver = c2.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
                    base_model = LogisticRegression(C=c_val, solver=solver,
                                                     max_iter=3000, class_weight="balanced")

                model = Pipeline([("preprocess", pre), ("model", base_model)])

            else:
                y_raw = pd.to_numeric(model_df[target_col], errors="coerce")
                valid = y_raw.notna()
                X, y_raw = X.loc[valid].copy(), y_raw.loc[valid].copy()

                use_log = False
                if target_col == "price":
                    use_log = st.checkbox("Log-transform target (log1p)", value=True)

                pre = build_preprocessor(X)
                
                # PRE-SELECTED LINEAR REGRESSION 
                algo = st.selectbox("Algorithm:", [
                    "Linear Regression", "Random Forest Regressor", "KNN Regressor"
                ], index=0) 
                
                c1, c2 = st.columns(2)

                if algo == "Random Forest Regressor":
                    n_est = c1.slider("Trees", 50, 500, 200, 50)
                    max_d = c2.slider("Max Depth", 2, 30, 15)
                    min_samp = c1.slider("Min Samples Leaf", 1, 20, 2)
                    base_reg = RandomForestRegressor(
                        n_estimators=n_est, max_depth=max_d,
                        min_samples_leaf=min_samp,
                        random_state=42, n_jobs=-1
                    )
                elif algo == "KNN Regressor":
                    n_nb = c1.slider("Neighbours (k)", 1, 30, 7)
                    weights = c2.selectbox("Weights", ["distance", "uniform"])
                    base_reg = KNeighborsRegressor(n_neighbors=n_nb, weights=weights)
                else:
                    base_reg = LinearRegression()

                reg_pipe = Pipeline([("preprocess", pre), ("model", base_reg)])
                if use_log:
                    model = TransformedTargetRegressor(
                        regressor=reg_pipe, func=np.log1p, inverse_func=np.expm1
                    )
                else:
                    model = reg_pipe
                y = y_raw

            # Train/test split
            st.markdown('<p class="section-header">Training Settings</p>', unsafe_allow_html=True)
            col_ts1, col_ts2 = st.columns(2)
            test_size = col_ts1.slider("Test Set Size (%)", 10, 40, 20) / 100
            cv_folds = col_ts2.slider("Cross-Validation Folds", 2, 10, 5)

            if st.button("🚀 Train Model"):
                with st.spinner("Training... hang tight"):
                    stratify_arg = y if is_classification and y.nunique() > 1 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=stratify_arg
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Cross-val
                    scoring = "accuracy" if is_classification else "r2"
                    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)

                    # Feature importances (RF only)
                    feat_importances = None
                    feat_names = None
                    if "Random Forest" in algo:
                        try:
                            inner = model.named_steps["model"] if hasattr(model, "named_steps") else model.regressor_.named_steps["model"]
                            pre_step = model.named_steps["preprocess"] if hasattr(model, "named_steps") else model.regressor_.named_steps["preprocess"]
                            num_c = X.select_dtypes(include=np.number).columns.tolist()
                            cat_c = X.select_dtypes(exclude=np.number).columns.tolist()
                            cat_names = []
                            if cat_c:
                                ohe_step = pre_step.named_transformers_["cat"].named_steps["ohe"]
                                cat_names = list(ohe_step.get_feature_names_out(cat_c))
                            feat_names = num_c + cat_names
                            feat_importances = inner.feature_importances_
                        except Exception:
                            pass

                    st.session_state.results = {
                        "task": "classification" if is_classification else "regression",
                        "y_test": np.array(y_test),
                        "y_pred": np.array(y_pred),
                        "model_name": algo,
                        "target_col": target_col,
                        "cv_scores": cv_scores,
                        "cv_metric": scoring,
                        "feat_importances": feat_importances,
                        "feat_names": feat_names,
                        "X_test": X_test,
                        "is_log_transform": use_log if not is_classification else False,
                    }
                    st.session_state.model_obj = model
                    st.success("✅ Model trained! Head to the Evaluation tab.")

    # ──────────────────────────────
    # TAB 5 — EVALUATION
    # ──────────────────────────────
    with tab5:
        if not st.session_state.results:
            st.info("Train a model first in the 'Model Training' tab.")
        else:
            res = st.session_state.results
            y_test = np.array(res["y_test"])
            y_pred = np.array(res["y_pred"])
            cv_scores = res["cv_scores"]

            st.markdown(f"### {res['model_name']} <span class='badge'>{res['task'].upper()}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='color:#6b7194;font-size:0.85rem;'>Target: <b style='color:#c8ccde'>{res['target_col']}</b></p>",
                        unsafe_allow_html=True)

            # ── Metrics Row
            st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)

            if res["task"] == "regression":
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("R² Score", f"{r2:.4f}")
                c2.metric("RMSE", f"{rmse:,.2f}")
                c3.metric("MAE", f"{mae:,.2f}")
                c4.metric("MAPE", f"{mape:.1f}%")
                c5.metric(f"CV R² ({len(cv_scores)}-fold)", f"{cv_mean:.4f} ± {cv_std:.4f}")

            else:
                acc = accuracy_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                macro = report.get("macro avg", {})

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{acc:.2%}")
                c2.metric("Macro Precision", f"{macro.get('precision',0):.2%}")
                c3.metric("Macro Recall", f"{macro.get('recall',0):.2%}")
                c4.metric("Macro F1", f"{macro.get('f1-score',0):.2%}")
                c5.metric(f"CV Acc ({len(cv_scores)}-fold)", f"{cv_mean:.2%} ± {cv_std:.3f}")

            # ── CV Scores chart
            st.markdown('<p class="section-header">Cross-Validation Scores</p>', unsafe_allow_html=True)
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                y=cv_scores,
                marker_color=["#7c6ef5" if s >= cv_scores.mean() else "#f5836e" for s in cv_scores],
                text=[f"{s:.4f}" for s in cv_scores],
                textposition="outside"
            ))
            fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#60e8a0",
                             annotation_text=f"Mean: {cv_scores.mean():.4f}")
            fig_cv.update_layout(title="CV Scores per Fold", **PLOTLY_THEME,
                                 yaxis_title=res["cv_metric"].upper())
            fig_cv.update_xaxes(gridcolor="#1e2132"); fig_cv.update_yaxes(gridcolor="#1e2132")
            st.plotly_chart(fig_cv, use_container_width=True)

            if res["task"] == "regression":
                col_v1, col_v2 = st.columns(2)

                # Actual vs Predicted scatter
                with col_v1:
                    st.markdown('<p class="section-header">Actual vs Predicted</p>', unsafe_allow_html=True)
                    n_show = min(800, len(y_test))
                    idx = np.random.choice(len(y_test), n_show, replace=False)
                    fig_avp = px.scatter(
                        x=y_test[idx], y=y_pred[idx],
                        labels={"x": "Actual", "y": "Predicted"},
                        opacity=0.5, color_discrete_sequence=["#7c6ef5"],
                        title=f"Actual vs Predicted (n={n_show})"
                    )
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig_avp.add_shape(type="line", x0=min_val, x1=max_val, y0=min_val, y1=max_val,
                                      line=dict(color="#60e8a0", dash="dash", width=2))
                    st.plotly_chart(styled_plotly(fig_avp), use_container_width=True)

                # Residuals
                with col_v2:
                    st.markdown('<p class="section-header">Residuals Distribution</p>', unsafe_allow_html=True)
                    residuals = y_test - y_pred
                    fig_res = px.histogram(residuals, nbins=50,
                                           color_discrete_sequence=["#f5836e"],
                                           title="Residuals (Actual − Predicted)")
                    fig_res.add_vline(x=0, line_dash="dash", line_color="#60e8a0")
                    st.plotly_chart(styled_plotly(fig_res), use_container_width=True)

                # Line chart: first 150
                st.markdown('<p class="section-header">Sample Comparison (first 150)</p>', unsafe_allow_html=True)
                plot_df = pd.DataFrame({"Actual": y_test[:150], "Predicted": y_pred[:150]})
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(y=plot_df["Actual"], mode="lines", name="Actual",
                                              line=dict(color="#7c6ef5", width=2)))
                fig_line.add_trace(go.Scatter(y=plot_df["Predicted"], mode="lines", name="Predicted",
                                              line=dict(color="#f5836e", width=2, dash="dot")))
                fig_line.update_layout(xaxis_title="Sample", yaxis_title="Value",
                                       hovermode="x unified", **PLOTLY_THEME)
                fig_line.update_xaxes(gridcolor="#1e2132"); fig_line.update_yaxes(gridcolor="#1e2132")
                st.plotly_chart(fig_line, use_container_width=True)

            else:
                col_v1, col_v2 = st.columns(2)

                # Confusion matrix
                with col_v1:
                    st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
                    labels_unique = sorted(list(set(y_test) | set(y_pred)))
                    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
                    fig_cm = px.imshow(
                        cm, x=labels_unique, y=labels_unique,
                        color_continuous_scale="Purples",
                        labels={"x": "Predicted", "y": "Actual"},
                        text_auto=True, title="Confusion Matrix"
                    )
                    fig_cm.update_layout(**PLOTLY_THEME)
                    st.plotly_chart(fig_cm, use_container_width=True)

                # Class report
                with col_v2:
                    st.markdown('<p class="section-header">Classification Report</p>', unsafe_allow_html=True)
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose().round(3)
                    st.dataframe(report_df, use_container_width=True)

                # Pred distribution
                st.markdown('<p class="section-header">Prediction Distribution</p>', unsafe_allow_html=True)
                pred_compare = pd.DataFrame({
                    "Type": ["Actual"] * len(y_test) + ["Predicted"] * len(y_pred),
                    "Class": list(y_test) + list(y_pred)
                })
                fig_dist = px.histogram(pred_compare, x="Class", color="Type", barmode="group",
                                        color_discrete_sequence=["#7c6ef5", "#f5836e"])
                st.plotly_chart(styled_plotly(fig_dist), use_container_width=True)

            # ── Feature Importances (RF only)
            if res["feat_importances"] is not None and res["feat_names"] is not None:
                st.markdown('<p class="section-header">Feature Importances</p>', unsafe_allow_html=True)
                fi_df = pd.DataFrame({
                    "feature": res["feat_names"],
                    "importance": res["feat_importances"]
                }).sort_values("importance", ascending=False).head(20)

                fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                title="Top 20 Feature Importances",
                                color="importance",
                                color_continuous_scale=[[0, "#1a1d27"], [1, "#7c6ef5"]])
                fig_fi.update_layout(**PLOTLY_THEME, yaxis=dict(autorange="reversed"),
                                     coloraxis_showscale=False, height=520)
                fig_fi.update_xaxes(gridcolor="#1e2132"); fig_fi.update_yaxes(gridcolor="#1e2132")
                st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
            # 🔮 UNIFIED INTERACTIVE PREDICTOR
            # ─────────────────────────────────────────────
            if st.session_state.model_obj is not None:
                st.markdown("---")
                
                # Dynamic titles based on whether it's predicting price (regression) or room type (classification)
                title_text = "Price Predictor" if res["task"] == "regression" else "Category Predictor"
                st.markdown(f'<p class="section-header">🔮 Interactive {title_text}</p>', unsafe_allow_html=True)
                st.markdown(f"<p style='color:#8a8fa8; font-size:0.9rem;'>Enter listing details below to predict the <b>{res['target_col']}</b> based on your trained model.</p>", unsafe_allow_html=True)

                expected_features = res["X_test"].columns.tolist()

                # 1. Define allowed columns based on the target task
                if res["task"] == "regression":
                    # Predicting Price
                    allowed_cols = ["room_type", "minimum_nights", "availability_365", "number_of_reviews", "calculated_host_listings_count"]
                else:
                    # Predicting Room Type
                    allowed_cols = ["price", "minimum_nights", "calculated_host_listings_count", "availability_365", "number_of_reviews"]

                # Filter to only show columns that actually exist in the trained model's features
                visible_features = [col for col in allowed_cols if col in expected_features]

                with st.form("prediction_form"):
                    cols = st.columns(3)
                    input_data = {}

                    # 2. Render only the allowed visible features
                    for i, col_name in enumerate(visible_features):
                        c = cols[i % 3]
                        display_name = col_name.replace('_', ' ').title()
                        
                        # Create number inputs for numerical columns
                        if pd.api.types.is_numeric_dtype(res["X_test"][col_name]):
                            mean_val = float(res["X_test"][col_name].mean())
                            
                            # Clean up formatting for integer-based columns
                            if col_name in ["minimum_nights", "number_of_reviews", "calculated_host_listings_count", "availability_365"]:
                                input_data[col_name] = c.number_input(f"{display_name}", min_value=0.0, value=float(int(mean_val)), step=1.0)
                            elif col_name == "price":
                                input_data[col_name] = c.number_input(f"{display_name}", min_value=0.0, value=mean_val, step=10.0)
                            else:
                                input_data[col_name] = c.number_input(f"{display_name}", value=mean_val)
                        
                        # Create selectboxes for categorical columns
                        else:
                            unique_vals = res["X_test"][col_name].dropna().unique().tolist()
                            input_data[col_name] = c.selectbox(f"{display_name}", options=unique_vals)

                    submit = st.form_submit_button(f"Predict {res['target_col'].replace('_', ' ').title()}")

                    if submit:
                        # 3. Construct the full data frame, filling hidden features with defaults
                        full_input_data = {}
                        for col in expected_features:
                            if col in visible_features:
                                full_input_data[col] = input_data[col]
                            else:
                                # Fill missing numeric columns with their mean
                                if pd.api.types.is_numeric_dtype(res["X_test"][col]):
                                    full_input_data[col] = float(res["X_test"][col].mean())
                                # Fill missing categorical columns with their mode
                                else:
                                    mode_series = res["X_test"][col].mode()
                                    full_input_data[col] = mode_series[0] if not mode_series.empty else "Unknown"

                        input_df = pd.DataFrame([full_input_data])
                        
                        try:
                            predicted_val = st.session_state.model_obj.predict(input_df)[0]
                            
                            if res["task"] == "regression":
                                # Handle log transformations and formatting for Regression (Price)
                                if res["target_col"] == "log_price" and not res.get("is_log_transform", False):
                                    final_val = np.expm1(predicted_val)
                                    prefix = "$"
                                elif res["target_col"] == "price":
                                    final_val = predicted_val
                                    prefix = "$"
                                else:
                                    final_val = predicted_val
                                    prefix = ""
                                    
                                final_val = max(0, final_val) # Prevent negative prices
                                display_text = f"{prefix}{final_val:,.2f}"
                                label_text = f"Estimated {res['target_col'].replace('_', ' ').title()}"
                                
                            else:
                                # Formatting for Classification (Room Type)
                                display_text = str(predicted_val).upper()
                                label_text = f"Predicted {res['target_col'].replace('_', ' ').title()}"

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(124, 110, 245, 0.2) 0%, rgba(91, 78, 212, 0.2) 100%); 
                                        border: 1px solid #7c6ef5; border-radius: 10px; padding: 1.5rem; text-align: center; margin-top: 1rem;">
                                <p style="color: #c8ccde; font-size: 1rem; margin: 0; font-family: 'DM Sans', sans-serif;">{label_text}</p>
                                <h2 style="color: #e8eaf2; font-size: 2.5rem; font-family: 'Syne', sans-serif; margin: 0;">{display_text}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}. Ensure all data transformations align with the input.")