"""
Student Habits and Academic Performance Analysis
A complete Streamlit dashboard for interactive data analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Habits & Academic Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #f8f9fb; }
    /* KPI card style */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin: 0; }
    .kpi-label { font-size: 0.82rem; color: #6b7280; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
    /* Section headers */
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; margin-bottom: 16px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1e3a5f; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    /* Insight boxes */
    .insight-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 12px 16px; border-radius: 6px; margin: 8px 0;
        font-size: 0.9rem; color: #1e3a5f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv("student_habits_performance.csv")

    # ── Data quality checks ──
    # Drop full duplicates (none expected, but defensive)
    df.drop_duplicates(inplace=True)

    # Drop student_id column (not analytical)
    df.drop(columns=["student_id"], inplace=True)

    # Ensure correct dtypes
    num_cols = [
        "age", "study_hours_per_day", "social_media_hours", "netflix_hours",
        "attendance_percentage", "sleep_hours", "exercise_frequency",
        "mental_health_rating", "exam_score",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle any remaining NaN by imputing with median (numerical) / mode (categorical)
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # ── Derived / engineered features ──
    # Performance tier
    df["performance_tier"] = pd.cut(
        df["exam_score"],
        bins=[0, 50, 65, 80, 100],
        labels=["Low (<50)", "Below Average (50-65)", "Average (65-80)", "High (>80)"],
        right=True,
    )

    # Total screen time
    df["screen_time"] = df["social_media_hours"] + df["netflix_hours"]

    # Productive hours ratio (study / (study + screen_time))
    df["productive_ratio"] = df["study_hours_per_day"] / (
        df["study_hours_per_day"] + df["screen_time"] + 1e-9
    )

    # Ordered categoricals for nice chart axes
    df["diet_quality"] = pd.Categorical(
        df["diet_quality"], categories=["Poor", "Fair", "Good"], ordered=True
    )
    df["internet_quality"] = pd.Categorical(
        df["internet_quality"], categories=["Poor", "Average", "Good"], ordered=True
    )
    df["parental_education_level"] = pd.Categorical(
        df["parental_education_level"],
        categories=["None", "High School", "Bachelor", "Master"],
        ordered=True,
    )

    return df


df_full = load_data()

# ──────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Analytics")
    st.markdown("### Filters")

    gender_opts = sorted(df_full["gender"].unique().tolist())
    sel_gender = st.multiselect("Gender", gender_opts, default=gender_opts)

    part_time_opts = ["Yes", "No"]
    sel_part_time = st.multiselect("Part-time Job", part_time_opts, default=part_time_opts)

    diet_opts = ["Poor", "Fair", "Good"]
    sel_diet = st.multiselect("Diet Quality", diet_opts, default=diet_opts)

    internet_opts = ["Poor", "Average", "Good"]
    sel_internet = st.multiselect("Internet Quality", internet_opts, default=internet_opts)

    edu_opts = ["None", "High School", "Bachelor", "Master"]
    sel_edu = st.multiselect("Parental Education", edu_opts, default=edu_opts)

    extracurr_opts = ["Yes", "No"]
    sel_extracurr = st.multiselect("Extracurricular Activities", extracurr_opts, default=extracurr_opts)

    st.markdown("---")
    st.markdown(
        "<small style='color:#94a3b8'>Dataset: 1,000 Students<br>"
        "Source: Kaggle — sonia212</small>",
        unsafe_allow_html=True,
    )

# ── Apply filters ──
@st.cache_data
def filter_data(df, genders, part_times, diets, internets, edus, extracurrs):
    mask = (
        df["gender"].isin(genders)
        & df["part_time_job"].isin(part_times)
        & df["diet_quality"].astype(str).isin(diets)
        & df["internet_quality"].astype(str).isin(internets)
        & df["parental_education_level"].astype(str).isin(edus)
        & df["extracurricular_participation"].isin(extracurrs)
    )
    return df[mask].copy()


df = filter_data(
    df_full,
    sel_gender, sel_part_time, sel_diet, sel_internet, sel_edu, sel_extracurr,
)

# Guard against empty filter result
if df.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar filters.")
    st.stop()

# ──────────────────────────────────────────────
# HELPER COLOUR PALETTE
# ──────────────────────────────────────────────
COLOR_SEQ = px.colors.qualitative.Safe
BLUE = "#3b82f6"
GREEN = "#10b981"
ORANGE = "#f59e0b"
RED = "#ef4444"
PURPLE = "#8b5cf6"

# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#1e3a5f; margin-bottom:0'>🎓 Student Habits & Academic Performance</h1>"
    "<p style='color:#6b7280; font-size:0.95rem; margin-top:4px'>"
    "Interactive analysis of lifestyle habits and their associations with academic outcomes</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────
# TAB LAYOUT
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview & KPIs", "📚 Academic Performance", "🏃 Habit Analysis", "🔗 Correlations", "💡 Key Insights"]
)

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW & KPIs
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    avg_score = df["exam_score"].mean()
    avg_study = df["study_hours_per_day"].mean()
    avg_attend = df["attendance_percentage"].mean()
    avg_sleep = df["sleep_hours"].mean()
    avg_screen = df["screen_time"].mean()
    pct_high = (df["exam_score"] >= 80).mean() * 100
    n_students = len(df)

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

    def kpi(col, value, label, color="#3b82f6"):
        col.markdown(
            f"<div class='kpi-card' style='border-left-color:{color}'>"
            f"<p class='kpi-value'>{value}</p>"
            f"<p class='kpi-label'>{label}</p></div>",
            unsafe_allow_html=True,
        )

    kpi(k1, f"{n_students:,}", "Students", BLUE)
    kpi(k2, f"{avg_score:.1f}", "Avg Exam Score", GREEN)
    kpi(k3, f"{avg_study:.1f}h", "Avg Study/Day", PURPLE)
    kpi(k4, f"{avg_attend:.1f}%", "Avg Attendance", ORANGE)
    kpi(k5, f"{avg_sleep:.1f}h", "Avg Sleep/Night", "#06b6d4")
    kpi(k6, f"{avg_screen:.1f}h", "Avg Screen Time", RED)
    kpi(k7, f"{pct_high:.1f}%", "High Performers (≥80)", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Performance tier donut
    tier_counts = df["performance_tier"].value_counts().sort_index()
    fig_donut = px.pie(
        names=tier_counts.index.astype(str),
        values=tier_counts.values,
        hole=0.55,
        title="Performance Tier Distribution",
        color_discrete_sequence=[RED, ORANGE, BLUE, GREEN],
    )
    fig_donut.update_traces(textposition="outside", textinfo="percent+label")
    fig_donut.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    c1.plotly_chart(fig_donut, use_container_width=True)

    # Exam score histogram
    fig_hist = px.histogram(
        df, x="exam_score", nbins=30,
        title="Exam Score Distribution",
        labels={"exam_score": "Exam Score"},
        color_discrete_sequence=[BLUE],
    )
    fig_hist.add_vline(x=avg_score, line_dash="dash", line_color=RED,
                       annotation_text=f"Mean: {avg_score:.1f}", annotation_position="top right")
    fig_hist.update_layout(bargap=0.05, xaxis_title="Exam Score", yaxis_title="Count", margin=dict(t=50))
    c2.plotly_chart(fig_hist, use_container_width=True)

    # Gender breakdown
    c3, c4 = st.columns(2)
    gender_score = df.groupby("gender")["exam_score"].agg(["mean", "count"]).reset_index()
    gender_score.columns = ["Gender", "Avg Score", "Count"]
    fig_gender = px.bar(
        gender_score, x="Gender", y="Avg Score",
        title="Average Score by Gender",
        color="Gender", color_discrete_sequence=COLOR_SEQ,
        text="Avg Score",
    )
    fig_gender.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_gender.update_layout(showlegend=False, yaxis_range=[0, 110], margin=dict(t=50))
    c3.plotly_chart(fig_gender, use_container_width=True)

    # Part-time job
    job_score = df.groupby("part_time_job")["exam_score"].mean().reset_index()
    job_score.columns = ["Part-time Job", "Avg Score"]
    fig_job = px.bar(
        job_score, x="Part-time Job", y="Avg Score",
        title="Average Score: Part-time Job",
        color="Part-time Job", color_discrete_sequence=[GREEN, RED],
        text="Avg Score",
    )
    fig_job.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_job.update_layout(showlegend=False, yaxis_range=[0, 110], margin=dict(t=50))
    c4.plotly_chart(fig_job, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — ACADEMIC PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Academic Performance Deep-Dive</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Exam score by parental education
    edu_order = ["None", "High School", "Bachelor", "Master"]
    edu_score = (
        df.groupby("parental_education_level", observed=True)["exam_score"]
        .mean()
        .reindex(edu_order)
        .reset_index()
    )
    edu_score.columns = ["Education Level", "Avg Score"]
    fig_edu = px.bar(
        edu_score, x="Education Level", y="Avg Score",
        title="Average Score by Parental Education Level",
        color="Avg Score", color_continuous_scale="Blues",
        text="Avg Score",
    )
    fig_edu.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_edu.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110], margin=dict(t=50))
    c1.plotly_chart(fig_edu, use_container_width=True)

    # Exam score by diet quality
    diet_score = (
        df.groupby("diet_quality", observed=True)["exam_score"]
        .mean()
        .reindex(["Poor", "Fair", "Good"])
        .reset_index()
    )
    diet_score.columns = ["Diet Quality", "Avg Score"]
    fig_diet = px.bar(
        diet_score, x="Diet Quality", y="Avg Score",
        title="Average Score by Diet Quality",
        color="Diet Quality",
        color_discrete_map={"Poor": RED, "Fair": ORANGE, "Good": GREEN},
        text="Avg Score",
    )
    fig_diet.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_diet.update_layout(showlegend=False, yaxis_range=[0, 110], margin=dict(t=50))
    c2.plotly_chart(fig_diet, use_container_width=True)

    c3, c4 = st.columns(2)

    # Score by internet quality
    inet_score = (
        df.groupby("internet_quality", observed=True)["exam_score"]
        .mean()
        .reindex(["Poor", "Average", "Good"])
        .reset_index()
    )
    inet_score.columns = ["Internet Quality", "Avg Score"]
    fig_inet = px.bar(
        inet_score, x="Internet Quality", y="Avg Score",
        title="Average Score by Internet Quality",
        color="Internet Quality",
        color_discrete_map={"Poor": RED, "Average": ORANGE, "Good": GREEN},
        text="Avg Score",
    )
    fig_inet.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_inet.update_layout(showlegend=False, yaxis_range=[0, 110], margin=dict(t=50))
    c3.plotly_chart(fig_inet, use_container_width=True)

    # Score by extracurricular
    extra_score = df.groupby("extracurricular_participation")["exam_score"].mean().reset_index()
    extra_score.columns = ["Extracurricular", "Avg Score"]
    fig_extra = px.bar(
        extra_score, x="Extracurricular", y="Avg Score",
        title="Average Score: Extracurricular Participation",
        color="Extracurricular",
        color_discrete_map={"Yes": GREEN, "No": ORANGE},
        text="Avg Score",
    )
    fig_extra.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_extra.update_layout(showlegend=False, yaxis_range=[0, 110], margin=dict(t=50))
    c4.plotly_chart(fig_extra, use_container_width=True)

    # Box plots — score distribution by tier
    st.markdown("<div class='section-header'>Score Distribution by Categorical Variables</div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    fig_box1 = px.box(
        df, x="parental_education_level", y="exam_score",
        category_orders={"parental_education_level": edu_order},
        title="Score Distribution by Parental Education",
        color="parental_education_level", color_discrete_sequence=COLOR_SEQ,
    )
    fig_box1.update_layout(showlegend=False, xaxis_title="Parental Education", yaxis_title="Exam Score")
    c5.plotly_chart(fig_box1, use_container_width=True)

    fig_box2 = px.box(
        df, x="diet_quality", y="exam_score",
        category_orders={"diet_quality": ["Poor", "Fair", "Good"]},
        title="Score Distribution by Diet Quality",
        color="diet_quality",
        color_discrete_map={"Poor": RED, "Fair": ORANGE, "Good": GREEN},
    )
    fig_box2.update_layout(showlegend=False, xaxis_title="Diet Quality", yaxis_title="Exam Score")
    c6.plotly_chart(fig_box2, use_container_width=True)

    # Attendance vs score scatter
    st.markdown("<div class='section-header'>Attendance vs Exam Score</div>", unsafe_allow_html=True)
    fig_att = px.scatter(
        df, x="attendance_percentage", y="exam_score",
        color="performance_tier",
        color_discrete_sequence=[RED, ORANGE, BLUE, GREEN],
        title="Attendance Percentage vs Exam Score",
        labels={"attendance_percentage": "Attendance (%)", "exam_score": "Exam Score"},
        opacity=0.65, trendline="ols",
        trendline_scope="overall",
        trendline_color_override="darkred",
    )
    fig_att.update_layout(legend_title="Performance Tier")
    st.plotly_chart(fig_att, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — HABIT ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Lifestyle & Habit Patterns</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Study hours vs exam score
    fig_study = px.scatter(
        df, x="study_hours_per_day", y="exam_score",
        color="performance_tier",
        color_discrete_sequence=[RED, ORANGE, BLUE, GREEN],
        title="Study Hours per Day vs Exam Score",
        labels={"study_hours_per_day": "Study Hours/Day", "exam_score": "Exam Score"},
        opacity=0.65, trendline="ols",
        trendline_scope="overall",
        trendline_color_override="darkred",
    )
    fig_study.update_layout(legend_title="Performance Tier")
    c1.plotly_chart(fig_study, use_container_width=True)

    # Sleep hours vs exam score
    fig_sleep = px.scatter(
        df, x="sleep_hours", y="exam_score",
        color="performance_tier",
        color_discrete_sequence=[RED, ORANGE, BLUE, GREEN],
        title="Sleep Hours vs Exam Score",
        labels={"sleep_hours": "Sleep Hours/Night", "exam_score": "Exam Score"},
        opacity=0.65, trendline="ols",
        trendline_scope="overall",
        trendline_color_override="darkred",
    )
    fig_sleep.update_layout(legend_title="Performance Tier")
    c2.plotly_chart(fig_sleep, use_container_width=True)

    c3, c4 = st.columns(2)

    # Social media vs score
    fig_sm = px.scatter(
        df, x="social_media_hours", y="exam_score",
        color="performance_tier",
        color_discrete_sequence=[RED, ORANGE, BLUE, GREEN],
        title="Social Media Hours vs Exam Score",
        labels={"social_media_hours": "Social Media Hours/Day", "exam_score": "Exam Score"},
        opacity=0.65, trendline="ols",
        trendline_scope="overall",
        trendline_color_override="darkred",
    )
    fig_sm.update_layout(legend_title="Performance Tier")
    c3.plotly_chart(fig_sm, use_container_width=True)

    # Exercise frequency vs score
    fig_ex = px.box(
        df, x="exercise_frequency", y="exam_score",
        title="Exercise Frequency (days/week) vs Exam Score",
        color_discrete_sequence=[PURPLE],
        labels={"exercise_frequency": "Exercise Days/Week", "exam_score": "Exam Score"},
    )
    c4.plotly_chart(fig_ex, use_container_width=True)

    # Mental health vs score
    c5, c6 = st.columns(2)
    fig_mh = px.box(
        df, x="mental_health_rating", y="exam_score",
        title="Mental Health Rating vs Exam Score",
        labels={"mental_health_rating": "Mental Health Rating (1–10)", "exam_score": "Exam Score"},
        color_discrete_sequence=[BLUE],
    )
    c5.plotly_chart(fig_mh, use_container_width=True)

    # Average habit profile by performance tier
    habit_cols = [
        "study_hours_per_day", "social_media_hours", "netflix_hours",
        "sleep_hours", "exercise_frequency", "mental_health_rating",
        "attendance_percentage",
    ]
    tier_profile = (
        df.groupby("performance_tier", observed=True)[habit_cols]
        .mean()
        .reset_index()
    )

    # Normalise for radar
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    radar_data = tier_profile.copy()
    radar_data[habit_cols] = scaler.fit_transform(radar_data[habit_cols])

    fig_radar = go.Figure()
    colours = [RED, ORANGE, BLUE, GREEN]
    for i, row in radar_data.iterrows():
        tier_name = str(row["performance_tier"])
        values = [row[c] for c in habit_cols] + [row[habit_cols[0]]]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=[c.replace("_", " ").title() for c in habit_cols] + [habit_cols[0].replace("_", " ").title()],
                fill="toself",
                name=tier_name,
                line_color=colours[i],
                opacity=0.75,
            )
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalised Habit Profile by Performance Tier",
        legend_title="Tier",
    )
    c6.plotly_chart(fig_radar, use_container_width=True)

    # Screen time bar by tier
    st.markdown("<div class='section-header'>Screen Time & Productive Ratio by Performance Tier</div>", unsafe_allow_html=True)
    c7, c8 = st.columns(2)

    screen_tier = (
        df.groupby("performance_tier", observed=True)[["social_media_hours", "netflix_hours"]]
        .mean()
        .reset_index()
    )
    screen_tier_melted = screen_tier.melt(
        id_vars="performance_tier", var_name="Type", value_name="Hours"
    )
    fig_screen = px.bar(
        screen_tier_melted,
        x="performance_tier", y="Hours", color="Type",
        barmode="stack",
        color_discrete_map={"social_media_hours": ORANGE, "netflix_hours": RED},
        title="Avg Screen Time by Performance Tier",
        labels={"performance_tier": "Performance Tier", "Hours": "Hours/Day"},
    )
    fig_screen.update_layout(legend_title="Screen Type", xaxis_title="", margin=dict(t=50))
    c7.plotly_chart(fig_screen, use_container_width=True)

    prod_tier = (
        df.groupby("performance_tier", observed=True)["productive_ratio"]
        .mean()
        .reset_index()
    )
    fig_prod = px.bar(
        prod_tier, x="performance_tier", y="productive_ratio",
        title="Avg Productive Study Ratio by Performance Tier",
        color="productive_ratio", color_continuous_scale="Greens",
        labels={"performance_tier": "Performance Tier", "productive_ratio": "Productive Ratio"},
        text="productive_ratio",
    )
    fig_prod.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_prod.update_layout(coloraxis_showscale=False, xaxis_title="", margin=dict(t=50))
    c8.plotly_chart(fig_prod, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Correlation Analysis</div>", unsafe_allow_html=True)

    num_cols_corr = [
        "exam_score", "study_hours_per_day", "attendance_percentage",
        "sleep_hours", "social_media_hours", "netflix_hours", "screen_time",
        "exercise_frequency", "mental_health_rating", "age", "productive_ratio",
    ]
    corr_matrix = df[num_cols_corr].corr()

    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation Matrix",
        aspect="auto",
    )
    fig_heat.update_layout(margin=dict(t=60, b=40))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<div class='section-header'>Top Correlations with Exam Score</div>", unsafe_allow_html=True)

    exam_corr = (
        corr_matrix["exam_score"]
        .drop("exam_score")
        .sort_values()
    )
    colours_bar = [RED if v < 0 else GREEN for v in exam_corr.values]
    fig_corr_bar = go.Figure(
        go.Bar(
            x=exam_corr.values,
            y=exam_corr.index.str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=colours_bar,
            text=[f"{v:.3f}" for v in exam_corr.values],
            textposition="outside",
        )
    )
    fig_corr_bar.update_layout(
        title="Correlation with Exam Score (Pearson r)",
        xaxis_title="Correlation Coefficient",
        xaxis=dict(range=[-1, 1]),
        margin=dict(t=50, l=180),
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    # Pairwise scatter matrix for top variables
    st.markdown("<div class='section-header'>Pairwise Relationships — Top Variables</div>", unsafe_allow_html=True)
    top_vars = ["exam_score", "study_hours_per_day", "attendance_percentage",
                "sleep_hours", "social_media_hours", "mental_health_rating"]
    fig_pair = px.scatter_matrix(
        df,
        dimensions=top_vars,
        color="performance_tier",
        color_discrete_sequence=[RED, ORANGE, BLUE, GREEN],
        title="Scatter Matrix — Key Variables",
        opacity=0.5,
    )
    fig_pair.update_traces(marker=dict(size=3), diagonal_visible=False)
    fig_pair.update_layout(height=700)
    st.plotly_chart(fig_pair, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — KEY INSIGHTS
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Key Analytical Findings & Insights</div>", unsafe_allow_html=True)

    # Recompute stats from the (possibly filtered) df
    corr_study = df["exam_score"].corr(df["study_hours_per_day"])
    corr_attend = df["exam_score"].corr(df["attendance_percentage"])
    corr_social = df["exam_score"].corr(df["social_media_hours"])
    corr_sleep = df["exam_score"].corr(df["sleep_hours"])
    corr_mental = df["exam_score"].corr(df["mental_health_rating"])
    corr_screen = df["exam_score"].corr(df["screen_time"])

    high_study_score = df[df["study_hours_per_day"] >= 5]["exam_score"].mean()
    low_study_score = df[df["study_hours_per_day"] < 2]["exam_score"].mean()
    high_attend_score = df[df["attendance_percentage"] >= 90]["exam_score"].mean()
    low_attend_score = df[df["attendance_percentage"] < 75]["exam_score"].mean()

    insights = [
        (
            "📖 Study Hours — Strongest Positive Association",
            f"Study hours per day show the strongest positive correlation with exam scores "
            f"(r = {corr_study:.3f}). Students studying ≥5 hours average "
            f"{high_study_score:.1f} vs {low_study_score:.1f} for those studying <2 hours — "
            f"a gap of {high_study_score - low_study_score:.1f} points.",
        ),
        (
            "🏫 Attendance — Strong Positive Association",
            f"Attendance percentage correlates positively with exam scores (r = {corr_attend:.3f}). "
            f"Students with ≥90% attendance average {high_attend_score:.1f} vs {low_attend_score:.1f} "
            f"for those below 75% attendance.",
        ),
        (
            "📱 Screen Time — Negative Association",
            f"Total screen time (social media + Netflix) is negatively associated with exam scores "
            f"(r = {corr_screen:.3f}). Social media alone: r = {corr_social:.3f}. "
            f"High performers tend to allocate less time to passive screen activities.",
        ),
        (
            "🧠 Mental Health — Positive Association",
            f"Mental health rating has a positive correlation with exam scores (r = {corr_mental:.3f}). "
            f"Students with higher self-reported mental wellbeing tend to perform better academically.",
        ),
        (
            "😴 Sleep — Nuanced Relationship",
            f"Sleep hours show a correlation of r = {corr_sleep:.3f} with exam scores. "
            f"The relationship is nuanced — both very low (<5h) and excessive sleep (>9h) "
            f"are associated with lower scores, suggesting a moderate optimum (~6–8h).",
        ),
        (
            "🎓 Parental Education — Positive Gradient",
            f"Students whose parents hold a Master's degree average higher exam scores than those "
            f"with no parental education. This association may reflect access to academic support, "
            f"resources, or educational culture at home.",
        ),
        (
            "🥗 Diet Quality — Marginal Association",
            f"Good diet quality is associated with slightly higher exam scores compared to poor diet. "
            f"While the difference is moderate, it may interact with other lifestyle factors such "
            f"as sleep and exercise.",
        ),
        (
            "💻 Internet Quality — Enabling Factor",
            f"Students with good internet quality tend to score higher than those with poor connectivity. "
            f"This is consistent with better access to online study resources.",
        ),
    ]

    for title, body in insights:
        st.markdown(
            f"<div class='insight-box'><strong>{title}</strong><br>{body}</div>",
            unsafe_allow_html=True,
        )

    # Summary stats table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Descriptive Statistics Summary</div>", unsafe_allow_html=True)
    num_cols_desc = [
        "exam_score", "study_hours_per_day", "attendance_percentage",
        "sleep_hours", "social_media_hours", "netflix_hours",
        "exercise_frequency", "mental_health_rating",
    ]
    desc = df[num_cols_desc].describe().T.round(2)
    desc.index = desc.index.str.replace("_", " ").str.title()
    st.dataframe(desc, use_container_width=True)

    # Download filtered data
    st.markdown("<br>", unsafe_allow_html=True)
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Dataset (CSV)",
        data=csv_download,
        file_name="student_habits_filtered.csv",
        mime="text/csv",
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.8rem'>"
    "Student Habits & Academic Performance Dashboard · "
    "Data: Kaggle — sonia212 · Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True,
)
