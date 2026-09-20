# 🎓 Student Habits and Academic Performance Analysis

> An interactive data analytics dashboard exploring how students' daily habits and lifestyle factors relate to their academic performance.

---

## 📋 Project Overview

This project performs a comprehensive exploratory data analysis (EDA) on a dataset of 1,000 university students, examining the relationships between various lifestyle habits (study hours, sleep, social media usage, diet, exercise, and more) and academic exam scores. The analysis is presented through a modern, interactive Streamlit dashboard powered by Plotly visualisations.

---

## ❓ Problem Statement

Academic performance is influenced by a complex interplay of factors beyond raw academic ability. Understanding which daily habits and lifestyle attributes are most strongly *associated* with exam scores can help students, educators, and institutions make more informed decisions. This project aims to surface those patterns and associations through data-driven analysis.

---

## 🎯 Objectives

1. Load, inspect, and clean the student habits dataset.
2. Perform exploratory data analysis and descriptive statistics.
3. Analyse correlations between lifestyle variables and exam scores.
4. Identify meaningful patterns across demographic and behavioural groups.
5. Present findings through an interactive, filterable analytics dashboard.
6. Communicate key insights and associations clearly and responsibly.

---

## 📂 Dataset Description

| Attribute | Description |
|---|---|
| `student_id` | Unique student identifier |
| `age` | Student age (17–24) |
| `gender` | Gender (Male / Female / Other) |
| `study_hours_per_day` | Daily study hours (0–8.3) |
| `social_media_hours` | Daily social media hours (0–7.2) |
| `netflix_hours` | Daily Netflix/streaming hours (0–5.4) |
| `part_time_job` | Whether the student works part-time (Yes/No) |
| `attendance_percentage` | Class attendance rate (56–100%) |
| `sleep_hours` | Nightly sleep hours (3.2–10) |
| `diet_quality` | Self-reported diet quality (Poor/Fair/Good) |
| `exercise_frequency` | Exercise days per week (0–6) |
| `parental_education_level` | Highest parental education (None/High School/Bachelor/Master) |
| `internet_quality` | Internet connection quality (Poor/Average/Good) |
| `mental_health_rating` | Self-rated mental health (1–10) |
| `extracurricular_participation` | Extracurricular activity participation (Yes/No) |
| `exam_score` | Final exam score (18.4–100.0) — **target variable** |

- **Records:** 1,000 students  
- **Missing values:** 91 in `parental_education_level` — imputed with mode during preprocessing
- **No duplicate records**

---

## 🔗 Dataset Source

[Kaggle — Student Habits vs Academic Performance (sonia212)](https://www.kaggle.com/datasets/sonia212/student-habits-vs-academic-performance)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Streamlit | Interactive web dashboard framework |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Plotly | Interactive visualisations |
| scikit-learn | Data normalisation (MinMaxScaler for radar charts) |
| statsmodels | OLS trendline support (via Plotly trendline="ols") |

---

## ✨ Features

- **Interactive sidebar filters** — filter by gender, part-time job, diet, internet quality, parental education, and extracurricular participation
- **KPI summary cards** — at-a-glance metrics for the filtered cohort
- **5-tab dashboard** covering Overview, Academic Performance, Habit Analysis, Correlations, and Key Insights
- **20+ Plotly charts** including scatter plots (with trendlines), bar charts, box plots, heatmaps, donut charts, scatter matrices, and radar charts
- **Performance tier segmentation** — Low / Below Average / Average / High
- **Downloadable filtered dataset**
- **Descriptive statistics table**
- **Written insights panel** with computed statistics

---

## 📁 Project Structure

```
archive/
├── app.py                            # Main Streamlit dashboard application
├── student_habits_performance.csv    # Dataset
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── Student_Habits_Performance_Report.docx  # Project report
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or download the project files into a directory
cd "path/to/project"

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

> **Note:** The `student_habits_performance.csv` file must be in the same directory as `app.py`.

---

## 📊 Dashboard Features

### Tab 1 — Overview & KPIs
- 7 KPI cards: student count, average exam score, study hours, attendance, sleep, screen time, % high performers
- Performance tier donut chart
- Exam score distribution histogram
- Average score by gender and part-time job status

### Tab 2 — Academic Performance
- Average score by parental education level
- Average score by diet quality
- Average score by internet quality
- Average score by extracurricular participation
- Box plot distributions for education and diet
- Attendance vs exam score scatter (with trendline)

### Tab 3 — Habit Analysis
- Study hours vs exam score scatter
- Sleep hours vs exam score scatter
- Social media hours vs exam score scatter
- Exercise frequency vs exam score box plot
- Mental health rating vs exam score box plot
- Radar chart — normalised habit profile by performance tier
- Stacked screen time bar by performance tier
- Productive study ratio bar by performance tier

### Tab 4 — Correlations
- Full Pearson correlation heatmap
- Horizontal bar chart of correlations with exam score
- Scatter matrix for top variables

### Tab 5 — Key Insights
- 8 written insight cards with computed statistics
- Full descriptive statistics table
- CSV download button for filtered data

---

## 💡 Key Analytical Insights

1. **Study hours** have the strongest positive association with exam scores. Students studying ≥5 h/day score ~18 points higher than those studying <2 h/day on average.
2. **Attendance** is strongly positively associated with exam scores; students with ≥90% attendance significantly outperform peers with <75% attendance.
3. **Screen time** (social media + Netflix) is negatively associated with scores — high performers allocate less time to passive screen activities.
4. **Mental health** shows a positive association; students with higher wellbeing ratings tend to score better.
5. **Sleep** has a nuanced relationship — both extremes (too little or too much) are associated with lower scores.
6. **Parental education** shows a positive gradient; children of Master's-degree holders tend to outperform those of parents with no formal education.
7. **Diet quality** and **internet quality** show moderate positive associations with scores.

> *All relationships are reported as associations/correlations. No causal claims are made.*

---

## ⚠️ Limitations

- The dataset is self-reported, which may introduce response bias.
- The sample is limited to 1,000 students from one (unspecified) institution/region, limiting generalisability.
- No causal mechanisms can be inferred from correlational analysis alone.
- Important confounding factors (e.g., socioeconomic status, prior academic ability) are not captured.
- Cross-sectional design prevents tracking changes over time.

---

## 🚀 Future Scope

- Expand dataset size and diversity (multiple institutions, countries).
- Apply machine learning models (regression, classification) to predict exam scores.
- Longitudinal tracking of habit changes and their impact on grades.
- A/B testing of interventions (e.g., structured study schedules).
- Integration with learning management system data for richer features.
- Cluster analysis to identify distinct student lifestyle profiles.

---

## 📄 License

This project is for educational and analytical purposes. Dataset credit: [sonia212 on Kaggle](https://www.kaggle.com/datasets/sonia212/student-habits-vs-academic-performance).
