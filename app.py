import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "dataset")
MODEL_DIR = os.path.join(APP_DIR, "models")
STUDENTS_PATH = os.path.join(DATA_DIR, "students.csv")
RESOURCES_PATH = os.path.join(DATA_DIR, "learning_resources.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "resource_rating_model.joblib")

st.set_page_config(page_title="Personalized Learning Recommender", page_icon="📚", layout="wide")

@st.cache_data
def load_data():
    students = pd.read_csv(STUDENTS_PATH)
    resources = pd.read_csv(RESOURCES_PATH)
    return students, resources

def train_model(students):
    # This demo predicts resource_rating from student-profile features.
    # The included ratings are synthetic, so evaluation is a demonstration only.
    feature_cols = [
        "subject", "topic", "previous_score", "study_hours_per_day",
        "attendance_pct", "interest_level", "learning_preference",
        "completion_rate_pct", "difficulty_level"
    ]
    X = students[feature_cols].copy()
    y = students["resource_rating"].astype(float)

    categorical = ["subject", "topic", "learning_preference", "difficulty_level"]
    numerical = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
            ("numerical", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numerical)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": r2_score(y_test, predictions)
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, metrics


def get_model(students):
    # Train the model and calculate evaluation metrics.
    # This also updates the previously saved model.
    model, metrics = train_model(students)
    return model, metrics


def resource_level_for_score(score):
    if score < 50:
        return "Beginner"
    elif score < 75:
        return "Intermediate"
    return "Advanced"

def recommend_resources(profile, resources, model, top_n=5):
    # Match subject/topic first, then rank available resources using preference,
    # difficulty suitability, and predicted rating. Fall back to the selected subject
    # if no resources are listed for the exact topic.
    candidates = resources[resources["subject"] == profile["subject"]].copy()
    exact_topic = candidates[candidates["topic"] == profile["topic"]]
    if not exact_topic.empty:
        candidates = exact_topic.copy()
    if candidates.empty:
        candidates = resources.copy()

    target_level = resource_level_for_score(profile["previous_score"])
    level_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    target_rank = level_order[target_level]

    features = pd.DataFrame([{
        "subject": profile["subject"],
        "topic": profile["topic"],
        "previous_score": profile["previous_score"],
        "study_hours_per_day": profile["study_hours_per_day"],
        "attendance_pct": profile["attendance_pct"],
        "interest_level": profile["interest_level"],
        "learning_preference": profile["learning_preference"],
        "completion_rate_pct": profile["completion_rate_pct"],
        "difficulty_level": profile["difficulty_level"]
    }])
    predicted_rating = float(model.predict(features)[0])

    candidates["preference_match"] = (
        candidates["resource_type"] == profile["learning_preference"]
    ).astype(int)
    candidates["level_distance"] = candidates["difficulty_level"].map(
        lambda x: abs(level_order.get(x, 1) - target_rank)
    )
    # Ranking score: preference and suitable difficulty plus model-predicted rating.
    # This is a transparent demo ranking, not a validated educational score.
    candidates["predicted_rating"] = predicted_rating
    candidates["ranking_score"] = (
        0.55 * candidates["preference_match"]
        + 0.25 * (1 / (1 + candidates["level_distance"]))
        + 0.20 * (predicted_rating / 5.0)
    )
    candidates = candidates.sort_values(
        ["ranking_score", "resource_title"], ascending=[False, True]
    ).head(top_n).copy()

    def reason(row):
        reasons = []
        if row["topic"] == profile["topic"]:
            reasons.append("matches your selected topic")
        if row["resource_type"] == profile["learning_preference"]:
            reasons.append("matches your learning preference")
        if row["difficulty_level"] == target_level:
            reasons.append("fits the suggested level")
        if not reasons:
            reasons.append("is another resource in your selected subject")
        return "; ".join(reasons).capitalize() + "."
    candidates["reason"] = candidates.apply(reason, axis=1)
    return candidates

try:
    students_df, resources_df = load_data()
except FileNotFoundError:
    st.error("Dataset files were not found. Keep students.csv and learning_resources.csv inside the dataset folder.")
    st.stop()

st.title("📚 Personalized Learning Recommendation System")
st.caption("A machine-learning demo that ranks learning resources using a student profile.")
st.info(
    "Project note: the included student records and ratings are synthetic sample data. "
    "This app demonstrates the workflow; its results are not evidence of real-world learning gains."
)

with st.sidebar:
    st.header("Student Profile")
    subject = st.selectbox("Choose subject", sorted(resources_df["subject"].unique()))
    topics_for_subject = sorted(resources_df.loc[resources_df["subject"] == subject, "topic"].unique())
    topic = st.selectbox("Topic to study", topics_for_subject)
    previous_score = st.slider("Previous score (%)", 0, 100, 55)
    study_hours = st.slider("Study hours per day", 0.0, 8.0, 2.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    interest = st.slider("Interest level (1 = low, 5 = high)", 1, 5, 3)
    preference = st.selectbox("Preferred resource type", ["Video", "Notes", "Practice", "Interactive"])
    completion = st.slider("Topic completion (%)", 0, 100, 40)
    difficulty = st.selectbox("How difficult is this topic?", ["Low", "Medium", "High"])
    number_of_recommendations = st.slider("Number of recommendations", 3, 10, 5)
    generate = st.button("Generate Recommendations", type="primary", use_container_width=True)

profile = {
    "subject": subject,
    "topic": topic,
    "previous_score": previous_score,
    "study_hours_per_day": study_hours,
    "attendance_pct": attendance,
    "interest_level": interest,
    "learning_preference": preference,
    "completion_rate_pct": completion,
    "difficulty_level": difficulty
}

tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Dataset & EDA", "Model Evaluation", "About"])

with tab1:
    st.subheader("Your learning recommendations")
    if generate or "recommendations" not in st.session_state:
        with st.spinner("Preparing recommendations..."):
            model, metrics = get_model(students_df)
            recs = recommend_resources(profile, resources_df, model, number_of_recommendations)
            st.session_state["recommendations"] = recs
            st.session_state["metrics"] = metrics
            st.session_state["model"] = model
            st.session_state["last_profile"] = profile
    else:
        recs = st.session_state["recommendations"]

    if generate or "recommendations" in st.session_state:
        recs = st.session_state["recommendations"]
        st.write(
            f"**Selected profile:** {profile['subject']} → {profile['topic']} | "
            f"Score: {profile['previous_score']}% | Preference: {profile['learning_preference']}"
        )
        display_cols = ["resource_title", "topic", "resource_type", "difficulty_level", "reason"]
        st.dataframe(
            recs[display_cols].rename(columns={
                "resource_title": "Resource",
                "topic": "Topic",
                "resource_type": "Format",
                "difficulty_level": "Level",
                "reason": "Why recommended"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "Download recommendations as CSV",
            data=recs[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="my_learning_recommendations.csv",
            mime="text/csv"
        )
        st.caption("Tip: change the profile in the sidebar and click Generate Recommendations to refresh the list.")

with tab2:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Student records", len(students_df))
    c2.metric("Learning resources", len(resources_df))
    c3.metric("Subjects", resources_df["subject"].nunique())

    st.markdown("**Preview: student dataset**")
    st.dataframe(students_df.head(10), use_container_width=True, hide_index=True)
    st.markdown("**Missing values by column**")
    missing = students_df.isna().sum().rename("missing_values").reset_index()
    missing.columns = ["column", "missing_values"]
    st.dataframe(missing, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.hist(students_df["previous_score"], bins=12, edgecolor="black")
        ax.set_title("Previous Score Distribution")
        ax.set_xlabel("Score (%)")
        ax.set_ylabel("Number of records")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        pref_counts = students_df["learning_preference"].value_counts()
        fig, ax = plt.subplots()
        pref_counts.plot(kind="bar", ax=ax)
        ax.set_title("Learning Preferences")
        ax.set_xlabel("Preference")
        ax.set_ylabel("Number of records")
        plt.xticks(rotation=25)
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.scatter(students_df["study_hours_per_day"], students_df["previous_score"], alpha=0.5)
        ax.set_title("Study Hours vs Previous Score")
        ax.set_xlabel("Study hours per day")
        ax.set_ylabel("Previous score (%)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        avg_score = students_df.groupby("subject")["previous_score"].mean().sort_values()
        fig, ax = plt.subplots()
        avg_score.plot(kind="barh", ax=ax)
        ax.set_title("Average Score by Subject")
        ax.set_xlabel("Average score (%)")
        ax.set_ylabel("Subject")
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_cols = ["previous_score", "study_hours_per_day", "attendance_pct",
                    "interest_level", "completion_rate_pct", "resource_rating"]
    sns.heatmap(students_df[numeric_cols].corr(numeric_only=True), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Model evaluation")
    if "metrics" not in st.session_state:
        model, metrics = get_model(students_df)
        st.session_state["model"] = model
        st.session_state["metrics"] = metrics
    metrics = st.session_state["metrics"]
    st.write("A Random Forest Regressor predicts the synthetic resource-rating target using an 80/20 train-test split.")
    st.warning("Because the included target ratings are synthetic, these metrics only demonstrate evaluation code. They do not validate educational effectiveness.")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.3f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c3.metric("R²", f"{metrics['R2']:.3f}")
    st.markdown("**Metric meanings**")
    st.write("- MAE: average absolute difference between predicted and observed ratings.")
    st.write("- RMSE: prediction error metric that penalizes larger errors more strongly.")
    st.write("- R²: proportion of target variance explained on the held-out test set; it can be negative.")

with tab4:
    st.subheader("How this project works")
    st.markdown("""
    1. **Input:** The student selects a subject, topic, performance indicators, and learning preference.
    2. **Preprocessing:** Categorical variables are one-hot encoded; missing values are handled in a pipeline.
    3. **Model:** A Random Forest Regressor predicts the sample resource-rating target.
    4. **Ranking:** Candidate resources are ranked using the selected topic, learning preference, difficulty suitability, and predicted rating.
    5. **Evaluation:** MAE, RMSE, and R² are calculated on a held-out test set.
    6. **EDA:** The dashboard shows score distribution, preference counts, study-hour patterns, subject averages, and correlations.

    **Limitations:** This package uses synthetic demonstration data. Replace it with consented, appropriately anonymized real data and meaningful resource feedback before drawing conclusions about real students.
    """)
    st.markdown("**Files used**")
    st.write("`dataset/students.csv` — synthetic student-learning records")
    st.write("`dataset/learning_resources.csv` — example learning-resource catalogue")
