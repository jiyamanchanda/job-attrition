
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AttriSense | HR Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F5F7FA;
    color: #1E293B;
    font-family: 'Arial', sans-serif;
}

/* Main content */
.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #142B3B;
}

[data-testid="stSidebar"] * {
    color: #EAF2F7 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B9CAD5 !important;
}

/* Headings */
h1 {
    color: #17324D !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #17324D !important;
    font-weight: 650 !important;
}

/* Paragraphs */
p {
    color: #64748B;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 22px;
    border: 1px solid #E6EBF0;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 14px !important;
}

[data-testid="stMetricValue"] {
    color: #17324D !important;
    font-weight: 700 !important;
}

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E6EBF0;
    border-radius: 12px;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #0F8B8D;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    min-height: 43px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #0B7476;
    color: white;
    border: none;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Form */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 24px;
    border: 1px solid #E6EBF0;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #E2E8F0;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #8493A5;
}

/* Hide default Streamlit footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("models/attrisense_model.pkl")


model_bundle = load_model()

model = model_bundle["model"]
threshold = model_bundle["threshold"]
features = model_bundle["features"]
background_data = model_bundle["background_data"]


@st.cache_resource
def load_shap_explainer():
    classifier = model.named_steps["classifier"]

    return shap.LinearExplainer(
        classifier,
        background_data
    )


explainer = load_shap_explainer()

@st.cache_resource
def display_shap_explanation(employee):

    st.markdown("### Why did the model make this prediction?")
    st.caption(
        "Individual feature contributions to the model's attrition prediction."
    )

    input_df = pd.DataFrame([employee])
    input_df = input_df[features]

    # Apply the same preprocessing used during training
    transformed = model.named_steps["imputer"].transform(input_df)
    transformed = model.named_steps["scaler"].transform(transformed)

    # Calculate SHAP contributions
    shap_values = explainer.shap_values(transformed)

    # Handle the single employee's SHAP values
    values = np.asarray(shap_values)

    if values.ndim == 2:
        values = values[0]

    explanation_df = pd.DataFrame({
        "Feature": features,
        "Contribution": values
    })

    explanation_df["Absolute"] = (
        explanation_df["Contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute",
        ascending=False
    ).head(10)

    explanation_df = explanation_df.sort_values(
        "Contribution"
    )

    # Visualize positive and negative contributions
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "#16A34A" if value < 0 else "#DC6262"
        for value in explanation_df["Contribution"]
    ]

    ax.barh(
        explanation_df["Feature"],
        explanation_df["Contribution"],
        color=colors
    )

    ax.axvline(0, color="#64748B", linewidth=0.8)

    ax.set_xlabel("SHAP contribution (log-odds)")
    ax.set_title("Individual Feature Contributions")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Positive contributions increase the model's attrition score; "
        "negative contributions decrease it. These are contributions "
        "in log-odds space, not percentage-point changes in probability."
    )

    positive = explanation_df[
        explanation_df["Contribution"] > 0
    ]

    negative = explanation_df[
        explanation_df["Contribution"] < 0
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Factors increasing attrition score**")

        if positive.empty:
            st.write("No positive contributions among the displayed features.")
        else:
            for _, row in positive.sort_values(
                "Contribution", ascending=False
            ).iterrows():
                st.write(
                    f"🔴 {row['Feature']}: "
                    f"+{row['Contribution']:.3f}"
                )

    with col2:
        st.markdown("**Factors decreasing attrition score**")

        if negative.empty:
            st.write("No negative contributions among the displayed features.")
        else:
            for _, row in negative.sort_values(
                "Contribution"
            ).iterrows():
                st.write(
                    f"🟢 {row['Feature']}: "
                    f"{row['Contribution']:.3f}"
                )

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 📊 AttriSense")
    st.caption("HR INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Attrition Prediction",
            "Model Insights",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### MODEL STATUS")

    st.success("Model loaded successfully")

    st.markdown(
        f"""
        **Algorithm:** Logistic Regression

        **Threshold:** {threshold:.0%}

        **Input Features:** {len(features)}
        """
    )

    st.divider()

    st.caption("24CSE0316 | Artificial Intelligence & Machine Learning")
    st.caption("AttriSense © 2026")


# =====================================================
# HEADER FUNCTION
# =====================================================

def page_header(title, subtitle):
    st.title(title)
    st.markdown(
        f"<p style='font-size:15px;color:#718096;margin-top:-10px;'>{subtitle}</p>",
        unsafe_allow_html=True
    )
    st.write("")


# =====================================================
# METRIC CARD FUNCTION
# =====================================================

def metric_card(label, value, description):
    st.metric(label, value, help=description)


# =====================================================
# EMPLOYEE INPUT FORM
# =====================================================

def employee_form():

    with st.form("employee_prediction_form"):

        st.markdown("### Personal & Professional Details")
        st.caption("Enter the employee's basic professional information.")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=65,
                value=30
            )

            job_level = st.selectbox(
                "Job Level",
                [1, 2, 3, 4, 5]
            )

            years_at_company = st.number_input(
                "Years at Company",
                min_value=0,
                max_value=50,
                value=5
            )

        with col2:
            monthly_income = st.number_input(
                "Monthly Income",
                min_value=0,
                value=5000
            )

            overtime = st.selectbox(
                "Overtime",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            job_satisfaction = st.select_slider(
                "Job Satisfaction",
                options=[1, 2, 3, 4],
                value=3
            )

        with col3:
            performance_rating = st.selectbox(
                "Performance Rating",
                [1, 2, 3, 4]
            )

            distance_from_home_km = st.number_input(
                "Distance From Home (km)",
                min_value=0.0,
                max_value=200.0,
                value=10.0
            )

            years_since_promotion = st.number_input(
                "Years Since Promotion",
                min_value=0,
                max_value=50,
                value=2
            )

        st.divider()

        st.markdown("### Workplace Information")
        st.caption("Specify the employee's department and working arrangement.")

        col1, col2, col3 = st.columns(3)

        with col1:
            department = st.selectbox(
                "Department",
                [
                    "Customer Support",
                    "Engineering",
                    "Finance",
                    "Human Resources",
                    "Marketing",
                    "Operations",
                    "Research",
                    "Sales"
                ]
            )

        with col2:
            work_mode = st.selectbox(
                "Work Mode",
                ["Hybrid", "Onsite", "Remote"]
            )

        with col3:
            hire_year = st.number_input(
                "Hire Year",
                min_value=1980,
                max_value=2026,
                value=2020
            )

        st.write("")

        submitted = st.form_submit_button(
            "Analyze Employee Risk →",
            use_container_width=True
        )

    if submitted:

        if years_since_promotion > years_at_company:
            st.error(
                "Years since promotion cannot exceed years at company."
            )
            return None

        current_year = 2026

        if hire_year > current_year:
            st.error("Hire year cannot be in the future.")
            return None

        if years_at_company > current_year - hire_year:
            st.error(
                "Years at company cannot exceed the time since the hire year."
            )
            return None

        input_data = {
            "job_level": job_level,
            "age": age,
            "years_at_company": years_at_company,
            "monthly_income": monthly_income,
            "overtime": overtime,
            "job_satisfaction": job_satisfaction,
            "performance_rating": performance_rating,
            "distance_from_home_km": distance_from_home_km,
            "years_since_promotion": years_since_promotion,
            "hire_year": hire_year
        }

        # Department encoding
        departments = [
            "Customer Support",
            "Engineering",
            "Finance",
            "Human Resources",
            "Marketing",
            "Operations",
            "Research",
            "Sales"
        ]

        for dept in departments:
            input_data[f"department_{dept}"] = int(
                department == dept
            )

        # Work mode encoding
        for mode in ["Hybrid", "Onsite", "Remote"]:
            input_data[f"work_mode_{mode}"] = int(
                work_mode == mode
            )

        # Feature engineering
        promotion_tenure_ratio = (
            years_since_promotion / years_at_company
            if years_at_company > 0 else 0
        )

        overtime_tenure = overtime * years_at_company

        input_data["promotion_tenure_ratio"] = promotion_tenure_ratio
        input_data["overtime_tenure"] = overtime_tenure

        input_df = pd.DataFrame([input_data])
        input_df = input_df[features]

        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability >= threshold)

        return {
            "probability": probability,
            "prediction": prediction,
            "employee": input_data
        }

    return None


# =====================================================
# PREDICTION RESULTS
# =====================================================

def display_prediction(result):

    if result is None:
        return

    probability = result["probability"]
    prediction = result["prediction"]
    employee = result["employee"]

    st.divider()

    st.markdown("## Employee Risk Assessment")
    st.caption("Individual attrition prediction generated by the ML model.")

    col1, col2 = st.columns([1, 1])

    with col1:

        with st.container(border=True):

            st.markdown("### Attrition Probability")

            st.markdown(
                f"""
                <h1 style='font-size:48px;color:#0F8B8D;margin:5px 0;'>
                    {probability * 100:.1f}%
                </h1>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(probability))

            st.caption(
                f"Classification threshold: {threshold:.0%}"
            )

    with col2:

        with st.container(border=True):

            st.markdown("### Risk Classification")

            if prediction == 1:
                st.error("HIGHER ATTRITION RISK")
                st.markdown(
                    "The predicted probability meets or exceeds "
                    "the model's decision threshold."
                )
            else:
                st.success("LOWER ATTRITION RISK")
                st.markdown(
                    "The predicted probability is below "
                    "the model's decision threshold."
                )

    st.markdown("### Employee Profile")

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    profile_col1.metric("Age", employee["age"])
    profile_col2.metric("Job Level", employee["job_level"])
    profile_col3.metric(
        "Years at Company",
        employee["years_at_company"]
    )

    with st.expander("View complete employee details"):

        summary = {
            "Department": employee[
                next(
                    key for key in employee
                    if key.startswith("department_")
                    and employee[key] == 1
                )
            ].__str__() if any(
                key.startswith("department_") and employee[key] == 1
                for key in employee
            ) else "Unknown",

            "Work Mode": next(
                (
                    key.replace("work_mode_", "")
                    for key in employee
                    if key.startswith("work_mode_") and employee[key] == 1
                ),
                "Unknown"
            ),

            "Monthly Income": employee["monthly_income"],
            "Overtime": "Yes" if employee["overtime"] else "No",
            "Job Satisfaction": employee["job_satisfaction"],
            "Performance Rating": employee["performance_rating"],
            "Distance From Home": employee["distance_from_home_km"],
            "Years Since Promotion": employee["years_since_promotion"],
            "Hire Year": employee["hire_year"]
        }

        st.dataframe(
            pd.DataFrame(
                list(summary.items()),
                columns=["Attribute", "Value"]
            ),
            use_container_width=True,
            hide_index=True
        )
    
    st.info(
        "This prediction is a statistical estimate, not a certainty "
        "that an employee will leave. HR decisions should consider "
        "additional context and direct employee feedback."
    )
    with st.expander("Explore individual prediction explanations"):
        display_shap_explanation(employee)



# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    page_header(
        "Employee Intelligence Overview",
        "A centralized view of employee attrition prediction and model performance."
    )

    st.markdown("### Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Accuracy",
            "89.75%",
            "Final model accuracy at the selected threshold."
        )

    with c2:
        metric_card(
            "ROC-AUC",
            "0.8898",
            "Final model's ROC-AUC score."
        )

    with c3:
        metric_card(
            "Attrition Recall",
            "57.8%",
            "Proportion of actual attrition cases identified."
        )

    with c4:
        metric_card(
            "Decision Threshold",
            "40%",
            "Selected using validation-set F1-score."
        )

    st.write("")

    left, right = st.columns([1.4, 1])

    with left:

        with st.container(border=True):

            st.markdown("### About AttriSense")

            st.write(
                "AttriSense is an AI-powered employee attrition "
                "prediction system designed to help HR teams identify "
                "potential employee retention risks."
            )

            st.write(
                "The system uses employee information and a trained "
                "Logistic Regression model to estimate the probability "
                "of attrition."
            )

            st.markdown("**Key capabilities**")

            st.markdown("""
            - Employee attrition probability prediction
            - Individual employee risk assessment
            - Model performance evaluation
            - Feature-based model explainability
            - Data-driven employee retention support
            """)

    with right:

        with st.container(border=True):

            st.markdown("### Model Configuration")

            st.markdown("""
            **Algorithm:** Logistic Regression

            **Preprocessing:** Median imputation and StandardScaler

            **Training dataset:** 100,000 records

            **Input features:** 23

            **Classification threshold:** 0.40
            """)

            st.success("Prediction model ready")


# =====================================================
# PAGE 2: ATTRITION PREDICTION
# =====================================================

elif page == "Attrition Prediction":

    page_header(
        "Employee Attrition Assessment",
        "Enter employee details to estimate their likelihood of attrition."
    )

    result = employee_form()

    if result is not None:
        st.session_state["last_prediction"] = result

    if "last_prediction" in st.session_state:
        display_prediction(st.session_state["last_prediction"])


# =====================================================
# PAGE 3: MODEL INSIGHTS
# =====================================================

elif page == "Model Insights":

    page_header(
        "Model Performance & Insights",
        "Understand how well the trained model identifies employee attrition."
    )

    st.markdown("### Final Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "89.75%")
    c2.metric("Precision", "75.9%")
    c3.metric("Recall", "57.8%")
    c4.metric("F1-score", "0.659")

    st.caption(
        "Metrics refer to the final Logistic Regression model "
        "evaluated at the 0.40 threshold."
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### Classification Metrics")

        metric_data = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-score"],
            "Score": [0.759, 0.578, 0.659]
        })

        st.bar_chart(
            metric_data.set_index("Metric"),
            y="Score",
            color="#0F8B8D"
        )

    with right:

        st.markdown("### Confusion Matrix")

        confusion_matrix_data = [
            [15992, 622],
            [1428, 1958]
        ]

        fig, ax = plt.subplots(figsize=(5, 4))

        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        sns.heatmap(
            confusion_matrix_data,
            annot=True,
            fmt=",d",
            cmap="GnBu",
            linewidths=1,
            linecolor="white",
            cbar=False,
            xticklabels=["Predicted: Stay", "Predicted: Leave"],
            yticklabels=["Actual: Stay", "Actual: Leave"],
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.markdown("### Understanding the Results")

    with st.container(border=True):

        st.markdown("**Why is recall important?**")

        st.write(
            "Recall measures how many employees who actually left "
            "were correctly identified by the model. In attrition "
            "prediction, missing employees who may leave can mean "
            "missing an opportunity to investigate retention concerns."
        )

        st.markdown("**Why was the threshold set to 0.40?**")

        st.write(
            "The 0.40 threshold was selected because it produced "
            "the highest F1-score on the validation set among the "
            "thresholds evaluated. It balances precision and recall "
            "for the final model."
        )

    st.markdown("### Threshold Comparison")

    threshold_data = pd.DataFrame({
        "Threshold": ["0.25", "0.30", "0.35", "0.40", "0.45", "0.50"],
        "Precision": [0.571, 0.628, 0.699, 0.763, 0.826, 0.880],
        "Recall": [0.711, 0.657, 0.607, 0.573, 0.537, 0.503],
        "F1-score": [0.633, 0.642, 0.650, 0.655, 0.651, 0.640]
    })

    st.dataframe(
        threshold_data,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# PAGE 4: ABOUT PROJECT
# =====================================================

elif page == "About Project":

    page_header(
        "About AttriSense",
        "Project overview, technology stack and machine learning workflow."
    )

    with st.container(border=True):

        st.markdown("## 📊 AttriSense")

        st.write(
            "AI-Powered Employee Attrition Prediction & "
            "Retention Intelligence"
        )

        st.write(
            "AttriSense explores how machine learning can help "
            "organizations understand employee attrition patterns "
            "and identify potential retention risks."
        )

    st.markdown("### Technology Stack")

    c1, c2, c3 = st.columns(3)

    c1.markdown("**Python**")
    c1.caption("Programming language")

    c2.markdown("**Pandas & NumPy**")
    c2.caption("Data processing")

    c3.markdown("**Scikit-learn**")
    c3.caption("Machine learning")

    c1, c2, c3 = st.columns(3)

    c1.markdown("**Matplotlib & Seaborn**")
    c1.caption("Data visualization")

    c2.markdown("**SHAP**")
    c2.caption("Model explainability")

    c3.markdown("**Streamlit**")
    c3.caption("Dashboard development")

    st.divider()

    st.markdown("### Machine Learning Workflow")

    st.markdown("""
    1. Dataset collection and understanding
    2. Data cleaning and preprocessing
    3. Exploratory data analysis
    4. Feature engineering and selection
    5. Train-test split
    6. Model training and comparison
    7. Threshold optimization
    8. Model evaluation
    9. Employee attrition prediction
    10. Model explainability
    """)

    st.divider()

    st.caption(
        "Academic Project | Course Code: 24CSE0316 | AI & ML"
    )
