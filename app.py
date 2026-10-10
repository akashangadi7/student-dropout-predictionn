"""
Streamlit demo: Student Dropout & Academic Success predictor
Run:  streamlit run app.py
Needs: models/best_model.joblib (made by train_models.py) and data/dataset.csv
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")


# ------------------------------------------------------------ load model + data
@st.cache_resource
def load_model():
    bundle = joblib.load("models/best_model.joblib")
    return bundle["model"], bundle["classes"], bundle["columns"]


@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")


model, classes, columns = load_model()
data = load_data()

# Default value for every feature we do NOT ask the user about:
# most common value for whole-number columns, median for decimal columns.
defaults = {}
for col in columns:
    if data[col].dtype == "float64":
        defaults[col] = float(data[col].median())
    else:
        defaults[col] = int(data[col].mode().iloc[0])


def explain(row_df, class_idx):
    """SHAP values of one student for one class (tree models only)."""
    try:
        import shap
        sv = np.array(shap.TreeExplainer(model).shap_values(row_df))
        # newer shap: (1, features, classes)   older shap: (classes, 1, features)
        vals = sv[0, :, class_idx] if sv.shape[-1] == len(classes) else sv[class_idx, 0, :]
        return pd.Series(vals, index=columns)
    except Exception:
        return None


def show_result(row_df):
    proba = model.predict_proba(row_df)[0]
    pred_idx = int(np.argmax(proba))
    pred = classes[pred_idx]

    colour = {"Dropout": "#C44E52", "Enrolled": "#DD8452", "Graduate": "#55A868"}[pred]
    st.markdown(
        f"<div style='padding:16px;border-radius:10px;background:{colour}22;"
        f"border:2px solid {colour}'><h2 style='margin:0;color:{colour}'>"
        f"Prediction: {pred} ({proba[pred_idx]*100:.0f}%)</h2></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("**Probability for each outcome**")
    st.bar_chart(pd.Series(proba, index=classes))

    st.write(f"**Why? Top factors for '{pred}'** (red = pushes towards {pred}, blue = pushes away)")
    s = explain(row_df, pred_idx)
    if s is None:
        st.info("Explanations are available for tree models (Random Forest / XGBoost).")
        return
    top = s.reindex(s.abs().sort_values(ascending=False).index[:8])[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(top.index, top.values, color=["#C44E52" if v > 0 else "#4C72B0" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    plt.tight_layout()
    st.pyplot(fig)


# ------------------------------------------------------------------- page
st.title("🎓 Student Dropout & Academic Success Predictor")
st.caption("Predicts whether a student will Drop out, stay Enrolled, or Graduate, "
           "and explains the decision with SHAP.")

tab1, tab2 = st.tabs(["✍️ Enter a student", "📂 Test on a real student from the dataset"])

# ----- Tab 1: manual form
with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Student details")
        a, b = st.columns(2)
        age = a.slider("Age at enrollment", 17, 70, 20)
        gender = b.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        a, b, c = st.columns(3)
        tuition = a.selectbox("Tuition fees up to date?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        debtor = b.selectbox("Debtor?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        scholar = c.selectbox("Scholarship?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        displaced = st.selectbox("Displaced (lives away from home)?", [0, 1],
                                 format_func=lambda x: "Yes" if x else "No")

        st.markdown("**1st semester**")
        a, b, c = st.columns(3)
        s1_enr = a.number_input("Units enrolled", 0, 30, 6, key="s1e")
        s1_app = b.number_input("Units approved", 0, 30, 5, key="s1a")
        s1_gr = c.number_input("Average grade (0-20)", 0.0, 20.0, 12.0, key="s1g")

        st.markdown("**2nd semester**")
        a, b, c = st.columns(3)
        s2_enr = a.number_input("Units enrolled", 0, 30, 6, key="s2e")
        s2_app = b.number_input("Units approved", 0, 30, 5, key="s2a")
        s2_gr = c.number_input("Average grade (0-20)", 0.0, 20.0, 12.0, key="s2g")

    row = defaults.copy()
    row.update({
        "Age at enrollment": age, "Gender": gender,
        "Tuition fees up to date": tuition, "Debtor": debtor,
        "Scholarship holder": scholar, "Displaced": displaced,
        "Curricular units 1st sem (enrolled)": s1_enr,
        "Curricular units 1st sem (approved)": s1_app,
        "Curricular units 1st sem (grade)": s1_gr,
        "Curricular units 1st sem (evaluations)": s1_enr,
        "Curricular units 2nd sem (enrolled)": s2_enr,
        "Curricular units 2nd sem (approved)": s2_app,
        "Curricular units 2nd sem (grade)": s2_gr,
        "Curricular units 2nd sem (evaluations)": s2_enr,
    })
    with right:
        st.subheader("Result")
        show_result(pd.DataFrame([row])[columns])
    st.caption("Fields not shown here use the most common value in the dataset.")

# ----- Tab 2: real student from the dataset
with tab2:
    st.write("Pick a real student and compare the model's prediction with what actually happened.")
    idx = st.number_input("Student row number", 0, len(data) - 1, 0)
    real = data.iloc[int(idx)]
    st.write(f"**Actual outcome:** {real['Target']}")
    show_result(pd.DataFrame([real.drop("Target")])[columns])

st.divider()
st.caption("⚠️ Educational prototype trained on one Portuguese university's data. "
           "Use it to start a supportive conversation with a student, never to make decisions about them.")
