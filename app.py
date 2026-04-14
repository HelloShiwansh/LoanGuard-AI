import os

import joblib
import pandas as pd
import streamlit as st


def load_artifacts():
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, "artifacts", "loan_approval_model.joblib")
    meta_path = os.path.join(root, "artifacts", "model_meta.joblib")
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return model, meta


def _repo_root():
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner=False)
def load_dataset_choices(input_columns: list[str]) -> dict[str, list]:
    """
    Pull selectbox options from your dataset so the UI only shows values
    that actually exist in the data.
    """
    data_path = os.path.join(_repo_root(), "LoanGuard_data.csv")
    if not os.path.exists(data_path):
        return {}

    df = pd.read_csv(data_path)
    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])
    if "Loan_Approved" in df.columns:
        df = df.drop(columns=["Loan_Approved"])

    choices: dict[str, list] = {}
    for col in input_columns:
        if col not in df.columns:
            continue
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            opts = [x for x in df[col].dropna().astype(str).unique().tolist() if x.strip() != ""]
            opts = sorted(set(opts))
            if opts:
                choices[col] = opts
    return choices


def section_card(title: str):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        return st.container()

import streamlit as st

st.set_page_config(page_title="LoanGuard AI", page_icon="🏦")

st.markdown(
    """
    <h1 style='text-align: center;'>🏦 LoanGuard AI 🏦</h1>
    <h3 style='text-align: center;'>Loan Approval Predictor</h3>
    <p style='text-align: center;'>Enter applicant details to predict loan approval.</p>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
<style>
/* Make card borders a bit softer and spacing consistent */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding-top: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

model, meta = load_artifacts()
choices = load_dataset_choices(meta["input_columns"])

st.subheader("Applicant details")

with st.form("loan_form", border=False):
    # Row 1
    with st.container(border=True):
        st.markdown("#### Income & Savings")
        c1, c2 = st.columns(2)
        with c1:
            Applicant_Income = st.number_input("Applicant_Income", min_value=0.0, value=5000.0, step=100.0)
            Savings = st.number_input("Savings", min_value=0.0, value=5000.0, step=100.0)
        with c2:
            Coapplicant_Income = st.number_input("Coapplicant_Income", min_value=0.0, value=0.0, step=100.0)
            Collateral_Value = st.number_input("Collateral_Value", min_value=0.0, value=10000.0, step=100.0)

    # Row 2
    with st.container(border=True):
        st.markdown("#### Loan Details")
        c1, c2 = st.columns(2)
        with c1:
            Loan_Amount = st.number_input("Loan_Amount", min_value=0.0, value=15000.0, step=100.0)
            Loan_Term = st.number_input("Loan_Term", min_value=0.0, value=60.0, step=1.0)
        with c2:
            Loan_Purpose = st.selectbox(
                "Loan_Purpose",
                options=choices.get("Loan_Purpose", []),
                index=0 if choices.get("Loan_Purpose") else None,
                placeholder="Select",
            )
            Property_Area = st.selectbox(
                "Property_Area",
                options=choices.get("Property_Area", []),
                index=0 if choices.get("Property_Area") else None,
                placeholder="Select",
            )

    # Row 3
    with st.container(border=True):
        st.markdown("#### Credit & Risk")
        c1, c2 = st.columns(2)
        with c1:
            Credit_Score = st.number_input("Credit_Score", min_value=300.0, max_value=900.0, value=650.0, step=1.0)
            Existing_Loans = st.number_input("Existing_Loans", min_value=0.0, value=0.0, step=1.0)
        with c2:
            DTI_Ratio = st.number_input("DTI_Ratio", min_value=0.0, max_value=2.0, value=0.30, step=0.01)

    # Row 4
    with st.container(border=True):
        st.markdown("#### Personal & Employment")
        c1, c2 = st.columns(2)
        with c1:
            Age = st.number_input("Age", min_value=18.0, max_value=100.0, value=30.0, step=1.0)
            Dependents = st.number_input("Dependents", min_value=0.0, value=0.0, step=1.0)
            Marital_Status = st.selectbox(
                "Marital_Status",
                options=choices.get("Marital_Status", []),
                index=0 if choices.get("Marital_Status") else None,
                placeholder="Select",
            )
            Education_Level = st.selectbox(
                "Education_Level",
                options=choices.get("Education_Level", []),
                index=0 if choices.get("Education_Level") else None,
                placeholder="Select",
            )
        with c2:
            Gender = st.selectbox(
                "Gender",
                options=choices.get("Gender", []),
                index=0 if choices.get("Gender") else None,
                placeholder="Select",
            )
            Employment_Status = st.selectbox(
                "Employment_Status",
                options=choices.get("Employment_Status", []),
                index=0 if choices.get("Employment_Status") else None,
                placeholder="Select",
            )
            Employer_Category = st.selectbox(
                "Employer_Category",
                options=choices.get("Employer_Category", []),
                index=0 if choices.get("Employer_Category") else None,
                placeholder="Select",
            )

    st.divider()
    predict = st.form_submit_button("Predict", type="primary", use_container_width=True)

input_dict = {
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Employment_Status": Employment_Status,
    "Age": Age,
    "Marital_Status": Marital_Status,
    "Dependents": Dependents,
    "Credit_Score": Credit_Score,
    "Existing_Loans": Existing_Loans,
    "DTI_Ratio": DTI_Ratio,
    "Savings": Savings,
    "Collateral_Value": Collateral_Value,
    "Loan_Amount": Loan_Amount,
    "Loan_Term": Loan_Term,
    "Loan_Purpose": Loan_Purpose,
    "Property_Area": Property_Area,
    "Education_Level": Education_Level,
    "Gender": Gender,
    "Employer_Category": Employer_Category,
}

# Ensure column order matches training expectation
X = pd.DataFrame([input_dict], columns=meta["input_columns"])

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_submitted_X" not in st.session_state:
    st.session_state.last_submitted_X = None

if predict:
    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    st.session_state.last_prediction = {"proba": proba, "pred": pred}
    st.session_state.last_submitted_X = X

if st.session_state.last_prediction is not None:
    proba = float(st.session_state.last_prediction["proba"])
    pred = int(st.session_state.last_prediction["pred"])
    approved = pred == 1

    if approved:
        st.success(f"Approved (probability: {proba:.3f})")
    else:
        st.error(f"Not Approved (probability: {proba:.3f})")

    with st.expander("View submitted data", expanded=True):
        st.dataframe(st.session_state.last_submitted_X, use_container_width=True)

