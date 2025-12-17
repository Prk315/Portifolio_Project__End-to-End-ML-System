import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.explainability import ModelExplainer

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model_and_preprocessor():
    """
    Loads model and preprocessor (cached).
    """
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'final_model.joblib'
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.joblib'

        if not model_path.exists():
            st.error("Model not found. Please train the model first.")
            return None, None, None

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        explainer = ModelExplainer(str(model_path), str(preprocessor_path))

        return model, preprocessor, explainer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def main():
    st.title("💰 Loan Default Prediction System")
    st.markdown("Predict loan default risk with explainable outcomes")

    model, preprocessor, explainer = load_model_and_preprocessor()

    if model is None:
        st.stop()

    st.sidebar.header("Loan Application Details")

    with st.sidebar:
        st.subheader("Loan Information")
        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=500
        )

        term = st.selectbox(
            "Loan Term",
            ["36 months", "60 months"]
        )

        int_rate = st.slider(
            "Interest Rate (%)",
            min_value=5.0,
            max_value=30.0,
            value=10.5,
            step=0.5
        )

        installment = st.number_input(
            "Monthly Installment ($)",
            min_value=50.0,
            max_value=2000.0,
            value=325.0,
            step=10.0
        )

        st.subheader("Borrower Information")

        annual_inc = st.number_input(
            "Annual Income ($)",
            min_value=20000,
            max_value=500000,
            value=60000,
            step=5000
        )

        dti = st.slider(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=50.0,
            value=15.5,
            step=0.5
        )

        emp_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years",
             "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
        )

        grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

        st.subheader("Credit Information")

        fico_range_high = st.slider(
            "FICO Score (High Range)",
            min_value=600,
            max_value=850,
            value=720,
            step=10
        )

        revol_bal = st.number_input(
            "Revolving Balance ($)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=500
        )

        revol_util = st.slider(
            "Revolving Utilization (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=5.0
        )

        delinq_2yrs = st.number_input(
            "Delinquencies (past 2 years)",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        predict_button = st.button("Predict Default Risk", type="primary")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Prediction Results")

        if predict_button:
            loan_data = {
                'loan_amnt': loan_amnt,
                'term': term,
                'int_rate': int_rate,
                'installment': installment,
                'grade': grade,
                'emp_length': emp_length,
                'annual_inc': annual_inc,
                'dti': dti,
                'delinq_2yrs': delinq_2yrs,
                'fico_range_high': fico_range_high,
                'revol_bal': revol_bal,
                'revol_util': revol_util
            }

            try:
                df = pd.DataFrame([loan_data])
                X = preprocessor.transform(df)

                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]

                prob_no_default = probabilities[0]
                prob_default = probabilities[1]

                if prob_default < 0.3:
                    risk_level = "Low"
                    risk_color = "green"
                elif prob_default < 0.6:
                    risk_level = "Medium"
                    risk_color = "orange"
                else:
                    risk_level = "High"
                    risk_color = "red"

                st.subheader("Risk Assessment")

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric(
                        "Prediction",
                        "Default" if prediction == 1 else "No Default",
                        delta=None
                    )

                with col_b:
                    st.metric(
                        "Default Probability",
                        f"{prob_default:.1%}",
                        delta=None
                    )

                with col_c:
                    st.metric(
                        "Risk Level",
                        risk_level,
                        delta=None
                    )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_default * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "lightyellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Explanation")

                with st.spinner("Generating explanation..."):
                    explanation, feature_contrib = explainer.explain_prediction(loan_data)

                st.info(explanation)

                st.subheader("Feature Contributions")

                top_features = feature_contrib.head(10)

                fig_bar = px.bar(
                    top_features,
                    x='shap',
                    y='feature',
                    orientation='h',
                    title='Top 10 Features Affecting Prediction',
                    labels={'shap': 'Impact on Default Risk', 'feature': 'Feature'},
                    color='shap',
                    color_continuous_scale=['green', 'yellow', 'red']
                )

                fig_bar.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

                with st.expander("View Detailed Feature Contributions"):
                    st.dataframe(
                        feature_contrib.style.background_gradient(
                            subset=['shap'],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.exception(e)

    with col2:
        st.header("Model Information")

        st.markdown("""
        ### About This Model

        This system predicts loan default risk using:

        - **Model**: XGBoost Classifier
        - **Features**: Loan amount, interest rate, borrower income, credit history, etc.
        - **Explainability**: SHAP values for transparency

        ### Risk Levels

        - 🟢 **Low Risk**: < 30% default probability
        - 🟡 **Medium Risk**: 30-60% default probability
        - 🔴 **High Risk**: > 60% default probability

        ### How to Use

        1. Enter loan details in the sidebar
        2. Click "Predict Default Risk"
        3. Review the risk assessment
        4. Examine feature contributions

        ### Disclaimer

        This is a demonstration system. Always consult with financial professionals for actual lending decisions.
        """)

if __name__ == "__main__":
    main()
