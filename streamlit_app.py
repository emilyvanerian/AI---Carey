import streamlit as st
from drug_analysis import DrugConsumptionAnalysis
import pandas as pd

# Page config
st.set_page_config(
    page_title="Drug Consumption Analysis Dashboard",
    page_icon="💊",
    layout="wide"
)

# Title
st.title("Drug Consumption Analysis Dashboard")

# Initialize analysis
@st.cache_resource
def get_analysis():
    return DrugConsumptionAnalysis()

analysis = get_analysis()

# Sidebar
st.sidebar.header("Dashboard Controls")
target_drug = st.sidebar.selectbox(
    "Select Target Drug",
    options=analysis.drug_names,
    index=analysis.drug_names.index("Cannabis")
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Dataset Overview")
    st.write(f"Total Samples: {len(analysis.X):,}")
    st.write(f"Features: {analysis.X.shape[1]}")
    st.write(f"Drug Types: {len(analysis.drug_names)}")

# Process data and train model
X_scaled, y_encoded = analysis.preprocess_data(target_drug=target_drug)
model, X_train, X_test, y_train, y_test = analysis.train_model(X_scaled, y_encoded)
evaluation_results = analysis.evaluate_model(model, X_test, y_test)

# Model Performance Metrics
with col2:
    st.header("Model Performance")
    metrics = evaluation_results['classification_report']
    st.write(f"Accuracy: {metrics['accuracy']:.2%}")
    st.write(f"Precision: {metrics['1']['precision']:.2%}")
    st.write(f"Recall: {metrics['1']['recall']:.2%}")
    st.write(f"F1-Score: {metrics['1']['f1-score']:.2%}")

# Visualizations
st.header("Data Visualizations")

# Drug Usage Distribution
st.subheader("Drug Usage Distribution")
drug_usage_fig = analysis.create_drug_usage_distribution()
st.plotly_chart(drug_usage_fig, use_container_width=True)

# Personality Traits Distribution
st.subheader("Personality Traits Distribution")
personality_fig = analysis.create_personality_distribution()
st.plotly_chart(personality_fig, use_container_width=True)

# Feature Importance
st.subheader("Feature Importance")
importance_fig = analysis.create_feature_importance_plot(model)
st.plotly_chart(importance_fig, use_container_width=True)

# Correlation Heatmap
st.subheader("Feature Correlation Matrix")
corr_fig = analysis.create_correlation_heatmap()
st.plotly_chart(corr_fig, use_container_width=True)

# Confusion Matrix
st.subheader("Confusion Matrix")
conf_matrix = pd.DataFrame(
    evaluation_results['confusion_matrix'],
    index=['Actual Negative', 'Actual Positive'],
    columns=['Predicted Negative', 'Predicted Positive']
)
st.dataframe(conf_matrix)

# Footer
st.markdown("---")
st.markdown("""
### About This Dashboard
This dashboard provides comprehensive analysis of the Drug Consumption Dataset from the UCI Machine Learning Repository.
It includes exploratory data analysis, predictive modeling, and real-time predictions for different drugs.
""")
