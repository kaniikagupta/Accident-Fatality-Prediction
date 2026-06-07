"""
Road Accident Severity Prediction - Streamlit App
Main application file with multi-page navigation
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AccidentPredict -  Road Safety Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low { background-color: #10b981; color: white; padding: 0.5rem; border-radius: 5px; }
    .risk-moderate { background-color: #f59e0b; color: white; padding: 0.5rem; border-radius: 5px; }
    .risk-high { background-color: #ef4444; color: white; padding: 0.5rem; border-radius: 5px; }
    .risk-critical { background-color: #991b1b; color: white; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# Load or train model
@st.cache_resource
def load_model():
    """Load trained model or return None"""
    with open(r"C:\Users\91741\OneDrive\Desktop\Accident Fatality Prediction\model\lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

numeric_features = [
    'Count_Vehicles_Total', 'Count_Occupants_MV', 'Count_Non_Motorists',
    'Crash_Hour', 'Crash_Minute',
    'Max_Driver_Age', 'Min_Driver_Age',
    'Total_Exceeding_Limit_Vehicles', 'Total_Too_Fast_Vehicles',
    'Driver_Max_Prev_Accident_Count', 'Driver_Max_Prev_Speeding_Count'
]
categorical_features = [
    'Clean_Light_Condition', 'Clean_Weather_Condition',
    'Area_Rural_Urban', 'Road_Functional_Class',
    'Intersection_Type', 'Driver_License_Category',
    'Light_Condition', 'Weather_Condition',
    'Crash_Road_Surface_Condition', 'Crash_Horizontal_Alignment',
    'Crash_Vertical_Profile', 'Crash_Impairment_Category',
    'Crash_Vehicle_Defect', 'On_NHS_Road',
    'Any_Driver_Drinking', 'Driver_Sex_List'
]

@st.cache_data
def load_dataset():
    """Load the accident dataset"""
    try:
        df = pd.read_csv(r'data/clean data/final_dataset.csv')
        return df
    except:
        return None


# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/car-accident.png", width=150)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Dataset Overview", "Predict", "Results", "Insights"],
        icons=["house", "gear", "search", "graph-up", "lightbulb", "speedometer2"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "color":"#4665f1",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#667eea","color":"#fafafa"},
            "icon": {"color": "#aebaeb"}
        }
    )

# Load model and dataset
model = load_model()
dataset = load_dataset()

# HOME PAGE
if selected == "Home":
    st.markdown('<h1 class="main-header">🚗 AccidentPredict </h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced LightGBM model predicting accident fatality with 73%+ accuracy</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h2>37,000+</h2><p>Accident Records</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h2>73.6%</h2><p>Prediction Accuracy</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h2>88.9%</h2><p>ROC AUC Score</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader(" Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Comprehensive Insights
        - Weather impact analysis
        - Time-based risk patterns
        - Location-based statistics
        """)
    
    with col2:
        st.markdown("""
        #### Model Evaluation
        - Precision (Low Fatality): 0.99
        - Precision (High Fatality): 0.20
        - Recall (Low Fatality): 0.72
        - Recall (High Fatality): 0.90          
        """)
    
    st.markdown("---")
    st.info(" Use the sidebar to navigate between different sections")

# Dataset Overview PAGE
elif selected == "Dataset Overview":
    st.markdown('<h1 class="main-header">🎓 Dataset Overview</h1>', unsafe_allow_html=True)
    
    if dataset is not None:
        st.success(f"✅ Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
        
        # Show dataset preview
        with st.expander(" View Dataset Preview"):
            st.dataframe(dataset.head(100), use_container_width=True)
            st.write(f"**Shape:** {dataset.shape}")
            st.write(f"**Columns:** {', '.join(dataset.columns.tolist())}")
        
        
        st.markdown("---")
        
    else:
        st.error("❌ Dataset not found! Please ensure 'public/data/final_dataset_1.csv' exists.")
        st.info("Upload your dataset file to the correct location and refresh the page.")

# PREDICT PAGE
elif selected == "Predict":
    st.markdown('<h1 class="main-header">🔮 Predict Accident Fatality</h1>', unsafe_allow_html=True)
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train the model first!")
        if st.button("Go to Train Model"):
            st.rerun()
    else:
        st.success("✅ Model loaded and ready for predictions")
        
        st.subheader("Enter Accident Details")
        with open(r"C:\Users\91741\OneDrive\Desktop\Accident Fatality Prediction\model\train_features.pkl", "rb") as f:
            train_columns = pickle.load(f)
        
        with open(r"C:\Users\91741\OneDrive\Desktop\Accident Fatality Prediction\model\categorical_unique.pkl", "rb") as f:
            cat_unique = pickle.load(f)

        # Collect input
        input_data = {}
        # Default values for specific numeric inputs
        default_numeric_values = {
            'Count_Vehicles_Total': 1,
            'Count_Occupants_MV': 1,
            'Count_Non_Motorists': 0,
            'Crash_Hour': 12,
            'Crash_Minute': 0,
            'Max_Driver_Age': 15,    # your request
            'Min_Driver_Age': 15,    # your request
            'Total_Exceeding_Limit_Vehicles': 0,
            'Total_Too_Fast_Vehicles': 0,
            'Driver_Max_Prev_Accident_Count': 0,
            'Driver_Max_Prev_Speeding_Count': 0,
            'Restraint_Used': 1
        }

        input_data = {}
        for col in numeric_features + ['Restraint_Used']:
            default_val = default_numeric_values.get(col, 1)  # fallback 1 if not specified
            input_data[col] = st.number_input(col, min_value=0, value=default_val)

        for col in categorical_features:
            unique_vals = cat_unique.get(col, ["Unknown"])
            input_data[col] = st.selectbox(col, unique_vals)

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Align with training columns
        from pandas.api.types import CategoricalDtype
        for col in categorical_features:
            dtype = CategoricalDtype(categories=cat_unique[col])
            input_df[col] = input_df[col].astype(dtype)


        if st.button("Predict Fatality Level"):
            # Predict using trained model
            pred = model.predict(input_df)[0]

            risk_level = "Low Fatality" if pred == 0 else "High Fatality"
            risk_color = "low" if pred == 0 else "high"

            
            # Save result for Results page
            st.session_state.prediction_result = {
                'prediction': int(pred),
                'risk_level': risk_level,
                'risk_color': risk_color,
            }

            st.success(f"✅ Prediction complete! Check the Results page.")

# RESULTS PAGE
elif selected == "Results":
    st.markdown('<h1 class="main-header"> Prediction Results</h1>', unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.info("No predictions yet. Please make a prediction first!")
        if st.button("Go to Predict"):
            st.rerun()
    else:
        result = st.session_state.prediction_result
        
        # Display risk level with color
        risk_class = f"risk-{result['risk_color']}"
        st.markdown(
            f'<h2 class="{risk_class}" style="text-align: center; padding: 2rem; border-radius: 10px; margin: 2rem 0;">'
            f'{result["risk_level"]}</h2>', unsafe_allow_html=True
        )
        
        
        # Key risk factors
        st.subheader("⚠️ Key Risk Factors")
        if model:
            feature_names = model.booster_.feature_name()  
            feature_imp = pd.DataFrame({
                 'feature': feature_names,
                 'importance': model.booster_.feature_importance(importance_type='gain')
            }).sort_values(by='importance', ascending=False)
    
            top_features = feature_imp.head(5)
            for _, row in top_features.iterrows():
                st.write(f"• **{row['feature']}**: contributes most to fatality risk")

# INSIGHTS PAGE
elif selected == "Insights":
    st.markdown('<h1 class="main-header">💡 Data Insights</h1>', unsafe_allow_html=True)
    
    if dataset is not None:
        st.subheader(" Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(dataset):,}")
        with col2:
            st.metric("Total Columns", dataset.shape[1])
        with col3:
            st.metric("Missing Values", dataset.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", dataset.duplicated().sum())
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs([" Time Analysis", " Weather Analysis", " Location Analysis"])
        
        with tab1:
            if 'Crash_Hour' in dataset.columns:
                st.subheader("Accidents by Hour of Day")
                hourly = dataset['Crash_Hour'].value_counts().sort_index()
                fig = px.line(x=hourly.index, y=hourly.values, 
                            labels={'x': 'Hour', 'y': 'Number of Accidents'},
                            title="Accident Distribution Throughout the Day")
                fig.update_xaxes(dtick=1, range=[0, 24])
                fig.update_traces(mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'Clean_Weather_Condition' in dataset.columns:
                st.subheader("Accidents by Weather Condition")
                weather = dataset['Clean_Weather_Condition'].value_counts()
                fig = px.pie(values=weather.values, names=weather.index,
                           title="Weather Conditions During Accidents")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if 'Latitude' in dataset.columns and 'Longitude' in dataset.columns:
                st.subheader("Accident Locations (Sample)")
                sample = dataset.sample(min(1000, len(dataset)))
                fig = px.scatter_mapbox(sample, lat='Latitude', lon='Longitude',
                                       zoom=10, height=500,
                                       title="Geographic Distribution of Accidents")
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Dataset not found!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Advanced Road Safety Analysis Platform</p>
    <p>Powered by LightGBM & Streamlit</p>
</div>
""", unsafe_allow_html=True)


