import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io

# Set a style for matplotlib plots
plt.style.use('seaborn-v0_8-darkgrid')

# Function to load data
@st.cache_data
def load_data():
    """Loads the California Housing dataset and splits it into features (X) and target (y)."""
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y, data.DESCR

# Function to train and evaluate models
@st.cache_resource
def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, params=None):
    """
    Trains a specified model and evaluates its performance.

    Args:
        model_name (str): The name of the model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        params (dict, optional): Hyperparameters for GridSearchCV. Defaults to None.

    Returns:
        tuple: Trained model, predictions, RMSE, MAE, R2 score, and cross-validation scores.
    """
    model = None
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge Regression":
        model = Ridge()
    elif model_name == "Lasso Regression":
        model = Lasso()
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "XGBoost Regressor":
        model = XGBRegressor(random_state=42)

    if model:
        # Hyperparameter tuning with GridSearchCV if parameters are provided
        if params:
            st.info(f"Performing GridSearchCV for {model_name}. This may take a while...")
            grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.success(f"GridSearchCV completed. Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)

        return model, y_pred, rmse, mae, r2, cv_rmse
    return None, None, None, None, None, None

# Streamlit UI
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more polished look
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 5px #aaa;
    }
    .subheader {
        font-size: 1.8em;
        color: #333;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 1.2em;
        transition-duration: 0.4s;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
        box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 5px;
    }
    .stAlert {
        border-radius: 8px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6; /* Light grey background for sidebar */
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🏠 California Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load data once
X, y, dataset_description = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X) # For full dataset predictions if needed
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Data Visualization", "Model Training & Prediction", "Make Custom Prediction"])

if page == "Dataset Overview":
    st.markdown("<h2 class='subheader'>Dataset Overview</h2>", unsafe_allow_html=True)
    st.write("The California Housing dataset contains information from the 1990 California census. It's a regression problem where the goal is to predict the median house value for California districts.")
    st.markdown("### Dataset Description")
    st.text(dataset_description)

    st.markdown("### First 5 Rows of Features (X)")
    st.dataframe(X.head())

    st.markdown("### Descriptive Statistics of Features (X)")
    st.dataframe(X.describe())

    st.markdown("### Target Variable (y) - Median House Value")
    st.dataframe(y.head())
    st.write(f"Mean of target: ${y.mean():,.2f}")
    st.write(f"Standard Deviation of target: ${y.std():,.2f}")

elif page == "Data Visualization":
    st.markdown("<h2 class='subheader'>Data Visualization</h2>", unsafe_allow_html=True)

    st.markdown("### Feature Correlations Heatmap")
    st.write("This heatmap shows the correlation between different features in the dataset. A higher absolute value indicates a stronger correlation.")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Feature Correlations", fontsize=16)
    st.pyplot(fig)

    st.markdown("### Distribution of Median Income (MedInc)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(X['MedInc'], bins=50, kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribution of Median Income", fontsize=16)
    ax.set_xlabel("Median Income (in tens of thousands of dollars)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)

    st.markdown("### Distribution of Median House Value (Target)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y, bins=50, kde=True, ax=ax, color='lightcoral')
    ax.set_title("Distribution of Median House Value", fontsize=16)
    ax.set_xlabel("Median House Value (in hundreds of thousands of dollars)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)

elif page == "Model Training & Prediction":
    st.markdown("<h2 class='subheader'>Model Training & Evaluation</h2>", unsafe_allow_html=True)

    model_options = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "XGBoost Regressor"
    ]
    selected_model = st.selectbox("Select a Model for Training:", model_options)

    perform_grid_search = st.checkbox("Perform Hyperparameter Tuning (GridSearchCV)", value=False)
    
    model_params = {
        "Linear Regression": {}, # No hyperparameters to tune usually
        "Ridge Regression": {'alpha': [0.1, 1.0, 10.0]},
        "Lasso Regression": {'alpha': [0.1, 1.0, 10.0]},
        "Decision Tree Regressor": {'max_depth': [None, 10, 20, 30]},
        "Random Forest Regressor": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        "Gradient Boosting Regressor": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        "XGBoost Regressor": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }

    params_to_use = model_params.get(selected_model) if perform_grid_search else None

    if st.button("Train and Evaluate Model"):
        with st.spinner(f"Training {selected_model}..."):
            trained_model, y_pred, rmse, mae, r2, cv_rmse = train_and_evaluate_model(
                selected_model, X_train_scaled_df, y_train, X_test_scaled_df, y_test, params_to_use
            )

            if trained_model:
                st.session_state['trained_model'] = trained_model
                st.session_state['model_name'] = selected_model
                st.session_state['rmse'] = rmse
                st.session_state['mae'] = mae
                st.session_state['r2'] = r2
                st.session_state['cv_rmse'] = cv_rmse
                st.session_state['scaler'] = scaler
                st.session_state['X_columns'] = X.columns.tolist()

                st.success(f"Model '{selected_model}' trained successfully!")

                st.markdown("### Model Performance on Test Set")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{rmse:.3f}")
                with col2:
                    st.metric("MAE", f"{mae:.3f}")
                with col3:
                    st.metric("R-squared", f"{r2:.3f}")

                st.markdown("### Cross-Validation RMSE Scores")
                st.write(f"Mean CV RMSE: {np.mean(cv_rmse):.3f} (Std: {np.std(cv_rmse):.3f})")
                st.line_chart(pd.DataFrame(cv_rmse, columns=["CV RMSE"]))

                st.markdown("### Predictions vs. Actual Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred, alpha=0.3, color='darkgreen')
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Values", fontsize=12)
                ax.set_ylabel("Predicted Values", fontsize=12)
                ax.set_title("Actual vs. Predicted House Values", fontsize=16)
                st.pyplot(fig)

                # Save model option
                st.markdown("### Save Trained Model")
                if st.button("Save Current Model"):
                    try:
                        model_buffer = io.BytesIO()
                        joblib.dump(trained_model, model_buffer)
                        model_buffer.seek(0)
                        st.download_button(
                            label=f"Download {selected_model}.pkl",
                            data=model_buffer,
                            file_name=f"{selected_model.replace(' ', '_').lower()}.pkl",
                            mime="application/octet-stream"
                        )
                        st.success("Model saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving model: {e}")
            else:
                st.error("Model training failed. Please select a valid model.")

    st.markdown("---")
    st.markdown("### Load Previously Saved Model")
    uploaded_file = st.file_uploader("Upload a .pkl model file", type=["pkl"])
    if uploaded_file is not None:
        try:
            loaded_model = joblib.load(uploaded_file)
            st.session_state['trained_model'] = loaded_model
            st.session_state['model_name'] = "Uploaded Model" # You might want to infer the model type
            st.session_state['scaler'] = scaler # Assume the same scaler is used
            st.session_state['X_columns'] = X.columns.tolist()
            st.success("Model loaded successfully!")
            st.write(f"Loaded Model Type: {type(loaded_model).__name__}")
        except Exception as e:
            st.error(f"Error loading model: {e}. Please ensure it's a valid scikit-learn model .pkl file.")

elif page == "Make Custom Prediction":
    st.markdown("<h2 class='subheader'>Make a Custom House Price Prediction</h2>", unsafe_allow_html=True)

    if 'trained_model' not in st.session_state:
        st.warning("Please train or load a model first on the 'Model Training & Prediction' page.")
    else:
        st.write(f"Using **{st.session_state['model_name']}** for prediction.")
        st.write("Enter the features for the house you want to predict the value for:")

        # Create input fields for each feature
        input_data = {}
        col1, col2 = st.columns(2)
        
        feature_descriptions = {
            'MedInc': 'Median income in block group (in tens of thousands of dollars)',
            'HouseAge': 'Median house age in block group',
            'AveRooms': 'Average number of rooms per household',
            'AveBedrms': 'Average number of bedrooms per household',
            'Population': 'Block group population',
            'AveOccup': 'Average number of household members',
            'Latitude': 'Block group latitude',
            'Longitude': 'Block group longitude'
        }

        for i, col in enumerate(st.session_state['X_columns']):
            if i % 2 == 0:
                with col1:
                    input_data[col] = st.number_input(
                        f"{col} ({feature_descriptions.get(col, '')})",
                        value=float(X[col].mean()), # Pre-fill with mean for convenience
                        min_value=float(X[col].min()),
                        max_value=float(X[col].max()),
                        step=float((X[col].max() - X[col].min()) / 100),
                        format="%.4f" if col in ['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude'] else "%f"
                    )
            else:
                with col2:
                    input_data[col] = st.number_input(
                        f"{col} ({feature_descriptions.get(col, '')})",
                        value=float(X[col].mean()),
                        min_value=float(X[col].min()),
                        max_value=float(X[col].max()),
                        step=float((X[col].max() - X[col].min()) / 100),
                        format="%.4f" if col in ['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude', 'Longitude'] else "%f"
                    )

        if st.button("Predict House Value"):
            try:
                # Convert input data to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Scale the input data using the trained scaler
                scaler = st.session_state['scaler']
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                trained_model = st.session_state['trained_model']
                prediction = trained_model.predict(input_scaled)[0]
                
                st.markdown(f"### Predicted Median House Value: <span style='color:#4CAF50; font-size:2em;'>${prediction*100000:,.2f}</span>", unsafe_allow_html=True)
                st.info("The predicted value is in hundreds of thousands of dollars. For example, 2.0 means $200,000.")

                # Optional: Show a comparison with average values
                st.markdown("---")
                st.markdown("### Input Data vs. Dataset Averages")
                avg_data = X.mean().to_dict()
                comparison_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Your Input': [input_data[col] for col in X.columns],
                    'Dataset Average': [avg_data[col] for col in X.columns]
                })
                st.dataframe(comparison_df.set_index('Feature'))

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}. Please ensure all inputs are valid numbers.")

st.markdown("---")
st.markdown("Developed for 1st Term AI Examination by Mohit 🚀")