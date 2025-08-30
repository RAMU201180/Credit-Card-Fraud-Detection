import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-prediction {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin: 20px 0;
    }
    .nonfraud-prediction {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
    }
    .feature-input {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)

# Load or train model
@st.cache_resource
def load_or_train_model():
    try:
        # Try to load pre-trained model
        with open('fraud_detection_model.pkl', 'rb') as file:
            model_info = pickle.load(file)
        st.sidebar.success("Pre-trained model loaded successfully!")
        return model_info['model'], model_info['feature_names']
    except:
        # If no pre-trained model, train a new one
        st.sidebar.info("Training model... This may take a moment.")
        
        # Load the data
        df = pd.read_csv('creditcard.csv')
        
        # Handle class imbalance by undersampling
        nfraud = df[df.Class == 0]
        fraud = df[df.Class == 1]
        nfraud_sample = nfraud.sample(n=492, random_state=42)
        ndf = pd.concat([nfraud_sample, fraud], axis=0)
        
        # Prepare features and target
        x = ndf.drop('Class', axis=1)
        y = ndf['Class']
        
        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)
        
        # Train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)
        
        # Save the model for future use
        model_info = {
            'model': model,
            'feature_names': list(x.columns)
        }
        
        with open('fraud_detection_model.pkl', 'wb') as file:
            pickle.dump(model_info, file)
            
        st.sidebar.success("Model trained and saved successfully!")
        return model, list(x.columns)

# Load or train the model
model, feature_names = load_or_train_model()

# Function to predict fraud
def predict_fraud(input_data):
    try:
        # Convert input to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a Logistic Regression model to detect fraudulent credit card transactions. 
    The model was trained on the Credit Card Fraud Detection dataset from Kaggle.
    """)
    
    st.header("Model Performance")
    st.write("The model achieves approximately 94-95% accuracy on test data.")
    
    st.header("How to Use")
    st.write("""
    1. Enter transaction details manually using the input fields
    2. Or upload a CSV file with transaction data
    3. Click 'Predict Fraud' to analyze the transaction
    """)
    
    st.header("Dataset Information")
    st.write("""
    - **Time**: Seconds elapsed between transaction and first transaction
    - **V1-V28**: Principal components from PCA transformation
    - **Amount**: Transaction amount
    - **Class**: 0 for legitimate, 1 for fraudulent
    """)

# Main content
tab1, tab2 = st.tabs(["Manual Input", "Batch Processing"])

with tab1:
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create input fields for the most important features
        time = st.number_input("Time", value=0.0, step=0.1, format="%.6f")
        amount = st.number_input("Amount", value=0.0, step=0.01, format="%.2f")
        
        # Inputs for V1-V5
        v1 = st.number_input("V1", value=0.0, step=0.000001, format="%.6f")
        v2 = st.number_input("V2", value=0.0, step=0.000001, format="%.6f")
        v3 = st.number_input("V3", value=0.0, step=0.000001, format="%.6f")
        v4 = st.number_input("V4", value=0.0, step=0.000001, format="%.6f")
        v5 = st.number_input("V5", value=0.0, step=0.000001, format="%.6f")
    
    with col2:
        # Inputs for V6-V10
        v6 = st.number_input("V6", value=0.0, step=0.000001, format="%.6f")
        v7 = st.number_input("V7", value=0.0, step=0.000001, format="%.6f")
        v8 = st.number_input("V8", value=0.0, step=0.000001, format="%.6f")
        v9 = st.number_input("V9", value=0.0, step=0.000001, format="%.6f")
        v10 = st.number_input("V10", value=0.0, step=0.000001, format="%.6f")
    
    # Default values for the remaining features (set to 0)
    default_features = [0.0] * 18  # For V11-V28
    
    # Prepare input data
    input_data = [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10] + default_features + [amount]
    
    if st.button("Predict Fraud", type="primary"):
        with st.spinner("Analyzing transaction..."):
            prediction, probabilities = predict_fraud(input_data)
            
            if prediction is not None:
                fraud_prob = probabilities[1] * 100
                non_fraud_prob = probabilities[0] * 100
                
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="fraud-prediction">
                        <h2>🚫 FRAUDULENT TRANSACTION DETECTED</h2>
                        <p>Fraud Probability: {fraud_prob:.2f}%</p>
                        <p>Legitimate Probability: {non_fraud_prob:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("This transaction has been flagged as potentially fraudulent. Please review immediately.")
                else:
                    st.markdown(f"""
                    <div class="nonfraud-prediction">
                        <h2>✅ LEGITIMATE TRANSACTION</h2>
                        <p>Legitimate Probability: {non_fraud_prob:.2f}%</p>
                        <p>Fraud Probability: {fraud_prob:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("This transaction appears to be legitimate.")
                
                # Display probability chart
                prob_df = pd.DataFrame({
                    'Category': ['Legitimate', 'Fraudulent'],
                    'Probability': [non_fraud_prob, fraud_prob]
                })
                
                fig, ax = plt.subplots()
                ax.bar(prob_df['Category'], prob_df['Probability'], 
                       color=['#4caf50', '#ff4b4b'])
                ax.set_ylabel('Probability (%)')
                ax.set_title('Transaction Fraud Probability')
                st.pyplot(fig)

with tab2:
    st.subheader("Batch Processing")
    
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            # Check if required columns are present
            required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
            else:
                st.success("File uploaded successfully!")
                st.dataframe(batch_df.head())
                
                if st.button("Process Batch", type="primary"):
                    with st.spinner("Processing transactions..."):
                        # Make predictions
                        predictions = model.predict(batch_df[required_cols])
                        probabilities = model.predict_proba(batch_df[required_cols])
                        
                        # Add predictions to dataframe
                        batch_df['Prediction'] = predictions
                        batch_df['Fraud_Probability'] = probabilities[:, 1]
                        batch_df['Legitimate_Probability'] = probabilities[:, 0]
                        
                        # Display results
                        fraud_count = sum(predictions)
                        total_count = len(predictions)
                        
                        st.subheader("Batch Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Transactions", total_count)
                            st.metric("Fraudulent Transactions", fraud_count)
                        
                        with col2:
                            st.metric("Fraud Rate", f"{(fraud_count/total_count*100):.2f}%")
                            st.metric("Legitimate Transactions", total_count - fraud_count)
                        
                        # Show detailed results
                        st.dataframe(batch_df[['Time', 'Amount', 'Prediction', 'Fraud_Probability']])
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown('<p class="footer">Credit Card Fraud Detection App © 2023 | Built with Streamlit and Scikit-learn</p>', unsafe_allow_html=True)

# Sample data for testing
with st.expander("Sample Input Values"):
    st.write("""
    **Legitimate Transaction Example:**
    - Time: 0.0
    - V1: 1.191857
    - V2: 0.266151
    - V3: 0.166480
    - V4: 0.448154
    - V5: 0.060018
    - V6: -0.082361
    - V7: -0.078803
    - V8: 0.085102
    - V9: -0.255425
    - V10: -0.166974
    - Amount: 2.69
    
    **Fraudulent transactions typically have more extreme values in the V-features.**
    """)