import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import base64

# Set background image
image_path = 'image1.jpg'  
if os.path.exists(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()});
            background-size: cover;
            background-repeat: no-repeat;
        }}
        .stHeader {{
            color: white;
            font-weight: bold;
        }}
        .stText {{
            color: white;
            font-weight: bold;
        }}
        .stTitle {{
            color: white;
            font-weight: bold;
            font-size: 24px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Background image not found.")

# Upload Files
st.title('Customer Satisfaction Dashboard', anchor='dashboard-title')

st.markdown("<h2 class='stHeader'>Upload Engagement and Experience Scores CSV Files</h2>", unsafe_allow_html=True)

uploaded_eng_file = st.file_uploader("Choose an engagement scores CSV file", type="csv")
uploaded_exp_file = st.file_uploader("Choose an experience scores CSV file", type="csv")

if uploaded_eng_file is not None and uploaded_exp_file is not None:
    eng_df = pd.read_csv(uploaded_eng_file)
    exp_df = pd.read_csv(uploaded_exp_file)
    
    st.write("<p class='stText'>Engagement Scores Data</p>", unsafe_allow_html=True)
    st.write(eng_df.head())

    st.write("<p class='stText'>Experience Scores Data</p>", unsafe_allow_html=True)
    st.write(exp_df.head())

    # Combine Data
    df = pd.concat([eng_df, exp_df], axis=1)
    
    # Handle duplicate column names
    df = df.rename(columns=lambda x: x.strip())
    df = df.loc[:, ~df.columns.duplicated()]
    
    df1 = df.fillna(df.median(numeric_only=True))

    # Calculate Satisfaction Score
    df1['satisfaction_score'] = df1[['engagement_score', 'experience_score']].mean(axis=1)

    # Plot Satisfaction Score
    st.markdown("<h2 class='stHeader'>1. Satisfaction Score Distribution</h2>", unsafe_allow_html=True)
    fig_sat_dist = px.histogram(df1, x='satisfaction_score', nbins=30, title='Distribution of Satisfaction Scores')
    st.plotly_chart(fig_sat_dist)

    st.markdown("<h2 class='stHeader'>2. Top 10 Satisfied Customers</h2>", unsafe_allow_html=True)

    # Display Top 10 Satisfied Customers
    top_satisfied_customers = df1[['MSISDN/Number', 'satisfaction_score']].sort_values(by='satisfaction_score', ascending=False).head(10)
    st.write(top_satisfied_customers)

    st.markdown("<h2 class='stHeader'>3. Regression Analysis</h2>", unsafe_allow_html=True)

    # Prepare features and target variable for regression
    X = df1[['engagement_score', 'experience_score']]
    y = df1['satisfaction_score']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    st.write(f'<p class="stText">Linear Regression Mean Squared Error: {mse_lr:.4f}</p>', unsafe_allow_html=True)
    st.write(f'<p class="stText">Linear Regression R-squared: {r2_lr:.4f}</p>', unsafe_allow_html=True)

    # Random Forest Regressor Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    st.write(f'<p class="stText">Random Forest Mean Squared Error: {mse_rf:.4f}</p>', unsafe_allow_html=True)
    st.write(f'<p class="stText">Random Forest R-squared: {r2_rf:.4f}</p>', unsafe_allow_html=True)

    # Save and Load Model
    joblib.dump(rf_model, 'rf_model.pkl')
    with open('rf_model.pkl', 'rb') as file:
        loaded_rf_model = joblib.load(file)

    st.markdown("<h2 class='stHeader'>4. Model Performance</h2>", unsafe_allow_html=True)
    fig_actual_vs_predicted_lr = go.Figure()
    fig_actual_vs_predicted_lr.add_trace(go.Scatter(
        x=y_test,
        y=y_pred_lr,
        mode='markers',
        name='Predicted vs Actual',
        marker=dict(color='blue')
    ))
    fig_actual_vs_predicted_lr.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    fig_actual_vs_predicted_lr.update_layout(
        title='Linear Regression: Actual vs Predicted Satisfaction Scores',
        xaxis_title='Actual Satisfaction Score',
        yaxis_title='Predicted Satisfaction Score'
    )
    st.plotly_chart(fig_actual_vs_predicted_lr)

    fig_residuals_lr = go.Figure()
    residuals_lr = y_test - y_pred_lr
    fig_residuals_lr.add_trace(go.Scatter(
        x=y_pred_lr,
        y=residuals_lr,
        mode='markers',
        name='Residuals',
        marker=dict(color='blue')
    ))
    fig_residuals_lr.add_trace(go.Scatter(
        x=[y_pred_lr.min(), y_pred_lr.max()],
        y=[0, 0],
        mode='lines',
        name='Zero Residual Line',
        line=dict(color='red', dash='dash')
    ))
    fig_residuals_lr.update_layout(
        title='Linear Regression Residual Plot',
        xaxis_title='Predicted Satisfaction Score',
        yaxis_title='Residuals'
    )
    st.plotly_chart(fig_residuals_lr)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig_feature_importance = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importances')
    st.plotly_chart(fig_feature_importance)

    # K-Means Clustering
    st.markdown("<h2 class='stHeader'>5. K-Means Clustering</h2>", unsafe_allow_html=True)

    # Normalize data
    normalized_df = (df1[['engagement_score', 'experience_score']] - df1[['engagement_score', 'experience_score']].mean()) / df1[['engagement_score', 'experience_score']].std()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(normalized_df)
    df1['cluster'] = kmeans.labels_

    # Average Satisfaction Scores by Cluster
    avg_sat_exp = df1.groupby('cluster').agg({
        'satisfaction_score': 'mean',
        'experience_score': 'mean',
        'engagement_score': 'mean'
    }).reset_index()

    fig_avg_scores = px.bar(avg_sat_exp, x='cluster', y=['satisfaction_score', 'experience_score', 'engagement_score'],
                           title='Average Scores by Cluster', labels={'cluster': 'Cluster'})
    st.plotly_chart(fig_avg_scores)

    # Plotting Clusters
    fig_clusters = go.Figure()
    fig_clusters.add_trace(go.Scatter(
        x=df1['engagement_score'],
        y=df1['experience_score'],
        mode='markers',
        marker=dict(color=df1['cluster'], colorscale='Viridis', size=10),
        name='Clustered Data'
    ))
    fig_clusters.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='Centroids'
    ))
    fig_clusters.update_layout(
        title='Engagement vs. Experience Score with Clustering',
        xaxis_title='Engagement Score',
        yaxis_title='Experience Score'
    )
    st.plotly_chart(fig_clusters)

    st.markdown("<h2 class='stHeader'>6. User Count by Cluster</h2>", unsafe_allow_html=True)
    user_count = df1.groupby('cluster').size().reset_index(name='user_count')

    fig_user_count = px.bar(user_count, x='cluster', y='user_count', title='Number of Users by Cluster')
    st.plotly_chart(fig_user_count)

    # Additional Information
    st.markdown("<p class='stText'>Dashboard successfully created by Minal Devikar.</p>", unsafe_allow_html=True)
    st.markdown("<p class='stText'>Special thanks to Shweta Mam for guidance.</p>", unsafe_allow_html=True)
