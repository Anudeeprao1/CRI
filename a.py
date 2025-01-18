import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load CSV Files ---
@st.cache_data
def load_csv_files(file1, file2):
    """
    Load two predefined CSV files into pandas DataFrames.
    """
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        return df1, df2
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

# --- 2. Feature Importance (Before Clustering) ---
def compute_overall_feature_importance(df):
    """
    Compute feature importance for the entire dataset before clustering.
    """
    st.subheader("Overall Features Importance")

    # Separate features and target
    features = df.drop(columns=['Country', 'Year and survey', 'Multidimensional Poverty Index Value'])
    target = df['Multidimensional Poverty Index Value']

    # Train XGBoost model
    dtrain = xgb.DMatrix(features, label=target)
    params = {
        'objective': 'reg:squarederror',  # Regression objective
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    }
    xg_reg = xgb.train(params, dtrain, num_boost_round=100)

    # Get feature importance
    importance = xg_reg.get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Visualize feature importance
    plt.figure(figsize=(10, 5))
    plt.bar(*zip(*sorted_importance), color='skyblue')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance (Weight)')
    plt.xticks(rotation=90)
    st.pyplot(plt)

    return sorted_importance

# --- 3. Cluster Analysis ---
def perform_clustering(df, num_clusters=3):
    """
    Perform clustering analysis using K-Means and add cluster labels to the DataFrame.
    """
    # Drop non-numeric columns and separate target
    features = df.drop(columns=['Country', 'Year and survey', 'Multidimensional Poverty Index Value'])
    target = df['Multidimensional Poverty Index Value']

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_scaled)

    # Visualize Clusters (using PCA for 2D projection)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(features_scaled)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['Cluster'], palette='Set2', s=100)
    plt.title('Clusters Based on Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)

    return df

# --- 4. Feature Importance for Each Cluster ---
def compute_feature_importance(df):
    """
    Compute feature importance for each cluster using XGBoost.
    """
    cluster_feature_importances = {}

    for cluster in sorted(df['Cluster'].unique()):
        st.subheader(f"Analyzing Feature Importance for Cluster {cluster}")

        # Filter data for the current cluster
        cluster_data = df[df['Cluster'] == cluster]
        cluster_target = cluster_data['Multidimensional Poverty Index Value']
        cluster_features = cluster_data.drop(columns=['Country', 'Year and survey', 'Cluster', 'Multidimensional Poverty Index Value'])

        # Convert data to DMatrix for XGBoost
        dtrain = xgb.DMatrix(cluster_features, label=cluster_target)
        params = {
            'objective': 'reg:squarederror',  # Regression objective
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }

        # Train XGBoost model
        xg_reg = xgb.train(params, dtrain, num_boost_round=100)

        # Get feature importance
        importance = xg_reg.get_score(importance_type='weight')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        cluster_feature_importances[cluster] = sorted_importance

        # Visualize feature importance
        plt.figure(figsize=(10, 5))
        plt.bar(*zip(*sorted_importance), color='skyblue')
        plt.title(f'Feature Importance for Cluster {cluster}')
        plt.xlabel('Features')
        plt.ylabel('Importance (Weight)')
        plt.xticks(rotation=90)
        st.pyplot(plt)

    return cluster_feature_importances

# --- Streamlit App ---
def main():
    st.title("FACTORES AFFECTING SOCIODEMOGRAPHIC CONDITIONS OF COUNTRIES")
    

    # Predefined file paths
    file1 = "filename1.csv"  # Replace with the name of your first file
    file2 = "filename.csv"  # Replace with the name of your second file
    

    # Load the files
    df1, df2 = load_csv_files(file1, file2)
    df1.fillna(0, inplace=True)
    df2.fillna(0, inplace=True)

    if df1 is not None and df2 is not None:
        
        # Feature importance for the entire dataset before clustering
        overall_importance_1 = compute_overall_feature_importance(df1)

        # Perform clustering and feature importance for file 1
        df1 = perform_clustering(df1, num_clusters=3)
        cluster_feature_importances_1 = compute_feature_importance(df1)

        st.subheader("Clustered Countries")
        clustered_countries_1 = df1.groupby("Cluster")["Country"].apply(list).reset_index()
        st.write(clustered_countries_1)

        
        # Feature importance for the entire dataset before clustering
        overall_importance_2 = compute_overall_feature_importance(df2)

        # Perform clustering and feature importance for file 2
        df2 = perform_clustering(df2, num_clusters=3)
        cluster_feature_importances_2 = compute_feature_importance(df2)

        # Display results
        st.subheader("Clustered Countries")
        clustered_countries_2 = df2.groupby("Cluster")["Country"].apply(list).reset_index()
        st.write(clustered_countries_2)

if __name__ == "__main__":
    main()
