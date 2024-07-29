import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the models for each technique and classifier
model_files = {
    "SelectKBest": {
        "Logistic Regression": 'feature_selection/select_k_best/logistic_regression_selectkbest.pkl',
        "SVM": 'feature_selection/select_k_best/svm_selectkbest.pkl',
        "k-NN": 'feature_selection/select_k_best/knn_selectkbest.pkl',
        "Random Forest": 'feature_selection/select_k_best/random_forest_selectkbest.pkl',
        "Gradient Boosting": 'feature_selection/select_k_best/gradient_boosting_selectkbest.pkl'
    },
    "Constant Features Removed": {
        "Logistic Regression": 'feature_selection/constant_features_removed/logistic_regression_constant_features_removed.pkl',
        "SVM": 'feature_selection/constant_features_removed/svm_constant_features_removed.pkl',
        "k-NN": 'feature_selection/constant_features_removed/knn_constant_features_removed.pkl',
        "Random Forest": 'feature_selection/constant_features_removed/random_forest_constant_features_removed.pkl',
        "Gradient Boosting": 'feature_selection/constant_features_removed/gradient_boosting_constant_features_removed.pkl'
    },
    "Near-Zero Variance Features Removed": {
        "Logistic Regression": 'feature_selection/variance_threshold/logistic_regression_near_zero_variance_removed.pkl',
        "SVM": 'feature_selection/variance_threshold/svm_near_zero_variance_removed.pkl',
        "k-NN": 'feature_selection/variance_threshold/knn_near_zero_variance_removed.pkl',
        "Random Forest": 'feature_selection/variance_threshold/random_forest_near_zero_variance_removed.pkl',
        "Gradient Boosting": 'feature_selection/variance_threshold/gradient_boosting_near_zero_variance_removed.pkl'
    },
    "LASSO": {
        "Logistic Regression": 'feature_selection/lasso/logistic_regression_lasso.pkl',
        "SVM": 'feature_selection/lasso/svm_lasso.pkl',
        "k-NN": 'feature_selection/lasso/knn_lasso.pkl',
        "Random Forest": 'feature_selection/lasso/random_forest_lasso.pkl',
        "Gradient Boosting": 'feature_selection/lasso/gradient_boosting_lasso.pkl'
    },
    "Random Forest": {
        "Logistic Regression": 'feature_selection/random_forest/logistic_regression_random_forest.pkl',
        "SVM": 'feature_selection/random_forest/svm_random_forest.pkl',
        "k-NN": 'feature_selection/random_forest/knn_random_forest.pkl',
        "Random Forest": 'feature_selection/random_forest/random_forest_random_forest.pkl',
        "Gradient Boosting": 'feature_selection/random_forest/gradient_boosting_random_forest.pkl'
    },
    "PCA": {
        "Logistic Regression": 'feature_selection/pca/logistic_regression_pca.pkl',
        "SVM": 'feature_selection/pca/svm_pca.pkl',
        "k-NN": 'feature_selection/pca/knn_pca.pkl',
        "Random Forest": 'feature_selection/pca/random_forest_pca.pkl',
        "Gradient Boosting": 'feature_selection/pca/gradient_boosting_pca.pkl'
    }
}

# Feature sets for each technique
feature_sets = {
    "SelectKBest": ['destination_port', 'flow_duration', 'total_fwd_packets',
                    'total_backward_packets', 'total_length_of_fwd_packets',
                    'total_length_of_bwd_packets', 'fwd_packet_length_max',
                    'fwd_packet_length_min', 'fwd_packet_length_mean',
                    'fwd_packet_length_std', 'bwd_packet_length_max',
                    'bwd_packet_length_min', 'bwd_packet_length_mean',
                    'bwd_packet_length_std', 'flow_bytes', 'flow_packets', 'flow_iat_mean',
                    'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_total',
                    'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
                    'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
                    'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags',
                    'bwd_urg_flags', 'fwd_header_length', 'bwd_header_length',
                    'fwd_packets', 'bwd_packets', 'min_packet_length', 'max_packet_length',
                    'packet_length_mean', 'packet_length_std', 'packet_length_variance',
                    'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
                    'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
                    'downup_ratio', 'average_packet_size', 'avg_fwd_segment_size',
                    'avg_bwd_segment_size', 'fwd_header_length1', 'fwd_avg_bytesbulk',
                    'fwd_avg_packetsbulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytesbulk',
                    'bwd_avg_packetsbulk', 'bwd_avg_bulk_rate', 'subflow_fwd_packets',
                    'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes',
                    'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd',
                    'min_seg_size_forward', 'active_mean', 'active_std', 'active_max',
                    'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min'],
    "Constant Features Removed": ['destination_port', 'flow_duration', 'total_fwd_packets',
                                  'total_backward_packets', 'total_length_of_fwd_packets',
                                  'total_length_of_bwd_packets', 'fwd_packet_length_max',
                                  'fwd_packet_length_min', 'fwd_packet_length_mean',
                                  'fwd_packet_length_std', 'bwd_packet_length_max',
                                  'bwd_packet_length_min', 'bwd_packet_length_mean',
                                  'bwd_packet_length_std', 'flow_bytes', 'flow_packets', 'flow_iat_mean',
                                  'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_total',
                                  'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
                                  'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
                                  'bwd_iat_min', 'fwd_psh_flags', 'fwd_header_length',
                                  'bwd_header_length', 'fwd_packets', 'bwd_packets', 'min_packet_length',
                                  'max_packet_length', 'packet_length_mean', 'packet_length_std',
                                  'packet_length_variance', 'fin_flag_count', 'syn_flag_count',
                                  'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
                                  'ece_flag_count', 'downup_ratio', 'average_packet_size',
                                  'avg_fwd_segment_size', 'avg_bwd_segment_size', 'fwd_header_length1',
                                  'subflow_fwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_packets',
                                  'subflow_bwd_bytes', 'init_win_bytes_forward',
                                  'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                                  'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean',
                                  'idle_std', 'idle_max', 'idle_min'],
    "Near-Zero Variance Features Removed": ['destination_port', 'flow_duration', 'total_fwd_packets',
                                            'total_backward_packets', 'total_length_of_fwd_packets',
                                            'total_length_of_bwd_packets', 'fwd_packet_length_max',
                                            'fwd_packet_length_min', 'fwd_packet_length_mean',
                                            'fwd_packet_length_std', 'bwd_packet_length_max',
                                            'bwd_packet_length_min', 'bwd_packet_length_mean',
                                            'bwd_packet_length_std', 'flow_bytes', 'flow_packets', 'flow_iat_mean',
                                            'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_total',
                                            'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
                                            'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
                                            'bwd_iat_min', 'fwd_psh_flags', 'fwd_header_length',
                                            'bwd_header_length', 'fwd_packets', 'bwd_packets', 'min_packet_length',
                                            'max_packet_length', 'packet_length_mean', 'packet_length_std',
                                            'packet_length_variance', 'fin_flag_count', 'syn_flag_count',
                                            'psh_flag_count', 'ack_flag_count', 'urg_flag_count', 'downup_ratio',
                                            'average_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size',
                                            'fwd_header_length1', 'subflow_fwd_packets', 'subflow_fwd_bytes',
                                            'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_win_bytes_forward',
                                            'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                                            'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean',
                                            'idle_std', 'idle_max', 'idle_min'],
    "LASSO": ['destination_port', 'flow_duration', 'total_fwd_packets',
              'total_backward_packets', 'total_length_of_fwd_packets',
              'total_length_of_bwd_packets', 'fwd_packet_length_max',
              'fwd_packet_length_min', 'fwd_packet_length_mean',
              'fwd_packet_length_std', 'bwd_packet_length_max',
              'bwd_packet_length_min', 'bwd_packet_length_mean',
              'bwd_packet_length_std', 'flow_bytes', 'flow_iat_mean', 'flow_iat_min',
              'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
              'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
              'bwd_iat_min', 'fwd_psh_flags', 'bwd_header_length', 'fwd_packets',
              'bwd_packets', 'min_packet_length', 'packet_length_std',
              'packet_length_variance', 'fin_flag_count', 'syn_flag_count',
              'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
              'ece_flag_count', 'downup_ratio', 'average_packet_size',
              'avg_fwd_segment_size', 'avg_bwd_segment_size', 'subflow_fwd_packets',
              'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes',
              'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
              'active_mean', 'active_std', 'idle_std', 'idle_min'],
    "Random Forest": ['destination_port', 'total_backward_packets', 'fwd_packet_length_min',
                      'bwd_packet_length_min', 'bwd_packet_length_mean',
                      'bwd_packet_length_std', 'flow_packets', 'flow_iat_max',
                      'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_max', 'fwd_header_length',
                      'bwd_header_length', 'fwd_packets', 'min_packet_length',
                      'packet_length_mean', 'avg_bwd_segment_size', 'subflow_bwd_packets',
                      'init_win_bytes_forward', 'init_win_bytes_backward'],
    "PCA": ['flow_bytes', 'destination_port', 'bwd_iat_min', 'urg_flag_count',
            'min_packet_length', 'idle_std', 'fwd_packet_length_min',
            'total_length_of_fwd_packets', 'subflow_fwd_bytes', 'bwd_packets',
            'flow_iat_min', 'fwd_packet_length_mean', 'avg_fwd_segment_size',
            'flow_iat_std', 'min_seg_size_forward', 'init_win_bytes_backward',
            'fwd_iat_min', 'fin_flag_count', 'active_min', 'downup_ratio', 'active_mean']
}


# Define your pages as functions
def home():
    st.title("ML Network Traffic Classifier")
    st.write("""
    Welcome to the Network Traffic Classifier web app! This application uses various machine learning techniques 
    to classify network traffic as either benign or an attack. You can navigate through different sections using 
    the sidebar to learn more about the models used and to test the classifier with your own data.
    """)


def about():
    st.title("About This Project")
    st.write("""
    This project leverages multiple feature selection and extraction techniques combined with various 
    machine learning models to classify network traffic data. The primary goal is to identify whether 
    the network traffic is benign or an attack (e.g., DoS attack). The feature techniques include:

    - SelectKBest
    - Removing Constant Features
    - Removing Near-Zero Variance Features
    - LASSO Feature Selection
    - Random Forest Feature Importance
    - Principal Component Analysis (PCA)

    The machine learning models used are:

    - Logistic Regression
    - Support Vector Machine (SVM)
    - k-Nearest Neighbors (k-NN)
    - Random Forest
    - Gradient Boosting
    """)


# def data():
#     st.title("Data Exploration")
#     st.write("""
#     Here, you can explore the dataset used for training the models. This includes visualizations
#     and statistics that help understand the features and their distributions.
#     """)
#     # Add any data exploration code and visualizations here
#     # For example, you can include charts and summary statistics of your dataset


def feature_selection():
    st.title("Feature Selection Techniques")
    st.write("""
    This section provides insights into the different feature selection and extraction techniques used in this project.
    Choose a technique to see the features and their importance scores.
    """)

    technique = st.selectbox("Select Feature Selection Technique",
                             ["SelectKBest", "Constant Features Removed", "Near-Zero Variance Features Removed", "LASSO",
                              "Random Forest",
                              "PCA"])

    if technique == "SelectKBest":
        st.write("Top features selected using SelectKBest:")
        feature_scores_df = pd.DataFrame(
            {'Feature': feature_sets[technique]}, index=range(1, len(feature_sets[technique]) + 1))
        st.write(feature_scores_df)

    elif technique == "Constant Features Removed":
        st.write("Features remaining after removing constants:")
        feature_scores_df = pd.DataFrame(
            {'Feature': feature_sets[technique]}, index=range(1, len(feature_sets[technique]) + 1))
        st.write(feature_scores_df)

    elif technique == "Near-Zero Variance Features Removed":
        st.write("Features remaining after removing those with near-zero variance:")
        feature_scores_df = pd.DataFrame(
            {'Feature': feature_sets[technique]}, index=range(1, len(feature_sets[technique]) + 1))
        st.write(feature_scores_df)

    elif technique == "LASSO":
        st.write("Features selected using LASSO:")
        feature_scores_df = pd.DataFrame(
            {'Feature': feature_sets[technique]}, index=range(1, len(feature_sets[technique]) + 1))
        st.write(feature_scores_df)

    elif technique == "Random Forest":
        st.write("Top features selected using Random Forest:")
        feature_scores_df = pd.DataFrame(
            {'Feature': feature_sets[technique]}, index=range(1, len(feature_sets[technique]) + 1))
        st.write(feature_scores_df)

    elif technique == "PCA":
        st.write("Principal Components selected using PCA:")
        feature_scores_df = pd.DataFrame(
            {'Feature': feature_sets[technique]}, index=range(1, len(feature_sets[technique]) + 1))
        st.write(feature_scores_df)


def predict():
    st.title("Make a Prediction")
    st.write("""
    Select the feature selection technique and model you want to use for making predictions. 
    Enter the feature values for the selected technique and click 'Predict' to classify the network traffic.
    """)

    technique = st.selectbox("Select Feature Technique:",
                             ["SelectKBest", "Constant Features Removed", "Near-Zero Variance Features Removed", "LASSO", "Random Forest",
                              "PCA"])

    model_name = st.selectbox("Select Model:",
                              ["Logistic Regression", "SVM", "k-NN", "Random Forest", "Gradient Boosting"])

    features = feature_sets[technique]
    feature_values = [st.number_input(f'{feature.replace("_", " ").title()}:') for feature in features]
    feature_values = np.array(feature_values).reshape(1, -1)

    if st.button("Predict"):
        if np.all(feature_values == 0):
            st.error("Error: All feature values cannot be 0. Please enter valid values.")
        else:
            try:
                model_path = model_files[technique][model_name]
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)

                prediction = model.predict(feature_values)
                st.write(f"The predicted class is: {'Benign' if prediction == 0 else 'Attack'}")
            except Exception as e:
                st.error(f"An error occurred: {e}")


# Page navigation
page = st.sidebar.radio("Navigate to:",
                        ["Home", "About", "Feature Selection Techniques", "Make a Prediction"])

if page == "Home":
    home()
elif page == "About":
    about()
# elif page == "Data Exploration":
#     data()
elif page == "Feature Selection Techniques":
    feature_selection()
elif page == "Make a Prediction":
    predict()

#
#
#
# CODE USED FOR AN EXTERNAL HTML & JS FILE - NOT NEEDED WITH STREAMLIT
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         features = np.array(data['features']).reshape(1, -1)
#         prediction = model.predict(features)[0]
#         return jsonify({'prediction': int(prediction)})
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'error': str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
