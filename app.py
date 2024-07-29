import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the models for each technique and classifier
model_files = {
    "SelectKBest": {
        "Logistic Regression": 'logistic_regression_selectkbest.pkl',
        "SVM": 'svm_selectkbest.pkl',
        "k-NN": 'knn_selectkbest.pkl',
        "Random Forest": 'random_forest_selectkbest.pkl',
        "Gradient Boosting": 'gradient_boosting_selectkbest.pkl'
    },
    "Constant Features Removed": {
        "Logistic Regression": 'logistic_regression_constant_features_removed.pkl',
        "SVM": 'svm_constant_features_removed.pkl',
        "k-NN": 'knn_constant_features_removed.pkl',
        "Random Forest": 'random_forest_constant_features_removed.pkl',
        "Gradient Boosting": 'gradient_boosting_constant_features_removed.pkl'
    },
    "Near-Zero Variance Features Removed": {
        "Logistic Regression": 'logistic_regression_near_zero_variance_removed.pkl',
        "SVM": 'svm_near_zero_variance_removed.pkl',
        "k-NN": 'knn_near_zero_variance_removed.pkl',
        "Random Forest": 'random_forest_near_zero_variance_removed.pkl',
        "Gradient Boosting": 'gradient_boosting_near_zero_variance_removed.pkl'
    },
    "LASSO": {
        "Logistic Regression": 'logistic_regression_lasso.pkl',
        "SVM": 'svm_lasso.pkl',
        "k-NN": 'knn_lasso.pkl',
        "Random Forest": 'random_forest_lasso.pkl',
        "Gradient Boosting": 'gradient_boosting_lasso.pkl'
    },
    "Random Forest": {
        "Logistic Regression": 'logistic_regression_random_forest.pkl',
        "SVM": 'svm_random_forest.pkl',
        "k-NN": 'knn_random_forest.pkl',
        "Random Forest": 'random_forest_random_forest.pkl',
        "Gradient Boosting": 'gradient_boosting_random_forest.pkl'
    },
    "PCA": {
        "Logistic Regression": 'logistic_regression_pca.pkl',
        "SVM": 'svm_pca.pkl',
        "k-NN": 'knn_pca.pkl',
        "Random Forest": 'random_forest_pca.pkl',
        "Gradient Boosting": 'gradient_boosting_pca.pkl'
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


# Function to prettify feature names
def prettify_feature_name(name):
    return ' '.join([word.capitalize() for word in name.split('_')])


# Title and dropdowns
st.title('Network Traffic Classifier - Benign or Attack')
selected_technique = st.selectbox("Select Feature Selection Technique", list(model_files.keys()))
selected_model = st.selectbox("Select Model", list(model_files[selected_technique].keys()))

# Load the selected model
model_path = model_files[selected_technique][selected_model]
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Get the corresponding feature set
features = feature_sets[selected_technique]

# Collect user inputs for the selected features
input_data = {}
for feature in features:
    input_data[feature] = st.text_input(prettify_feature_name(feature))

# Convert inputs to a DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f"The predicted class is: {'Benign' if prediction[0] == 0 else 'Attack'}")

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
