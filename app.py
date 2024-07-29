from flask import Flask, request, jsonify, render_template
import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Network Traffic Classifier')

st.header('Enter Network Traffic Features:')
destination_port = st.number_input('Destination Port', value=0)
total_backward_packets = st.number_input('Total Backward Packets', value=0)
fwd_packet_length_min = st.number_input('FWD Packet Length Min', value=0)
bwd_packet_length_min = st.number_input('BWD Packet Length Min', value=0)
bwd_packet_length_mean = st.number_input('BWD Packet Length Mean', value=0.0)
bwd_packet_length_std = st.number_input('BWD Packet Length Std', value=0.0)
flow_packets = st.number_input('Flow Packets', value=0)
flow_iat_max = st.number_input('Flow IAT Max', value=0.0)
fwd_iat_total = st.number_input('FWD IAT Total', value=0.0)
fwd_iat_mean = st.number_input('FWD IAT Mean', value=0.0)
fwd_iat_max = st.number_input('FWD IAT Max', value=0.0)
fwd_header_length = st.number_input('FWD Header Length', value=0)
bwd_header_length = st.number_input('BWD Header Length', value=0)
fwd_packets = st.number_input('FWD Packets', value=0)
min_packet_length = st.number_input('Min Packet Length', value=0)
packet_length_mean = st.number_input('Packet Length Mean', value=0.0)
avg_bwd_segment_size = st.number_input('Avg BWD Segment Size', value=0.0)
subflow_bwd_packets = st.number_input('Subflow BWD Packets', value=0)
init_win_bytes_forward = st.number_input('Init Win Bytes Forward', value=0)
init_win_bytes_backward = st.number_input('Init Win Bytes Backward', value=0)

# Create a button to make predictions
if st.button('Predict'):
    features = np.array([
        destination_port, total_backward_packets, fwd_packet_length_min,
        bwd_packet_length_min, bwd_packet_length_mean, bwd_packet_length_std,
        flow_packets, flow_iat_max, fwd_iat_total, fwd_iat_mean, fwd_iat_max,
        fwd_header_length, bwd_header_length, fwd_packets, min_packet_length,
        packet_length_mean, avg_bwd_segment_size, subflow_bwd_packets,
        init_win_bytes_forward, init_win_bytes_backward
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    result = 'Benign' if prediction == 0 else 'Attack'
    st.success(f'The traffic is: {result}')


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
