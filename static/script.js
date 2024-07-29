// CODE USED FOR AN EXTERNAL JS FILE - NOT NEEDED WITH STREAMLIT
// document.getElementById('predict-form').addEventListener('submit', function (e) {
//     e.preventDefault();
//
//     const features = {
//         destination_port: parseFloat(document.getElementById('destination_port').value),
//         total_backward_packets: parseFloat(document.getElementById('total_backward_packets').value),
//         fwd_packet_length_min: parseFloat(document.getElementById('fwd_packet_length_min').value),
//         bwd_packet_length_min: parseFloat(document.getElementById('bwd_packet_length_min').value),
//         bwd_packet_length_mean: parseFloat(document.getElementById('bwd_packet_length_mean').value),
//         bwd_packet_length_std: parseFloat(document.getElementById('bwd_packet_length_std').value),
//         flow_packets: parseFloat(document.getElementById('flow_packets').value),
//         flow_iat_max: parseFloat(document.getElementById('flow_iat_max').value),
//         fwd_iat_total: parseFloat(document.getElementById('fwd_iat_total').value),
//         fwd_iat_mean: parseFloat(document.getElementById('fwd_iat_mean').value),
//         fwd_iat_max: parseFloat(document.getElementById('fwd_iat_max').value),
//         fwd_header_length: parseFloat(document.getElementById('fwd_header_length').value),
//         bwd_header_length: parseFloat(document.getElementById('bwd_header_length').value),
//         fwd_packets: parseFloat(document.getElementById('fwd_packets').value),
//         min_packet_length: parseFloat(document.getElementById('min_packet_length').value),
//         packet_length_mean: parseFloat(document.getElementById('packet_length_mean').value),
//         avg_bwd_segment_size: parseFloat(document.getElementById('avg_bwd_segment_size').value),
//         subflow_bwd_packets: parseFloat(document.getElementById('subflow_bwd_packets').value),
//         init_win_bytes_forward: parseFloat(document.getElementById('init_win_bytes_forward').value),
//         init_win_bytes_backward: parseFloat(document.getElementById('init_win_bytes_backward').value)
//     };
//
//     fetch('/predict', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ features: Object.values(features) }),
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert('Error: ' + data.error);
//         } else {
//             alert('Prediction: ' + (data.prediction === 0 ? 'Benign' : 'Attack'));
//         }
//     })
//     .catch(error => console.error('Error:', error));
// });
