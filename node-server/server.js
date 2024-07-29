// CODE USED FOR A NODE/EXPRESS SERVER - NOT NEEDED WITH STREAMLIT
// const express = require('express');
// const path = require('path');
// const fetch = require('node-fetch');
//
// const app = express();
// const port = process.env.PORT || 3000;
//
// // Middleware to parse JSON bodies
// app.use(express.json());
//
// // Serve static files (like the JS file and CSS)
// app.use(express.static(path.join(__dirname, 'public')));
//
// // Serve the HTML file for the homepage
// app.get('/', (req, res) => {
//     res.sendFile(path.join(__dirname, 'public', 'index.html'));
// });
//
// // Endpoint to forward the prediction request to the Flask app
// app.post('/predict', async (req, res) => {
//     try {
//         const response = await fetch('http://localhost:5000/predict', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify(req.body)
//         });
//         const data = await response.json();
//         res.json(data);
//     } catch (error) {
//         console.error('Error:', error);
//         res.status(500).send('Internal Server Error');
//     }
// });
//
// // Start the server
// app.listen(port, () => {
//     console.log(`Server running on port ${port}`);
// });
