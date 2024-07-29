document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('prediction-form').addEventListener('submit', function (e) {
        e.preventDefault();
        predict();
    });
});

function predict() {
    const data = document.getElementById('data').value.split(',').map(Number);

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: data })
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

