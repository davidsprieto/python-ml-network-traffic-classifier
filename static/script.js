// static/script.js
document.getElementById('predict-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const inputData = document.getElementById('input-data').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data: JSON.parse(inputData) })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('results').innerText = JSON.stringify(data);
    });
});
