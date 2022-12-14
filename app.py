from flask import Flask, jsonify, request
from main import get_prediction

app = Flask(__name__)

@app.route("/predict-digit", methods = ["POST"])
def predict_data():
    img = request.files.get("digit")
    pred = get_prediction(img)
    return jsonify({
        "prediction": pred
    }), 200

if __name__ == "__main__":
    app.run(debug=True)