from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open("sentiment_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Sentiment Analysis API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review = data.get("review")

    prediction = model.predict([review])[0]

    return jsonify({
        "review": review,
        "sentiment": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
