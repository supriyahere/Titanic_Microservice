from flask import Flask, request, Response
from google.cloud import bigquery
import json

app = Flask(__name__)

# Initialize BigQuery client
client = bigquery.Client()

# BigQuery ML model path
MODEL = "bufflehead-migration-analysis.bufflehead_in_port.titanic_survival_model"


@app.route("/")
def home():
    return "Titanic Survival Prediction API is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ Validate input
        if not data:
            return {"error": "No input data provided"}, 400

        required_fields = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

        for field in required_fields:
            if field not in data:
                return {"error": f"Missing field: {field}"}, 400

        # Extract values safely
        pclass = data.get("pclass")
        sex = data.get("sex")
        age = data.get("age")
        sibsp = data.get("sibsp")
        parch = data.get("parch")
        fare = data.get("fare")

        # Build query
        query = f"""
        SELECT predicted_survived, predicted_survived_probs
        FROM ML.PREDICT(
          MODEL `{MODEL}`,
          (
            SELECT
              {pclass} AS pclass,
              '{sex}' AS sex,
              {age} AS age,
              {sibsp} AS sibsp,
              {parch} AS parch,
              {fare} AS fare
          )
        )
        """

        # Run query
        result = list(client.query(query).result())[0]

        # Extract probabilities
        probs = {item['label']: item['prob'] for item in result.predicted_survived_probs}

        confidence_survived = float(probs.get(1, 0))
        confidence_not_survived = float(probs.get(0, 0))

        # Build response
        response_data = {
            "Prediction": "Survived" if result.predicted_survived == 1 else "Did not survive",
            "Survival Probability (%)": round(confidence_survived * 100, 2),
            "Non-Survival Probability (%)": round(confidence_not_survived * 100, 2)
        }

        # Pretty JSON output
        return Response(json.dumps(response_data, indent=4), mimetype='application/json')

    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
