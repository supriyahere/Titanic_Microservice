from flask import Flask, request, Response
from google.cloud import bigquery
import json

app = Flask(__name__)

# Enable readable JSON (optional, but good)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Initialize BigQuery client
client = bigquery.Client()

# Your model path
MODEL = "bufflehead-migration-analysis.bufflehead_in_port.titanic_survival_model"


@app.route("/")
def home():
    return "Titanic Survival Prediction API is running."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Build query
    query = f"""
    SELECT predicted_survived, predicted_survived_probs
    FROM ML.PREDICT(
      MODEL `{MODEL}`,
      (
        SELECT
          {data['pclass']} AS pclass,
          '{data['sex']}' AS sex,
          {data['age']} AS age,
          {data['sibsp']} AS sibsp,
          {data['parch']} AS parch,
          {data['fare']} AS fare
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
        "Probability of Survival": round(confidence_survived, 4)
    }

    # Return pretty JSON
    return Response(json.dumps(response_data, indent=4), mimetype='application/json')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
