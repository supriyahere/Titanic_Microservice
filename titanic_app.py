from flask import Flask, request, jsonify
from google.cloud import bigquery

app = Flask(__name__)
client = bigquery.Client()

MODEL = "bufflehead-migration-analysis.bufflehead_in_port.titanic_model"

@app.route("/")
def home():
    return "BigQuery ML Titanic API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    query = f"""
    SELECT predicted_survived
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

    result = list(client.query(query).result())[0]

    return jsonify({
        "predicted_survival": int(result.predicted_survived)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
