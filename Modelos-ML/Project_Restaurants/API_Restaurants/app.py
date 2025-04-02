from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

modelo = joblib.load("best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        dados = request.get_json()
        entrada = pd.DataFrame([dados])
        predicao = modelo.predict(entrada)
        resultado = predicao.tolist()[0]

        if resultado == 1:
            mensagem = "Restaurante Bem Avaliado"
        else:
            mensagem = "Restaurante Mal Avaliado"

        return jsonify({
            "prediction": resultado,
            "message": mensagem
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})
    

if __name__ == "__main__":
    app.run(debug=True)