from fastapi import FastAPI
import joblib
import numpy as np
import os

model = joblib.load('app/model.joblib')

class_names = np.array(['Bottom 6','Mid-Table', 'European Places'])

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message':'Premier League Prediction'}

@app.post('/predict')
def predict(data:dict):
    features = np.array(data['features'].reshape(1,-1))
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted'}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))