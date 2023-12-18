#Import Library
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from preprocessing import preprocess_data
import uvicorn

class profiling(BaseModel):
    age: int
    gender: int
    income: float
    education: int
    marital_status: int
    number_of_children: int
    home_ownership: int

#Create app object
app = FastAPI()

# Load model Fitur 1

def predict_profile(normalized_data):
    # Load the TensorFlow model
    model = tf.keras.models.load_model('investor_profiling.h5')
    predictions = model.predict(normalized_data)
    return predictions


@app.get("/")
def index():
    return {"message": "OK"}


# Endpoint untuk prediksi ukm | fitur ml 1
@app.post("/predict_profiling")
def predict(data: profiling):
    age = data.age
    gender = data.gender
    income = data.income
    education = data.education
    marital_status = data.marital_status
    number_of_children = data.number_of_children
    home_ownership = data.home_ownership
    
    normalized_data = preprocess_data(age, gender, income, education, marital_status, number_of_children, home_ownership)
    predictions = predict_profile(normalized_data)

    # Mendapatkan indeks nilai maksimum
    predicted_class = np.argmax(predictions)

    # Definisikan label kelas sesuai dengan indeks maksimum
    class_labels = ['Pragmatic', 'Progresif', 'Pioneering']
    predicted_label = class_labels[predicted_class]


    return {
        "prediction": predictions.tolist(),
        'label': predicted_label
    }
    

# run API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)





