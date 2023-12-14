#Import Library
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
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
loaded_model = tf.keras.models.load_model('profiling.h5')


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
    
    prediction = loaded_model.predict([[age, gender, income, education, marital_status, number_of_children, home_ownership]])

    # Mendapatkan indeks nilai maksimum
    predicted_class = np.argmax(prediction)

    # Definisikan label kelas sesuai dengan indeks maksimum
    class_labels = ['paragmatic', 'progresif', 'pioneering']
    predicted_label = class_labels[predicted_class]


    return {
        "prediction": prediction.tolist(),
        'label': predicted_label
    }
    

# run API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)





