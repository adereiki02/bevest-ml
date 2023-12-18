#Import Library
# from numpy import double
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import math
import gunicorn

# Define class model

class UKM(BaseModel):
    total_aset: float
    penjualan_rata2: float
    tenaga_kerja: int
    aset_jaminan_kredit: float
    jumlah_dokumen_kredit: int

class Pendanaan(BaseModel):
    total_aset: float
    penjualan_rata2: float
    tenaga_kerja: int
    aset_jaminan_kredit: float
    jumlah_dokumen_kredit: int

class Investor(BaseModel):
    age: int
    gender: int
    income: float
    education: int
    marital_status: int
    number_of_children: int
    home_ownership: int

# Define FastAPI for webserver
app = FastAPI()

# Load model fitur 1, 2, 3 
screening_model = tf.keras.models.load_model('model_screening.h5')
valuation_model = tf.keras.models.load_model('model_valuation.h5')
profilling_model = tf.keras.models.load_model('model_profiling.h5')

# Endpoint index
@app.get("/")
def index():
    return {'message': 'OK'}

# Endpoint screening ukm (skrining ukm) | fitur ml 1
@app.post("/screening")
def predict(data: UKM):
    total_aset = data.total_aset
    penjualan_rata2 = data.penjualan_rata2
    tenaga_kerja = data.tenaga_kerja
    aset_jaminan_kredit = data.aset_jaminan_kredit
    jumlah_dokumen_kredit = data.jumlah_dokumen_kredit

    screening_result = screening_model.predict([[total_aset, penjualan_rata2, tenaga_kerja, aset_jaminan_kredit, jumlah_dokumen_kredit]])

    if (screening_result > 0.5 ):
        label = "Layak"
    else:
        label = "Tidak Layak"

    return {
        "Screening Result:": screening_result.tolist(),
        "Label:": label
    }

# Endpoint profilling investor (investor profilling) | fitur ml 2
@app.post("/profilling")
def predict(data: Investor):
    age = data.age
    gender = data.gender
    income = data.income
    education = data.education
    marital_status = data.marital_status
    number_of_children = data.number_of_children
    home_ownership = data.home_ownership

    prediction = profilling_model.predict([[age, gender, income, education, marital_status, number_of_children, home_ownership]])

    # Mendapatkan indeks nilai maksimum
    predicted_class = np.argmax(prediction)

    # Definisikan label kelas sesuai dengan indeks maksimum
    class_labels = ['paragmatic', 'progresif', 'pioneering']
    predicted_label = class_labels[predicted_class]


    return {
        "prediction": prediction.tolist(),
        'label': predicted_label
    }

# Endpoint pendanaan ukm (business valuation) | fitur ml 3
@app.post("/valuation")
def predict(data: Pendanaan):
    total_aset = data.total_aset
    penjualan_rata2 = data.penjualan_rata2
    tenaga_kerja = data.tenaga_kerja
    aset_jaminan_kredit = data.aset_jaminan_kredit
    jumlah_dokumen_kredit = data.jumlah_dokumen_kredit

    valuation_result = valuation_model.predict([[total_aset, penjualan_rata2, tenaga_kerja, aset_jaminan_kredit, jumlah_dokumen_kredit]])
    
    rounded_value = math.floor(valuation_result[0][0])*1000000
                               
    return {
        "Valuation:": rounded_value
    }

# run API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)