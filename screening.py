#Import Library
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class UKM(BaseModel):
    total_aset: float
    penjualan_rata2: float
    tenaga_kerja: int
    aset_jaminan_kredit: float
    jumlah_dokumen_kredit: int

#Create app object
app = FastAPI()

# Load model Fitur 1
loaded_model = tf.keras.models.load_model('model_screening.h5')


@app.get("/")
def index():
    return {"message": "OK"}


# Endpoint untuk prediksi ukm | fitur ml 1
@app.post("/predict_ukm")
def predict(data: UKM):
    total_aset = data.total_aset
    penjualan_rata2 = data.penjualan_rata2
    tenaga_kerja = data.tenaga_kerja
    aset_jaminan_kredit = data.aset_jaminan_kredit
    jumlah_dokumen_kredit = data.jumlah_dokumen_kredit
    
    prediction = loaded_model.predict([[total_aset, penjualan_rata2, tenaga_kerja, aset_jaminan_kredit, jumlah_dokumen_kredit]])

    if(prediction > 0.5):
        label = "Layak"
    else:
        label = "Tidak Layak"

    return {
        "prediction": prediction.tolist(),
        "label" : label
    }
    

# run API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)






