import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class Pendanaan(BaseModel):
    total_aset: float
    penjualan_rata2: float
    tenaga_kerja: int
    aset_jaminan_kredit: float
    jumlah_dokumen_kredit: int

app = FastAPI()
loaded_model = tf.keras.models.load_model('model_valuation.h5')

@app.get("/")
def home():
    return {"message": "OK"}

@app.post("/predict")
def predict(data: Pendanaan):
    total_aset = data.total_aset
    penjualan_rata2 = data.penjualan_rata2
    tenaga_kerja = data.tenaga_kerja
    aset_jaminan_kredit = data.aset_jaminan_kredit
    jumlah_dokumen_kredit = data.jumlah_dokumen_kredit


    pendanaan = loaded_model.predict([[total_aset, penjualan_rata2, tenaga_kerja, aset_jaminan_kredit, jumlah_dokumen_kredit]])
    return {"pendanaan": pendanaan.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn app.main:app --reload