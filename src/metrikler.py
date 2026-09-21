"""
Hata metrikleri: MAE, MAPE, RMSE.

MAPE'de gercek degeri sifir olan periyotlar haric tutulur
(sifira bolme). 30 dakikalik sette 8 periyotta cagri sayisi sifir.
"""

import numpy as np
import pandas as pd


def hesapla(gercek, tahmin):
    y = np.asarray(gercek, dtype=float)
    p = np.clip(np.asarray(tahmin, dtype=float), 0, None)   # negatif tahmin olamaz

    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))

    gecerli = y > 0
    mape = np.mean(np.abs((y[gecerli] - p[gecerli]) / y[gecerli])) * 100

    return {
               "MAE": round(float(mae), 2),
        "MAPE": round(float(mape), 2),
        "RMSE": round(float(rmse), 2),
        "haric": int((~gecerli).sum()),   # MAPE'den cikarilan satir sayisi
    }


def tablo(sonuclar):
    """sonuclar: [{'model':..., 'MAE':..., ...}, ...] -> DataFrame"""
    return pd.DataFrame(sonuclar)


if __name__ == "__main__":
    y = np.array([100, 200, 0, 50, 80])
    p = np.array([110, 180, 5, 45, -10])
    print(hesapla(y, p))