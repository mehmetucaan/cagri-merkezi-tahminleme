"""
Modeller, ortak arayuz: .egit_tahmin(tr, te) -> tahmin dizisi

XGBoost : agac tabanli, ozellik tablosunu dogrudan kullanir
MLP     : sinir agi, ayni ozellikleri kullanir ama olcekleme sart
LSTM    : dizi modeli, ham seriyi pencereleyerek kullanir

Secilen modeller XGBoost + MLP. LSTM karsilastirma calismasinda
kullanildi, sonuclari raporda duruyor; sinifi referans olarak burada
kaliyor ama varsayilan olarak calistirilmiyor.
LSTM'i de kosmak icin AKTIF_MODELLER listesine "LSTM" ekle.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

TOHUM = 42
AKTIF_MODELLER = ["XGBoost", "MLP", "ExtraTrees", "LightGBM", "CatBoost"]


class XGB:
    ad = "XGBoost"

    def __init__(self, ozellikler, hedef):
        self.oz, self.hedef = ozellikler, hedef

    def egit_tahmin(self, tr, te):
        m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, random_state=TOHUM, n_jobs=-1)
        m.fit(tr[self.oz], tr[self.hedef])
        self.model = m
        return m.predict(te[self.oz])

    def ozellik_onemi(self):
        """Hangi ozellik ne kadar katki yapmis - rapor ve arayuz icin."""
        return dict(sorted(zip(self.oz, self.model.feature_importances_),
                           key=lambda x: -x[1]))


class MLP:
    ad = "MLP"

    def __init__(self, ozellikler, hedef):
        self.oz, self.hedef = ozellikler, hedef

    def egit_tahmin(self, tr, te):
        # MLP olceklenmemis veride cokuyor - hem X hem y olceklenir
        sx, sy = StandardScaler(), StandardScaler()
        X = sx.fit_transform(tr[self.oz])
        y = sy.fit_transform(tr[[self.hedef]]).ravel()

        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         early_stopping=True, random_state=TOHUM)
        m.fit(X, y)
        self.model = m

        p = m.predict(sx.transform(te[self.oz]))
        return sy.inverse_transform(p.reshape(-1, 1)).ravel()


class ExtraTrees:
    ad = "ExtraTrees"

    def __init__(self, ozellikler, hedef):
        self.oz, self.hedef = ozellikler, hedef

    def egit_tahmin(self, tr, te):
        m = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                random_state=TOHUM, n_jobs=-1)
        m.fit(tr[self.oz], tr[self.hedef])
        self.model = m
        return m.predict(te[self.oz])

    def ozellik_onemi(self):
        return dict(sorted(zip(self.oz, self.model.feature_importances_),
                           key=lambda x: -x[1]))


class LightGBM:
    ad = "LightGBM"

    def __init__(self, ozellikler, hedef):
        self.oz, self.hedef = ozellikler, hedef

    def egit_tahmin(self, tr, te):
        m = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                          num_leaves=31, random_state=TOHUM,
                          n_jobs=-1, verbose=-1)
        m.fit(tr[self.oz], tr[self.hedef])
        self.model = m
        return m.predict(te[self.oz])

    def ozellik_onemi(self):
        return dict(sorted(zip(self.oz, self.model.feature_importances_),
                           key=lambda x: -x[1]))


class CatBoost:
    ad = "CatBoost"

    def __init__(self, ozellikler, hedef):
        self.oz, self.hedef = ozellikler, hedef

    def egit_tahmin(self, tr, te):
        m = CatBoostRegressor(iterations=400, learning_rate=0.05,
                              depth=6, random_state=TOHUM, verbose=0)
        m.fit(tr[self.oz], tr[self.hedef])
        self.model = m
        return m.predict(te[self.oz])

    def ozellik_onemi(self):
        return dict(sorted(zip(self.oz, self.model.feature_importances_),
                           key=lambda x: -x[1]))







class LSTM:
    ad = "LSTM"

    def __init__(self, ozellikler, hedef, pencere):
        self.hedef, self.pencere = hedef, pencere   # ozellikler kullanilmaz

    def egit_tahmin(self, tr, te):
        import tensorflow as tf
        tf.random.set_seed(TOHUM)
        np.random.seed(TOHUM)

        seri = np.concatenate([tr[self.hedef].values, te[self.hedef].values])
        seri = seri.astype("float32").reshape(-1, 1)

        # olcekleyici SADECE egitim verisiyle fit edilir - sizinti olmasin
        sc = StandardScaler().fit(seri[:len(tr)])
        v = sc.transform(seri).ravel()

        X, Y = [], []
        for i in range(self.pencere, len(v)):
            X.append(v[i - self.pencere:i])
            Y.append(v[i])
        X = np.array(X)[..., None]
        Y = np.array(Y)

        n_te = len(te)
        Xtr, Ytr, Xte = X[:-n_te], Y[:-n_te], X[-n_te:]

        m = tf.keras.Sequential([
            tf.keras.layers.Input((self.pencere, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        m.fit(Xtr, Ytr, epochs=20, batch_size=64, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  patience=4, restore_best_weights=True)])

        p = m.predict(Xte, verbose=0)
        return sc.inverse_transform(p).ravel()


def model_olustur(ad, ozellikler, hedef, pencere=None):
    """Tek bir modeli adiyla olusturur - arayuz bunu kullanir."""
    if ad == "XGBoost":
        return XGB(ozellikler, hedef)
    if ad == "MLP":
        return MLP(ozellikler, hedef)
    if ad == "ExtraTrees":
        return ExtraTrees(ozellikler, hedef)
    if ad == "LightGBM":
        return LightGBM(ozellikler, hedef)
    if ad == "CatBoost":
        return CatBoost(ozellikler, hedef)
    if ad == "LSTM":
        return LSTM(ozellikler, hedef, pencere)
    raise ValueError(f"bilinmeyen model: {ad}")


def modelleri_kur(ozellikler, hedef, pencere=None, adlar=None):
    adlar = adlar or AKTIF_MODELLER
    return [model_olustur(a, ozellikler, hedef, pencere) for a in adlar]