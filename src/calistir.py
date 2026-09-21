"""
Ana calistirici: dort periyot x iki hedef x uc model.

Kullanim:
    python src/calistir.py                      # hepsi
    python src/calistir.py --periyot 2saat      # tek periyot
    python src/calistir.py --periyot 2saat --hedef cagri_sayisi
"""

import argparse
import time

import pandas as pd

import veri
import metrikler
import modeller

# LSTM pencere boyutu: yaklasik bir is gunu kadar gecmise baksin
PENCERELER = {"30dk": 48, "2saat": 30, "4saat": 21, "12saat": 10}


def tek_kosu(periyot, hedef, kayit=True):
    d = veri.yukle(periyot)
    s, ozellikler = veri.ozellik_ekle(d, periyot, hedef)
    tr, te = veri.egitim_test_ayir(s, periyot)

    print(f"\n{'='*70}")
    print(f"{periyot} | {hedef} | egitim {len(tr)} | test {len(te)} "
          f"({te.index.min().date()} -> {te.index.max().date()})")
    print(f"{'='*70}")

    gercek = te[hedef].values
    satirlar = []
    tahminler = pd.DataFrame({"gercek": gercek}, index=te.index)

    # referans: her zaman egitim ortalamasini tahmin et
    ort = [tr[hedef].mean()] * len(te)
    satirlar.append({"model": "Baseline", "sure_sn": 0.0, **metrikler.hesapla(gercek, ort)})
    tahminler["Baseline"] = ort

    for m in modeller.modelleri_kur(ozellikler, hedef, PENCERELER[periyot]):
        t = time.time()
        p = m.egit_tahmin(tr, te)
        sure = round(time.time() - t, 1)
        satirlar.append({"model": m.ad, "sure_sn": sure,
                         **metrikler.hesapla(gercek, p)})
        tahminler[m.ad] = p
        print(f"  {m.ad:<10} bitti ({sure} sn)")

    tablo = pd.DataFrame(satirlar)[["model", "MAE", "MAPE", "RMSE", "sure_sn", "haric"]]
    print()
    print(tablo.to_string(index=False))

    if kayit:
        tahminler.to_csv(f"sonuclar/tahmin_{periyot}_{hedef}.csv")
        tablo.insert(0, "hedef", hedef)
        tablo.insert(0, "periyot", periyot)

    return tablo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--periyot", default=None, choices=list(veri.PERIYOTLAR))
    ap.add_argument("--hedef", default=None, choices=veri.HEDEFLER)
    a = ap.parse_args()

    periyotlar = [a.periyot] if a.periyot else list(veri.PERIYOTLAR)
    hedefler = [a.hedef] if a.hedef else veri.HEDEFLER

    hepsi = []
    for p in periyotlar:
        for h in hedefler:
            hepsi.append(tek_kosu(p, h))

    ozet = pd.concat(hepsi, ignore_index=True)
    ozet.to_csv("sonuclar/ozet.csv", index=False)

    print(f"\n\n{'#'*70}\nTUM SONUCLAR\n{'#'*70}")
    print(ozet.to_string(index=False))
    print("\nkaydedildi -> sonuclar/ozet.csv")