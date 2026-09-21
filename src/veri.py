"""
Veri yukleme, ozellik uretimi ve egitim/test ayrimi.
Test kumesi: veri setinin SON HAFTASI (5 is gunu).
"""

import pandas as pd

# periyot adi -> (dosya adi, bir gundeki periyot sayisi)
PERIYOTLAR = {
    "30dk":   ("halifax_hi_30dk.csv",   24),
    "2saat":  ("halifax_hi_2saat.csv",   6),
    "4saat":  ("halifax_hi_4saat.csv",   3),
    "12saat": ("halifax_hi_12saat.csv",  1),
}

HEDEFLER = ["cagri_sayisi", "ort_konusma_sn"]

def yukle(periyot):
    dosya, _ = PERIYOTLAR[periyot]
    d = pd.read_csv(f"data/processed/{dosya}", parse_dates=["periyot"])
    return d.set_index("periyot").sort_index()

def ozellik_ekle(d, periyot, hedef):
    """Takvim + lag + hareketli ortalama ozellikleri uretir."""
    _, gun = PERIYOTLAR[periyot]
    hafta = gun * 5          # hafta ici veri: 1 hafta = 5 is gunu

    s = d.copy()

    # konusma suresi hedefinde tanimsiz satirlar cikarilir
    # (o periyotta hic cagri karsilanmamis -> ortalama yok)
    if hedef == "ort_konusma_sn":
        s = s.dropna(subset=[hedef])

    # gecmis degerler: 1-2-3 periyot once, 1 gun once, 1 hafta once
    laglar = sorted({1, 2, 3, gun, hafta})
    for l in laglar:
        s[f"lag_{l}"] = s[hedef].shift(l)

         # hareketli ortalamalar - shift(1) sart, yoksa gelecege bakariz
    s["roll_gun"] = s[hedef].shift(1).rolling(max(gun, 3)).mean()
    s["roll_hafta"] = s[hedef].shift(1).rolling(hafta).mean()

    s = s.dropna()

    ozellikler = (["saat", "hafta_gunu", "ay", "tatil", "roll_gun", "roll_hafta"]
                  + [f"lag_{l}" for l in laglar])
    return s, ozellikler

def egitim_test_ayir(s, periyot):
    """Son 1 hafta test, oncesi egitim."""
    _, gun = PERIYOTLAR[periyot]
    n_test = gun * 5
    return s.iloc[:-n_test], s.iloc[-n_test:]


if __name__ == "__main__":
    for p in PERIYOTLAR:
        for h in HEDEFLER:
            d = yukle(p)
            s, oz = ozellik_ekle(d, p, h)
            tr, te = egitim_test_ayir(s, p)
            bas = te.index.min().date()
            son = te.index.max().date()
            print(f"{p:>7} | {h:<15} egitim {len(tr):>6} | test {len(te):>4} | ozellik {len(oz):>2} | test {bas} -> {son}")