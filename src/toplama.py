"""
Disaridan yuklenen veriyi okur, periyodunu tespit eder ve
istenen periyoda toplar.

Toplama kurali:
  - sayimlar (cagri, karsilanan, abandone...) TOPLANIR
  - konusma suresi AGIRLIKLI ortalama alinir:
        toplam_konusma_sn / karsilanan
    Duz ortalama yanlis olur - 2 cagrilik periyot ile 80 cagrilik
    periyot esit agirlik alamaz.
"""

import numpy as np
import pandas as pd

ZAMAN_SUTUNU = "periyot"

# arayuzde secilebilecek periyotlar: (etiket, pandas kurali, dakika)
PERIYOT_SECENEK = [
    ("30 dakika", "30min", 30),
    ("1 saat", "1h", 60),
    ("2 saat", "2h", 120),
    ("3 saat", "3h", 180),
    ("4 saat", "4h", 240),
    ("6 saat", "6h", 360),
    ("12 saat", "12h", 720),
]

# toplama sirasinda toplanacak sutunlar (dosyada varsa)
TOPLANACAK = ["cagri_sayisi", "karsilanan", "abandone",
              "ivr_de_cozulen", "toplam_konusma_sn"]

ZORUNLU = ["cagri_sayisi"]


def dosya_oku(kaynak):
    """CSV veya Excel okur, zaman sutununu indekse alir."""
    ad = getattr(kaynak, "name", str(kaynak)).lower()
    if ad.endswith((".xlsx", ".xls")):
        d = pd.read_excel(kaynak)
    else:
        d = pd.read_csv(kaynak, encoding="utf-8-sig")

    # zaman sutununu bul
    if ZAMAN_SUTUNU in d.columns:
        zaman = ZAMAN_SUTUNU
    else:
        aday = [c for c in d.columns
                if any(k in c.lower() for k in ("periyot", "tarih", "date", "time"))]
        if not aday:
            raise ValueError(
                "Zaman sutunu bulunamadi. Dosyada 'periyot' adinda bir "
                "sutun olmali (ya da tarih/date/time iceren bir ad).")
        zaman = aday[0]

    d[zaman] = pd.to_datetime(d[zaman])
    d = d.rename(columns={zaman: ZAMAN_SUTUNU})
    d = d.set_index(ZAMAN_SUTUNU).sort_index()

    eksik = [c for c in ZORUNLU if c not in d.columns]
    if eksik:
        raise ValueError(f"Zorunlu sutun eksik: {', '.join(eksik)}")

    return d


def periyot_tespit(d):
    """Verinin kendi periyodunu dakika cinsinden dondurur."""
    if len(d) < 3:
        raise ValueError("Veri cok kisa, periyot tespit edilemiyor.")
    fark = pd.Series(d.index).diff().dt.total_seconds().dropna()
    return int(fark.mode().iloc[0] / 60)


def uygun_periyotlar(taban_dk):
    """Taban periyottan turetilebilecek secenekler:
    taban veya ustu, ve tam kati olanlar."""
    return [(et, kural, dk) for et, kural, dk in PERIYOT_SECENEK
            if dk >= taban_dk and dk % taban_dk == 0]


def topla(d, kural, gun_basi_saat=None):
    """Veriyi verilen periyoda toplar."""
    if gun_basi_saat is None:
        gun_basi_saat = int(d.index.hour.min())
    r = d.resample(kural, offset=f"{gun_basi_saat}h")

    parcalar = {}
    for c in TOPLANACAK:
        if c in d.columns:
            parcalar[c] = r[c].sum()
    ts = pd.DataFrame(parcalar)
    ts["_n"] = r[ZORUNLU[0]].count()
    ts = ts[ts["_n"] > 0].drop(columns="_n")

    # agirlikli ortalama konusma suresi
    if {"toplam_konusma_sn", "karsilanan"}.issubset(ts.columns):
        ts["ort_konusma_sn"] = (ts.toplam_konusma_sn / ts.karsilanan).round(1)
        ts.loc[ts.karsilanan == 0, "ort_konusma_sn"] = np.nan
    elif "ort_konusma_sn" in d.columns:
        # agirlik bilgisi yoksa duz ortalamaya mecburuz
        ts["ort_konusma_sn"] = r["ort_konusma_sn"].mean().reindex(ts.index).round(1)

    if {"abandone", "cagri_sayisi"}.issubset(ts.columns):
        ts["abandone_orani"] = (ts.abandone / ts.cagri_sayisi * 100).round(2)

    if "tatil" in d.columns:
        ts["tatil"] = r["tatil"].max().reindex(ts.index).fillna(0).astype(int)
    else:
        ts["tatil"] = 0

    ts["saat"] = ts.index.hour
    ts["hafta_gunu"] = ts.index.dayofweek
    ts["ay"] = ts.index.month
    ts["yil"] = ts.index.year
    ts.index.name = ZAMAN_SUTUNU
    return ts


def gun_basina_adim(ts):
    """Bir gune kac periyot dusuyor - lag hesabi icin."""
    gunluk = ts.groupby(ts.index.normalize()).size()
    return int(gunluk.mode().iloc[0])


def hafta_gun_sayisi(ts):
    """Veride haftanin kac gunu var. Hafta ici veride 5,
    hafta sonu da varsa 7. Haftalik lag bunu kullanir."""
    return int(pd.Series(ts.index.dayofweek).nunique())


def tarih_araligi(d):
    return d.index.min().date(), d.index.max().date()


def tarih_filtrele(d, bas, son):
    return d.loc[str(bas):str(son)]


def hedef_sutunlar(ts):
    """Modellenebilir hedefler."""
    aday = [("cagri_sayisi", "Cagri sayisi"),
            ("ort_konusma_sn", "Ortalama konusma suresi (sn)"),
            ("karsilanan", "Karsilanan cagri"),
            ("abandone", "Abandone cagri")]
    return {k: v for k, v in aday if k in ts.columns and ts[k].notna().any()}


def ozellik_ekle(ts, hedef, gun_adim, is_gunu_hafta=None):
    """Takvim + lag + hareketli ortalama ozellikleri."""
    if is_gunu_hafta is None:
        is_gunu_hafta = hafta_gun_sayisi(ts)
    s = ts.dropna(subset=[hedef]).copy() if ts[hedef].isna().any() else ts.copy()
    hafta = gun_adim * is_gunu_hafta

    laglar = sorted({1, 2, 3, gun_adim, hafta})
    for l in laglar:
        s[f"lag_{l}"] = s[hedef].shift(l)
    s["roll_gun"] = s[hedef].shift(1).rolling(max(gun_adim, 3)).mean()
    s["roll_hafta"] = s[hedef].shift(1).rolling(hafta).mean()
    s = s.dropna()

    ozellikler = ["saat", "hafta_gunu", "ay", "tatil", "roll_gun", "roll_hafta"]
    ozellikler += [f"lag_{l}" for l in laglar]
    ozellikler = [o for o in ozellikler if o in s.columns]
    return s, ozellikler


def egitim_test_ayir(s, gun_adim, n_hafta=1, is_gunu_hafta=None):
    if is_gunu_hafta is None:
        is_gunu_hafta = hafta_gun_sayisi(s)
    n_test = gun_adim * is_gunu_hafta * n_hafta
    if n_test >= len(s):
        raise ValueError("Test kumesi veriden buyuk.")
    return s.iloc[:-n_test], s.iloc[-n_test:]