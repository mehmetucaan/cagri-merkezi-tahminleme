"""
Erlang C - personel ihtiyaci hesabi

A. K. Erlang'in 1917'de telefon santralleri icin gelistirdigi kuyruk
modeli. Bugun butun buyuk is gucu yonetimi (WFM) yazilimlarinin
temelinde bu model var.

Zincir:
    cagri sayisi x ort. konusma suresi -> is yuku (Erlang)
    is yuku + hedef servis seviyesi    -> ham temsilci sayisi
    ham temsilci / (1 - shrinkage)     -> gercek kadro (FTE)

Varsayimlar:
  1. Cagrilar Poisson surecine gore, birbirinden bagimsiz geliyor
  2. Hizmet sureleri ustel dagiliyor
  3. Bekleyen hic kimse telefonu kapatmiyor  <-- bizim veride YANLIS
  4. Periyot boyunca temsilci sayisi sabit

3. varsayim yuzunden Erlang C, terk eden cagrilari hala kuyrukta
sayar ve gercekte gerekenden FAZLA personel onerir. Yani muhafazakar
(guvenli) yonde yanli. Terki de modelleyen Erlang A (Palm/Garnett)
modeli var ama karmasikligi cok daha yuksek.
"""

import math

import numpy as np
import pandas as pd

# Sektor varsayilanlari
SL_HEDEF = 0.80        # cagrilarin %80'i
HEDEF_SN = 20          # 20 saniye icinde karsilansin
SHRINKAGE = 0.30       # mola, egitim, izin, devamsizlik payi (%25-35 tipik)
MAKS_DOLULUK = 0.85    # temsilci basina maksimum mesguliyet


def erlang_c(N, A):
    """N temsilci ve A Erlang is yuku icin bekleme olasiligi."""
    if N <= A:
        return 1.0
    top = (A ** N / math.factorial(N)) * (N / (N - A))
    alt = sum(A ** k / math.factorial(k) for k in range(N)) + top
    return top / alt


def servis_seviyesi(N, A, aht_sn, hedef_sn=HEDEF_SN):
    """hedef_sn icinde karsilanan cagri orani."""
    if N <= A:
        return 0.0
    return 1 - erlang_c(N, A) * math.exp(-(N - A) * hedef_sn / aht_sn)


def ortalama_bekleme(N, A, aht_sn):
    """Butun cagrilar uzerinden ortalama bekleme suresi (saniye)."""
    if N <= A:
        return float("inf")
    return erlang_c(N, A) * aht_sn / (N - A)


def is_yuku(cagri, aht_sn, periyot_sn):
    """Erlang cinsinden trafik yogunlugu."""
    return cagri * aht_sn / periyot_sn


def gereken_temsilci(cagri, aht_sn, periyot_sn,
                     sl_hedef=SL_HEDEF, hedef_sn=HEDEF_SN,
                     maks_doluluk=MAKS_DOLULUK, maks_temsilci=500):
    """Hedef servis seviyesini tutan minimum temsilci sayisi (ham).

    Doluluk siniri da uygulanir: temsilciler surekli %85'in uzerinde
    mesgulse tukenme ve kalite kaybi olur, sektor bunu sinirlar.
    """
    if cagri <= 0 or not aht_sn or aht_sn <= 0 or np.isnan(aht_sn):
        return 0

    A = is_yuku(cagri, aht_sn, periyot_sn)
    N = max(1, math.ceil(A))
    while N < maks_temsilci:
        if servis_seviyesi(N, A, aht_sn, hedef_sn) >= sl_hedef:
            break
        N += 1

    # doluluk siniri
    if maks_doluluk:
        while N < maks_temsilci and A / N > maks_doluluk:
            N += 1
    return N


def kadro(ham_temsilci, shrinkage=SHRINKAGE):
    """Ham Erlang sonucunu gercek kadroya cevirir.

    Ham sayi 'ayni anda telefonda kac kisi lazim' demek. Ama
    temsilciler molada, egitimde, izinde de oluyor - shrinkage bu
    kayip zamani telafi eder. Sektor ortalamasi %25-35."""
    if shrinkage >= 1:
        raise ValueError("shrinkage 1'den kucuk olmali")
    return math.ceil(ham_temsilci / (1 - shrinkage))


def tablo_uret(df, cagri_sutun, aht_sutun, periyot_sn,
               sl_hedef=SL_HEDEF, hedef_sn=HEDEF_SN,
               shrinkage=SHRINKAGE, maks_doluluk=MAKS_DOLULUK):
    """Bir tahmin tablosundan personel ihtiyaci tablosu uretir."""
    out = pd.DataFrame(index=df.index)
    out["cagri"] = df[cagri_sutun].round(0)
    out["aht_sn"] = df[aht_sutun].round(1)
    out["is_yuku_erlang"] = [
        round(is_yuku(c, a, periyot_sn), 2)
        for c, a in zip(out.cagri, out.aht_sn)]
    out["ham_temsilci"] = [
        gereken_temsilci(c, a, periyot_sn, sl_hedef, hedef_sn, maks_doluluk)
        for c, a in zip(out.cagri, out.aht_sn)]
    out["kadro_fte"] = [kadro(n, shrinkage) for n in out.ham_temsilci]
    out["doluluk"] = (out.is_yuku_erlang / out.ham_temsilci * 100).round(1)
    out["servis_seviyesi"] = [
        round(servis_seviyesi(n, a_, aht, hedef_sn) * 100, 1)
        if n > 0 else 0.0
        for n, a_, aht in zip(out.ham_temsilci, out.is_yuku_erlang, out.aht_sn)]
    return out


def kadro_hatasi(gercek_tablo, tahmin_tablo):
    """Tahmine dayali kadro ile gercege dayali kadroyu karsilastirir."""
    k = pd.DataFrame({
        "N_gercek": gercek_tablo.ham_temsilci,
        "N_tahmin": tahmin_tablo.ham_temsilci,
        "FTE_gercek": gercek_tablo.kadro_fte,
        "FTE_tahmin": tahmin_tablo.kadro_fte,
    })
    k["fark"] = k.N_tahmin - k.N_gercek
    k["fte_fark"] = k.FTE_tahmin - k.FTE_gercek
    return k


def ozet(k):
    return {
        "ort_mutlak_hata_kisi": round(k.fark.abs().mean(), 2),
        "en_kotu_sapma": int(k.fark.abs().max()),
        "tam_isabet_yuzde": round((k.fark == 0).mean() * 100, 1),
        "eksik_personel_yuzde": round((k.fark < 0).mean() * 100, 1),
        "fazla_personel_yuzde": round((k.fark > 0).mean() * 100, 1),
    }