"""
Sonuc grafikleri. Kayitli tahmin dosyalarindan uretir,
model egitmez.

Cikti: sonuclar/grafik/ klasoru
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import veri

KLASOR = "sonuclar/grafik"
MODELLER = ["XGBoost", "MLP"]
RENK = {"gercek": "black", "XGBoost": "tab:blue",
        "MLP": "tab:orange", "LSTM": "tab:green"}

BASLIK = {"cagri_sayisi": "Cagri sayisi",
          "ort_konusma_sn": "Ortalama konusma suresi (sn)"}

FREKANS = {"30dk": "30min", "2saat": "2h", "4saat": "4h", "12saat": "12h"}


def tahmin_grafigi(periyot, hedef):
    """Test haftasi: gercek deger vs uc modelin tahmini."""
    d = pd.read_csv(f"sonuclar/tahmin_{periyot}_{hedef}.csv",
                    parse_dates=["periyot"]).set_index("periyot")

        # 12 saatlik sette gunde tek nokta var, reindex ardisik noktalari
    # NaN ile ayirip cizgiyi tamamen yok ediyor - o yuzden atlaniyor
    if periyot != "12saat":
        tam = pd.date_range(d.index.min(), d.index.max(), freq=FREKANS[periyot])
        d = d.reindex(tam)

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(d.index, d["gercek"], color=RENK["gercek"],
            lw=2.5, marker="o", ms=4, label="Gercek", zorder=5)
    for m in MODELLER:
        ax.plot(d.index, d[m], color=RENK[m], lw=1.4,
                marker="o", ms=3, alpha=0.85, label=m)

    ax.set_title(f"{BASLIK[hedef]} - {periyot} - test haftasi (24-28 Haziran 2024)")
    ax.set_ylabel(BASLIK[hedef])
    ax.legend(ncol=4)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    yol = f"{KLASOR}/tahmin_{periyot}_{hedef}.png"
    fig.savefig(yol, dpi=120)
    plt.close(fig)
    return yol


def mape_grafigi(ozet):
    """Periyoda gore MAPE - iki hedef yan yana."""
    sira = ["30dk", "2saat", "4saat", "12saat"]
    etiket = ["30 dk", "2 saat", "4 saat", "12 saat"]

    fig, axlar = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, hedef in zip(axlar, veri.HEDEFLER):
        alt = ozet[ozet.hedef == hedef]
        for m in ["Baseline"] + MODELLER:
            y = [alt[(alt.periyot == p) & (alt.model == m)]["MAPE"].iloc[0]
                 for p in sira]
            stil = "--" if m == "Baseline" else "-"
            ax.plot(etiket, y, stil, marker="o",
                    color=RENK.get(m, "gray"), label=m)
        ax.set_title(BASLIK[hedef])
        ax.set_xlabel("Periyot")
        ax.set_ylabel("MAPE (%)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Periyoda gore tahmin hatasi")
    fig.tight_layout()
    yol = f"{KLASOR}/mape_periyot.png"
    fig.savefig(yol, dpi=120)
    plt.close(fig)
    return yol


if __name__ == "__main__":
    os.makedirs(KLASOR, exist_ok=True)

    ozet = pd.read_csv("sonuclar/ozet.csv")

    for p in veri.PERIYOTLAR:
        for h in veri.HEDEFLER:
            print("kaydedildi ->", tahmin_grafigi(p, h))

    print("kaydedildi ->", mape_grafigi(ozet))