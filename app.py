"""
Çağrı Merkezi Tahminleme Arayüzü

Veri dışarıdan yüklenir, periyot arayüzden seçilir.

Çalıştırmak için:
    streamlit run app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import erlang
import metrikler
import modeller
import toplama

KURAL_SANIYE = {"30min": 1800, "1h": 3600, "2h": 7200, "3h": 10800,
                "4h": 14400, "6h": 21600, "12h": 43200}

st.set_page_config(page_title="Çağrı Merkezi Tahminleme",
                   page_icon="📞", layout="wide")


# ---------- önbelleğe alınan işlemler ----------
@st.cache_data(show_spinner=False)
def _oku(icerik, ad):
    import io
    return toplama.dosya_oku(io.BytesIO(icerik) if not ad.endswith(
        (".xlsx", ".xls")) else io.BytesIO(icerik))


@st.cache_data(show_spinner=False)
def _topla(icerik, ad, kural, bas, son):
    d = _oku(icerik, ad)
    d = toplama.tarih_filtrele(d, bas, son)
    return toplama.topla(d, kural)


@st.cache_data(show_spinner=False)
def _egit(icerik, ad, kural, bas, son, hedef, n_hafta, model_adi):
    ts = _topla(icerik, ad, kural, bas, son)
    gun = toplama.gun_basina_adim(ts)
    s, ozellikler = toplama.ozellik_ekle(ts, hedef, gun)
    tr, te = toplama.egitim_test_ayir(s, gun, n_hafta)

    m = modeller.model_olustur(model_adi, ozellikler, hedef)
    t0 = time.time()
    tahmin = m.egit_tahmin(tr, te)
    sure = time.time() - t0
    onem = m.ozellik_onemi() if hasattr(m, "ozellik_onemi") else None
    return pd.Series(tahmin, index=te.index), sure, onem


@st.cache_data(show_spinner=False)
def _izgara_arama(icerik, ad, kural, bas, son, hedef, n_hafta):
    """XGBoost için ızgara arama. Zaman serisi olduğu için rastgele
    çapraz doğrulama değil TimeSeriesSplit kullanılır - gelecekten
    geçmişe bilgi sızmasın."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from xgboost import XGBRegressor

    ts = _topla(icerik, ad, kural, bas, son)
    gun = toplama.gun_basina_adim(ts)
    s, ozellikler = toplama.ozellik_ekle(ts, hedef, gun)
    tr, te = toplama.egitim_test_ayir(s, gun, n_hafta)

    izgara = {"max_depth": [4, 6, 8],
              "n_estimators": [200, 400],
              "learning_rate": [0.03, 0.05, 0.1]}

    t0 = time.time()
    gs = GridSearchCV(
        XGBRegressor(subsample=0.8, random_state=modeller.TOHUM, n_jobs=1),
        izgara, cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(tr[ozellikler], tr[hedef])
    sure = time.time() - t0

    tahmin = gs.best_estimator_.predict(te[ozellikler])
    return (pd.Series(tahmin, index=te.index), gs.best_params_, sure,
            len(gs.cv_results_["params"]))


# ================= KENAR ÇUBUĞU =================
st.sidebar.title("Ayarlar")

yuklenen = st.sidebar.file_uploader(
    "Veri dosyası (CSV / Excel)", type=["csv", "xlsx", "xls"])

if yuklenen is None:
    st.title("Çağrı Merkezi Tahminleme")
    st.info("Başlamak için sol taraftan bir veri dosyası yükleyin.")
    st.markdown("""
**Dosyada olması gerekenler**

| Sütun | Zorunlu | Açıklama |
|---|---|---|
| `periyot` | evet | Zaman damgası (tarih + saat) |
| `cagri_sayisi` | evet | Periyottaki gelen çağrı sayısı |
| `karsilanan` | hayır | Karşılanan çağrı - ağırlıklı ortalama için |
| `toplam_konusma_sn` | hayır | Toplam konuşma süresi - ağırlıklı ortalama için |
| `abandone` | hayır | Terk edilen çağrı |
| `tatil` | hayır | Tatil işareti (0/1) |

Veri en ince periyotta yüklenmeli (örn. 30 dakika veya 1 saat).
Daha geniş periyotlar arayüzden türetilir.

**Önemli:** `karsilanan` ve `toplam_konusma_sn` sütunları varsa ortalama
konuşma süresi **ağırlıklı** hesaplanır. Yoksa düz ortalama alınır ve
sonuç yanlı olur.
""")
    st.stop()

icerik = yuklenen.getvalue()

try:
    ham = _oku(icerik, yuklenen.name)
    taban_dk = toplama.periyot_tespit(ham)
except Exception as e:
    st.sidebar.error(str(e))
    st.title("Çağrı Merkezi Tahminleme")
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

secenekler = toplama.uygun_periyotlar(taban_dk)
etiketler = [e for e, _, _ in secenekler]

st.sidebar.success(f"{len(ham):,} satır · taban periyot {taban_dk} dakika")

secilen_et = st.sidebar.selectbox("Periyot", etiketler)
kural = dict((e, k) for e, k, _ in secenekler)[secilen_et]

# --- tarih aralığı ---
v_bas, v_son = toplama.tarih_araligi(ham)
aralik = st.sidebar.date_input("Tarih aralığı", value=(v_bas, v_son),
                               min_value=v_bas, max_value=v_son)
if isinstance(aralik, (list, tuple)) and len(aralik) == 2:
    bas, son = aralik
else:
    bas, son = v_bas, v_son

ts = _topla(icerik, yuklenen.name, kural, bas, son)
if len(ts) < 50:
    st.sidebar.error("Seçilen tarih aralığında çok az veri var.")
    st.stop()
gun_adim = toplama.gun_basina_adim(ts)
hedefler = toplama.hedef_sutunlar(ts)

hedef = st.sidebar.selectbox("Hedef değişken", list(hedefler),
                             format_func=hedefler.get)
secili_modeller = st.sidebar.multiselect(
    "Modeller", ["XGBoost", "MLP", "ExtraTrees", "LightGBM", "CatBoost"],
    default=["XGBoost", "MLP"])

hafta_gun = toplama.hafta_gun_sayisi(ts)
n_hafta = st.sidebar.slider("Test kümesi (hafta)", 1, 8, 1)

st.sidebar.divider()
st.sidebar.caption(f"Periyot: {secilen_et}\n\n"
                   f"Gün başına {gun_adim} periyot · haftada {hafta_gun} gün\n\n"
                   f"Test: son {n_hafta} hafta "
                   f"({gun_adim * hafta_gun * n_hafta} satır)")


# ================= BAŞLIK =================
st.title("Çağrı Merkezi Tahminleme")
st.caption(f"{yuklenen.name} · {ts.index.min().date()} - {ts.index.max().date()}")

if {"toplam_konusma_sn", "karsilanan"}.issubset(ham.columns):
    pass
elif "ort_konusma_sn" in ham.columns:
    st.warning("Dosyada `karsilanan` ve `toplam_konusma_sn` sütunları yok. "
               "Konuşma süresi **düz ortalama** ile toplandı - az çağrılı "
               "periyotlar fazla ağırlık alıyor, sonuç yanlı olabilir.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Toplam periyot", f"{len(ts):,}")
k2.metric("Ortalama çağrı", f"{ts.cagri_sayisi.mean():.1f}")
if "ort_konusma_sn" in ts.columns:
    k3.metric("Ort. konuşma süresi", f"{ts.ort_konusma_sn.mean():.0f} sn")
k4.metric("Gün sayısı", f"{ts.index.normalize().nunique():,}")

sekme_veri, sekme_grafik, sekme_tahmin, sekme_erlang = st.tabs(
    ["Veri", "Grafikler", "Tahminleme", "Personel (Erlang C)"])




# ================= SEKME 1: VERİ =================
with sekme_veri:
    st.subheader(f"{secilen_et} periyoda toplanmış veri")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(ts.tail(300), use_container_width=True, height=380)
    with c2:
        st.write("**Özet istatistikler**")
        sayisal = [c for c in ["cagri_sayisi", "karsilanan", "abandone",
                               "ort_konusma_sn"] if c in ts.columns]
        st.dataframe(ts[sayisal].describe().round(1),
                     use_container_width=True)

    bos = ts[hedef].isna().sum()
    if bos:
        st.warning(f"{hedef} sütununda {bos} tanımsız değer var "
                   f"(%{bos / len(ts) * 100:.1f}). Bu satırlar modelleme "
                   "dışında bırakılır.")

    st.download_button("Toplanmış veriyi indir",
                       ts.to_csv().encode("utf-8"),
                       f"veri_{kural}.csv", "text/csv")


# ================= SEKME 2: GRAFİKLER =================
with sekme_grafik:
    st.subheader(hedefler[hedef])

    # DIKKAT: burada "son" adini kullanma - tarih araliginin bitisi o.
    seri = ts[hedef].tail(gun_adim * hafta_gun * 4).copy()
    if gun_adim > 1:
        tam = pd.date_range(seri.index.min(), seri.index.max(), freq=kural)
        seri = seri.reindex(tam)

    fig = px.line(seri, labels={"value": hedefler[hedef], "index": "Tarih"},
                  title="Son 4 hafta")
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        if gun_adim == 1:
            st.info("Bu periyotta günde tek değer var, "
                    "gün içi profil oluşmuyor.")
        else:
            saat = ts.groupby("saat")[hedef].mean().reset_index()
            saat["saat_et"] = saat.saat.astype(str) + ":00"
            fig = px.bar(saat, x="saat_et", y=hedef, title="Gün içi profil",
                         labels={"saat_et": "Saat", hedef: hedefler[hedef]})
            fig.update_layout(height=320, xaxis_type="category")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        adlar = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
        hg = ts.groupby("hafta_gunu")[hedef].mean().reset_index()
        hg["gun"] = [adlar[i] for i in hg.hafta_gunu]
        # farklar kucuk - eksen sifirdan baslarsa duz gorunur
        fig = px.line(hg, x="gun", y=hedef, markers=True,
                      title="Hafta günü ortalaması",
                      labels={"gun": "Gün", hedef: hedefler[hedef]})
        fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
        fig.update_layout(height=320,
                          yaxis_range=[hg[hedef].min() * 0.95,
                                       hg[hedef].max() * 1.03])
        st.plotly_chart(fig, use_container_width=True)


# ================= SEKME 3: TAHMİNLEME =================
with sekme_tahmin:
    if not secili_modeller:
        st.warning("Kenar çubuğundan en az bir model seçin.")
        st.stop()

    try:
        s, ozellikler = toplama.ozellik_ekle(ts, hedef, gun_adim)
        tr, te = toplama.egitim_test_ayir(s, gun_adim, n_hafta)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"Eğitim {len(tr):,} satır · Test {len(te)} satır "
               f"({te.index.min().date()} - {te.index.max().date()}) · "
               f"{len(ozellikler)} özellik")

    if len(te) < 10:
        st.warning(f"Test kümesi sadece {len(te)} satır. Bu boyutta model "
                   "karşılaştırması güvenilir değil.")

    gercek = te[hedef].values
    satirlar = [{"model": "Baseline", "sure_sn": 0.0,
                 **metrikler.hesapla(gercek, [tr[hedef].mean()] * len(te))}]
    cizim = pd.DataFrame({"Gerçek": te[hedef]})
    onemler = {}

    ilerleme = st.progress(0.0, "Modeller eğitiliyor...")
    for i, ad in enumerate(secili_modeller, 1):
        tahmin, sure, onem = _egit(icerik, yuklenen.name, kural, bas, son,
                                   hedef, n_hafta, ad)
        satirlar.append({"model": ad, "sure_sn": round(sure, 2),
                         **metrikler.hesapla(gercek, tahmin.values)})
        cizim[ad] = tahmin
        if onem:
            onemler[ad] = onem
        ilerleme.progress(i / len(secili_modeller))
    ilerleme.empty()

    tablo = pd.DataFrame(satirlar)[["model", "MAE", "MAPE", "sure_sn"]]

    st.subheader("Hata metrikleri")
    st.dataframe(tablo.style.highlight_min(subset=["MAE", "MAPE"],
                                           color="#1b5e20"),
                 use_container_width=True, hide_index=True)

    st.subheader("Gerçek vs tahmin")
    fig = go.Figure()
    fig.add_scatter(x=cizim.index, y=cizim["Gerçek"], name="Gerçek",
                    line=dict(color="white", width=3))
    for ad in secili_modeller:
        fig.add_scatter(x=cizim.index, y=cizim[ad], name=ad,
                        line=dict(width=1.8))
    fig.update_layout(height=420, yaxis_title=hedefler[hedef],
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if onemler:
        st.subheader("Özellik önemi (XGBoost)")
        for ad, onem in onemler.items():
            o = pd.Series(onem).head(10).sort_values()
            fig = px.bar(x=o.values, y=o.index, orientation="h",
                         labels={"x": "Önem", "y": ""})
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    with st.expander("Izgara arama (XGBoost hiperparametre optimizasyonu)"):
        st.caption("max_depth × n_estimators × learning_rate = 18 kombinasyon, "
                   "TimeSeriesSplit ile 3 katlı çapraz doğrulama. "
                   "Veri boyutuna göre 20-40 saniye sürer.")
        if st.button("Aramayı başlat"):
            with st.spinner("Izgara araması çalışıyor..."):
                gs_tahmin, en_iyi, gs_sure, n_komb = _izgara_arama(
                    icerik, yuklenen.name, kural, bas, son, hedef, n_hafta)

            varsayilan = next((r for r in satirlar if r["model"] == "XGBoost"), None)
            gs_metrik = metrikler.hesapla(gercek, gs_tahmin.values)

            st.write(f"**En iyi parametreler:** `{en_iyi}`")
            st.caption(f"{n_komb} kombinasyon · {gs_sure:.1f} saniye")

            kars = pd.DataFrame([
                {"ayar": "Varsayılan", "MAE": varsayilan["MAE"] if varsayilan else None,
                 "MAPE": varsayilan["MAPE"] if varsayilan else None},
                {"ayar": "Izgara arama", "MAE": gs_metrik["MAE"],
                 "MAPE": gs_metrik["MAPE"]},
            ])
            st.dataframe(kars, use_container_width=True, hide_index=True)

            if varsayilan and gs_metrik["MAPE"] > varsayilan["MAPE"]:
                st.info("Izgara arama sonucu varsayılan ayardan kötü çıktı. "
                        "Bu beklenmedik değil: arama çapraz doğrulama "
                        "katlarında en iyi olanı seçer, bizim testimiz ise "
                        "son hafta. Katlarda iyi olan parametre son haftada "
                        "iyi olmayabilir.")

    st.download_button("Tahminleri indir", cizim.to_csv().encode("utf-8"),
                       f"tahmin_{kural}_{hedef}.csv", "text/csv")


# ================= SEKME 4: PERSONEL (ERLANG C) =================
with sekme_erlang:
    st.subheader("Personel ihtiyacı - Erlang C")

    if "ort_konusma_sn" not in ts.columns:
        st.warning("Bu hesap için konuşma süresi verisi gerekli, "
                   "yüklenen dosyada yok.")
        st.stop()

    with st.expander("Ayarlar", expanded=True):
        e1, e2, e3, e4 = st.columns(4)
        sl_hedef = e1.slider("Hedef servis seviyesi (%)", 50, 95, 80) / 100
        hedef_sn = e2.slider("Hedef süre (sn)", 10, 60, 20)
        shrinkage = e3.slider("Shrinkage (%)", 10, 45, 30) / 100
        maks_doluluk = e4.slider("Maks. doluluk (%)", 70, 95, 85) / 100

    periyot_sn = KURAL_SANIYE.get(kural)
    if periyot_sn is None:
        st.error("Bu periyot için saniye karşılığı tanımsız.")
        st.stop()

    # gercek degerler - Erlang icin her iki sutun da gerekli
    gercek_df = te[["cagri_sayisi", "ort_konusma_sn"]].rename(
        columns={"cagri_sayisi": "c", "ort_konusma_sn": "a"})
    tg = erlang.tablo_uret(gercek_df, "c", "a", periyot_sn,
                           sl_hedef, hedef_sn, shrinkage, maks_doluluk)

    # her iki hedefi de AYNI model ile tahmin ediyoruz - iki ayri
    # egitim, ama ikisi de ayni algoritmayi kullanir
    model_secim = st.selectbox("Personel hesabında kullanılacak model",
                               secili_modeller)

    cagri_tahmin, _, _ = _egit(icerik, yuklenen.name, kural, bas, son,
                               "cagri_sayisi", n_hafta, model_secim)
    aht_tahmin, _, _ = _egit(icerik, yuklenen.name, kural, bas, son,
                             "ort_konusma_sn", n_hafta, model_secim)

    tahmin_df = pd.DataFrame({"c": cagri_tahmin, "a": aht_tahmin}).dropna()
    tt = erlang.tablo_uret(tahmin_df, "c", "a", periyot_sn,
                           sl_hedef, hedef_sn, shrinkage, maks_doluluk)

    ortak_ix = tg.index.intersection(tt.index)
    tg, tt = tg.loc[ortak_ix], tt.loc[ortak_ix]

    k1, k2, k3 = st.columns(3)
    k1.metric("Ort. gereken temsilci (gerçek)", f"{tg.ham_temsilci.mean():.1f}")
    k2.metric("Ort. kadro - FTE (gerçek)", f"{tg.kadro_fte.mean():.1f}")
    k3.metric("Ort. servis seviyesi", f"%{tg.servis_seviyesi.mean():.1f}")

    hata = erlang.kadro_hatasi(tg, tt)
    oz = erlang.ozet(hata)
    st.write(f"**Tahminin personel hatası:** ortalama {oz['ort_mutlak_hata_kisi']} "
             f"kişi, en kötü sapma {oz['en_kotu_sapma']} kişi, "
             f"tam isabet %{oz['tam_isabet_yuzde']}, "
             f"eksik personel önerme oranı %{oz['eksik_personel_yuzde']}")

    st.subheader("Periyot periyot tablo")
    birlesik = pd.DataFrame({
        "Çağrı (gerçek)": tg.cagri.astype(int),
        "Çağrı (tahmin)": tt.cagri.astype(int),
        "Konuşma süresi sn (gerçek)": tg.aht_sn.round(0),
        "Konuşma süresi sn (tahmin)": tt.aht_sn.round(0),
        "Gereken temsilci (gerçek)": tg.ham_temsilci,
        "Gereken temsilci (tahmin)": tt.ham_temsilci,
        "Kadro - FTE (gerçek)": tg.kadro_fte,
        "Kadro - FTE (tahmin)": tt.kadro_fte,
        "Hizmet seviyesi %% (gerçek)": tg.servis_seviyesi,
        "Hizmet seviyesi %% (tahmin)": tt.servis_seviyesi,
        "Fark (kişi)": hata.fark,
    })
    st.dataframe(
        birlesik.style.background_gradient(subset=["Fark (kişi)"], cmap="RdYlGn_r",
                                           vmin=-8, vmax=8),
        use_container_width=True)

    st.subheader("Gereken temsilci - gerçek vs tahmin")
    fig = go.Figure()
    fig.add_scatter(x=tg.index, y=tg.ham_temsilci, name="Gerçek",
                    line=dict(color="white", width=3))
    fig.add_scatter(x=tt.index, y=tt.ham_temsilci, name=f"Tahmin ({model_secim})",
                    line=dict(color="#1f77b4", width=1.8))
    fig.update_layout(height=380, yaxis_title="Gereken temsilci",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Varsayımlar ve sınırlamalar"):
        st.markdown("""
- Çağrılar Poisson sürecine göre rastgele geliyor, hizmet süreleri üstel
  dağılıyor kabul edilir.
- **Terk edilen çağrılar modellenmez** - Erlang C onları hâlâ kuyrukta
  sayar, bu yüzden gerçekte gerekenden fazla personel önerebilir.
- Konuşma süresi verisi sadece talk time içeriyorsa (hold/wrap yoksa)
  gerçek AHT daha yüksektir - bu durumda personel ihtiyacı **olduğundan
  az** hesaplanır.
- Shrinkage ve maksimum doluluk değerleri sektör ortalamalarıdır,
  gerçek değerler kurumdan kuruma değişir.
""")

    st.download_button("Personel tablosunu indir",
                       birlesik.to_csv().encode("utf-8"),
                       f"personel_{kural}.csv", "text/csv")