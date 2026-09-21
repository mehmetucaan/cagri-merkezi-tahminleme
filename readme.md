# Çağrı Merkezi Tahminleme

Çağrı merkezi verisinden çağrı sayısı ve ortalama görüşme süresini
tahminleyen, tahminleri Erlang C ile personel ihtiyacına çeviren
Streamlit arayüzü.

## Özellikler
- CSV/Excel veri yükleme, periyot otomatik tespiti
- 30 dakika - 12 saat arası esnek periyot türetme (ağırlıklı ortalama ile)
- XGBoost, MLP, ExtraTrees, LightGBM, CatBoost model karşılaştırması
- Erlang C ile personel ihtiyacı hesaplama (shrinkage, doluluk oranı dahil)

## Kurulum
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Veri formatı
Dosyada `periyot` (zaman damgası) ve `cagri_sayisi` sütunları zorunlu.
`karsilanan`, `toplam_konusma_sn` sütunları varsa ortalama görüşme
süresi ağırlıklı hesaplanır.