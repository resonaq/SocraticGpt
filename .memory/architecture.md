# Mimari ve Teknik Kararlar

## Iki Seviyeli Deney Tasarimi

### Seviye 1: Karakter (microgpt.py)
- 4,224 parametre, tek transformer katmani
- 28 karakterlik vocab, 16 karakter context window
- 9 terapotik cumle uzerinde egitilir
- **Sonuc:** CGI metrikleri calisiyor AMA nadirllik (rarity) ile karisik
- WhatIf > Why > When siralama tutarli

### Seviye 2: Semantik (cgi_semantic.py)
- distilgpt2 (82M param), pretrained
- Ayni 3 soru, ayni baglam, buyuk vocab
- Nadirllik kontrolu eklendi: `KL_normalized = KL_raw * familiarity`
- **Sonuc:** Nadirllik kontrol edilince bile WhatIf en yuksek

### Seviye 3: Konusma Analizi (socratic_engine.py)
- cgi_semantic.py'den tum metrikleri import eder
- Herhangi bir konusmayi parse eder, her soruyu analiz eder
- Model bir kez yuklenir, tum konusmalar icin kullanilir

## CGI Metrikleri

```
P_before = model(base_context)           # sorudan once dagilim
P_after  = model(base_context + soru)    # sorudan sonra dagilim

KL_raw       = KL(P_before || P_after)   # ham bilgi kaymasi
JSD          = Jensen-Shannon divergence  # simetrik mesafe
Spearman_rho = siralama korelasyonu       # 1.0 = ayni sira, 0 = farkli
TopK_Jaccard = ilk K tokenin ortakligi    # 1.0 = ayni set
Delta_H      = entropi degisimi           # belirsizlik artisi/azalisi
Delta_Attn_H = attention entropi degisimi
Familiarity  = corpus n-gram yakinligi    # nadirllik kontrolu
Delta_Div    = continuation cesitliligi   # uretim farklilasmasi
```

## Composite Score Formulu

```
composite = (1 - rho) * (1 - Jaccard) * KL_normalized * max(0.01, 1 + DeltaDiversity)
```

- `(1 - rho)`: Siralama ne kadar degisti
- `(1 - Jaccard)`: Top-K tokenlarin ne kadari farkli
- `KL_normalized`: Nadirllik kontrollü bilgi kaymasi
- `max(0.01, 1 + DeltaDiversity)`: Uretim cesitliligi artisi

## Esik Degerleri

```
TRANSFORMATIVE >= 0.001
UNCERTAIN      >= 0.0001
MECHANICAL     <  0.0001
```

**NOT:** Bu esikler ilk 3-soru deneyinden turetildi. 10-ornek testinde
TUM sorular > 0.001 cikti, esikler kalibre edilmeli.

## Konusma Formati (JSON)

```json
{
  "id": "conversation_id",
  "description": "aciklama",
  "turns": [
    {"role": "human", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Bilinen Teknik Sorunlar (Cozulmus)

- **Windows cp1254 encoding:** Unicode karakterler (box-drawing vb) crash yapiyor
  -> `safe_print()` fonksiyonu eklendi, ASCII'ye cevirir
- **Model 3 kez yukleniyor:** Her konusma icin ayri yukleme
  -> `model=None, tokenizer=None` parametreleri eklendi, main'de bir kez yuklenir
- **Familiarity=0 composite'i sifirliyor:** Kucuk korpuslarda tum n-gramlar "yeni"
  -> `familiarity = max(familiarity, 0.05)` tabani eklendi
