# Test Sonuclari ve Bulgular

## Test 1: Orijinal 3-Soru Deneyi (cgi_semantic.py)

Tek baglam: "I feel stuck in my career"
3 soru: When / Why / WhatIf

| Soru | Composite | Verdict |
|------|-----------|---------|
| What if stuck is protecting you? | 0.00259 | TRANSFORMATIVE |
| Why do you think you feel stuck? | 0.00030 | UNCERTAIN |
| When did this start? | 0.00000 | MECHANICAL |

**Bulgu:** Ontolojik soru (WhatIf) en yuksek, kronolojik soru (When) en dusuk.
Nadirllik kontrol edilince bile siralama korunuyor.

## Test 2: Mental Health Parquet (5 ornek)

Kaynak: `socratic-lens/tests/Mental Health Counseling Dataset/0000.parquet`
5 konusma secildi (yanitta soru iceren satirlar).

| Composite | Verdict |
|-----------|---------|
| 0.1485 | TRANSFORMATIVE |
| 0.0953 | TRANSFORMATIVE |
| 0.0395 | TRANSFORMATIVE |
| 0.0295 | TRANSFORMATIVE |
| 0.0065 | TRANSFORMATIVE |

**Bulgu:** Tum 5 ornek TRANSFORMATIVE cikti. Esikler kucuk kaliyorr.

## Test 3: 10 Manuel Korpus (KRITIK TEST)

Kaynak: `test_manual_corpus.json`
Beklenti: #5, #6, #8 donusturucu / #1, #2, #3, #4, #7, #9, #10 mekanik

### Sonuclar

| Sira | Ornek | Composite | Beklenen | Sonuc |
|------|-------|-----------|----------|-------|
| 1 | #1 burden ("depression lies") | 0.0557 | MECHANICAL | TRANSFORMATIVE |
| 2 | #7 sleep ("tried melatonin?") | 0.0487 | MECHANICAL | TRANSFORMATIVE |
| 3 | #3 husband ("discuss labor?") | 0.0237 | MECHANICAL | TRANSFORMATIVE |
| 4 | #5 identity ("who is underneath?") | 0.0186 | TRANSFORMATIVE | TRANSFORMATIVE |
| 5 | #8 boundaries ("loving = obeying?") | 0.0096 | TRANSFORMATIVE | TRANSFORMATIVE |
| 6 | #6 monster ("before you get angry?") | 0.0091 | TRANSFORMATIVE | TRANSFORMATIVE |

Atlanalar (yanit '?' icermiyor): #2, #4, #9, #10

### Kritik Bulgular

**1. SIRALAMA TERS:**
Mekanik yanitlar en yuksek, donusturucler en dusuk skoru aldi.

**Neden:** Motor "distributional novelty" olcuyor, "ontological transformation" degil.
- "Have you tried melatonin?" -> konuyu tamamen degistiriyor (uyku -> ilac) = BUYUK kayma
- "Who is the person underneath?" -> ayni anlam alaninda kaliyor ama cercereveyi kiriyor = KUCUK kayma

Yani: **Konu degistirmek != cerceve kirmak**. Motor bu ayrimi yapammiyor.

**2. SORU TESPITI YETERSIZ:**
Sadece '?' ariyorr. 4 ornek atlanadi cunku yanitlari soru icermiyor ama
yine de reframe/teknik iceriyor. Implicit reframe'ler de yakalanmali.

**3. ESIKLER ANLAMSIZ:**
Tum skorlar > 0.001 cikti. Esikler cok dusuk, ya da skor formulu
kalibre edilmeli.

## Temel Icingooru

> Composite score su anda "soru ne kadar FARKLI bir konu aciyor?" olcuyor.
> Olmasi gereken: "soru mevcut cercereveyi ne kadar KIRIYOIR?"
>
> Cozum yolu: Semantic similarity faktoru ekle.
> Soru ile baglam arasindaki konu yakinligi yuksekse VE dagilim kaymasi buyukse
> = GERCEK donusum (ayni konuda cercere kirma)
> Konu yakinligi dusukse VE dagilim kaymasi buyukse
> = KONU DEGISTIRME (farkli konu, donusum degil)
