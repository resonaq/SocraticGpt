# Yapilacaklar ve Sonraki Adimlar

## ONCELIK 1: Composite Score Kalibrasyonu

**Sorun:** Motor "konu degistirme" ile "cerceve kirma" arasini ayirt edemiyor.

**Cozum oneris:** Semantic similarity faktoru ekle.

```
semantic_sim = cosine_similarity(embed(question), embed(context))

# Yuksek similarity + yuksek KL = GERCEK donusum (ayni konuda cerceve kirma)
# Dusuk similarity + yuksek KL = KONU DEGISTIRME (farkli alana gecis)

adjusted_composite = composite * semantic_sim_factor
```

Uygulama secenekleri:
- distilgpt2'nin kendi embedding'leri (ek model yukleme yok)
- sentence-transformers (daha iyi ama ek bagimlilik)
- TF-IDF cosine (en basit, bagimliliksiz)

## ONCELIK 2: Soru Tespiti Genisletme

**Sorun:** Sadece '?' ariyorr. Implicit reframe'ler kaciriliyor.

**Cozum:** Her assistant turn'unu analiz et, sadece soru icerenileri degil.
Ya da: response'un tamamini "soru" olarak degerlendir, '?' filtresi kaldir.

## ONCELIK 3: Esik Kalibrasyonu

Mevcut esikler (0.001 / 0.0001) cok dusuk.
10 ornekten yeni esikler turetilmeli:
- En dusuk gercek donusturucu: #6 monster = 0.0091
- En yuksek gercek mekanik: #1 burden = 0.0557

Bu zaten ters oldugu icin ONCELIK 1'den sonra tekrar bakilmali.

## ONCELIK 4: Coklu Tur Konusmalar

Mevcut test verileri 2 turlu (insan + yanit).
Gercek terapotik konusmalar 6-10 turlu.
Demo konusmalar (socratic_engine.py icindeki 3 ornek) zaten 7 turlu ama
manuel korpusten de coklu turlu ornekler eklenmeli.

## ONCELIK 5: socratic-lens Entegrasyonu

cgi_runner.py'deki `call_llm()` placeholder'i gercek bir LLM ile degistir:
- Anthropic Claude API (en uygun)
- Ya da: Matematik skorlarini LLM'e input olarak ver,
  LLM "neden donusturucu?" aciklamasini yapssin

## FIKIRLER (dusuk oncelik)

- [ ] Web UI: Streamlit/Gradio ile interaktif analiz
- [ ] Batch processing: buyuk dataset uzerinde toplu analiz
- [ ] Model karsilastirma: distilgpt2 vs gpt2-medium vs diger modeller
- [ ] Attention visualization: hangi token'lara dikkat degisiyor
- [ ] microgpt.py'yi de ayni pipeline'a bagla (2-seviyeli karsilastirma)
