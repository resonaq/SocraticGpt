# SocraticGpt - Project Memory

> Bu dosya projenin durumunu, kararlarini ve bulgualarini ozetler.
> Devam edecegimiz zaman buradan tarayin.

## Proje Ozeti

**Amac:** "Dogru soru baglami donusturur" hipotezini matematiksel olarak olcmek.
CGI (Context Grammar Induction) teorisi: Donusturucu sorular bir dil modelinin
uretici grammarini yeniden organize eder -- sadece sasirtmaz, yapiyi degistirir.

## Dosya Haritasi

| Dosya | Ne Yapar |
|-------|----------|
| `microgpt.py` | Karakter seviyesi GPT (4,224 param), sifirdan autograd, Karpathy ilhami |
| `cgi_semantic.py` | Semantik seviye analiz (distilgpt2, 82M param), tum CGI metrikleri |
| `socratic_engine.py` | **Birlesik motor** - konusma analizi + CGI metrikleri |
| `socratic-lens/` | 6-chain LLM pipeline (prompt tabanli, placeholder) |
| `ROADMAP.md` | Gelistirme yol haritasi (Turkce) |
| `test_manual_corpus.json` | 10 manuel korpus ornegi (test verisi) |
| `test_mental_health.json` | 5 parquet ornegi (test verisi) |

## Detayli Notlar

- [architecture.md](architecture.md) - Mimari, veri akisi, formul
- [findings.md](findings.md) - Test sonuclari, kalibrasyon bulgulari
- [next-steps.md](next-steps.md) - Yapilacaklar, fikirler, bilinen sorunlar
