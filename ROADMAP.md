# SocraticGpt — Geliştirme Yol Haritası

> *"Algoritma tamamdır. Gerisi sadece verimlilik."* — Andrej Karpathy
>
> *"Bildiğim tek şey, hiçbir şey bilmediğimdir."* — Sokrates
>
> *"Doğru soru, en iyi cevabı alan değil — bağlamı dönüştüren sorudur."* — CGI Teorisi

---

## Karpathy × Sokrates: Think Tank Konsensüsü

Bu yol haritası, iki zihniyet arasındaki hayali bir masabaşı toplantısından doğdu.

### Karpathy Masaya Ne Getirdi

| İlke | Açıklama | Projede Karşılığı |
|------|----------|-------------------|
| **Sıfırdan inşa et** | Her katmanı elle yaz, soyutlamayı kaldır | `microgpt.py` — sıfır bağımlılık, 4.224 parametre |
| **Kademeli ölçekle** | micrograd → nanoGPT → llm.c sıkıştırma arkı | Karakter → Semantik → Etkileşimli |
| **Ampirik doğrula** | Teori güzel, ölçüm gerçek | KL divergence, Spearman, Jaccard, Attention Entropy |
| **Şeffaflık** | Kara kutu yok, her işlem görünür | Autograd motoru, el yazısı Adam optimizer |

### Sokrates Masaya Ne Getirdi

| İlke | Açıklama | Projede Karşılığı |
|------|----------|-------------------|
| **Elenchus** (sorgulama) | Varsayımları soru sorarak ortaya çıkar | Zincir 1-3: Korpustan "bağlam" tanımı öğren |
| **Aporia** (çıkmaz) | Üretken şaşkınlık → derin kavrayış | Faz 1 bulgusu: KL nadirliği ölçüyor, anlamı değil |
| **Maieutik** (doğurtma) | Cevap verme, keşfettir | Zincir 5-6: Aday göster, karar insana kalsın |
| **"Bilmiyorum"** | Öğrenmeden yargılama | Lens olmadan sınıflandırma yok |

### Ortak Konsensüs

```
Karpathy: "İnşa et ki anlayasın."
Sokrates:  "Soru sor ki göresin."
Birlikte:  "Soru soran sistemler inşa et — hem sistem anlasın, hem soran."
```

**Üzerinde anlaştıkları evrim ilkesi:**

1. Her faz bir öncekinin **varsayımını** sorgulamalı (Sokratik ilerleme)
2. Her faz öncekinin **üzerine** inşa etmeli, bağımsız çalışabilmeli (Karpathy arkı)
3. Her faz **ölçülebilir çıktılar** üretmeli — teori değil, veri (ampirik disiplin)
4. İnsan her zaman **son otorite** kalmalı (Sokratik alçakgönüllülük)

---

## Faz 0 — Mevcut Durum: Temel Atıldı

### Ne Var

```
SocraticGpt/
├── microgpt.py                  # Faz 0a: Karakter seviyesi CGI kanıtı
├── cgi_semantic.py              # Faz 0b: Semantik seviye doğrulama
├── cgi_char_results.json        # Karakter seviye metrik çıktıları
├── cgi_semantic_results.json    # Semantik seviye metrik çıktıları
├── README.md                    # Proje dokümantasyonu
└── socratic-lens/               # Faz 0c: Pratik keşif aracı
    ├── cgi_runner.py            # 6-zincir pipeline motoru
    ├── gpt-instructions.md      # LLM sistem talimatları
    ├── chains/CGI-{1..6}*.yaml  # Zincir tanımları
    └── tests/                   # Mental health dataset testleri
```

### Ne Kanıtlandı

| Bulgu | Kanıt |
|-------|-------|
| Ontolojik sorular semantik seviyede en yüksek yeniden-örgütlenme skoru üretir | `WhatIf composite: 0.00259 > Why: 0.00030 > When: 0.00000` |
| Karakter ve semantik seviye sıralaması **tersine döner** — nadirllik kontrol edilince | Char: Why > WhatIf > When → Semantic: WhatIf > Why > When |
| Gerçek danışmanlık verilerinin %90'ı mekanik yanıt içerir | 30 örnekten 27'si mekanik, 3'ü dönüştürücü |
| Lens kendini günceller: "duygu sormak" → "değer takası yaratmak" | Zincir 6 meta-yansıması |

### Ne Eksik

| Eksiklik | Neden Kritik |
|----------|-------------|
| İstatistiksel güven aralıkları yok | Tek seed (42), bootstrap yok, hata çubukları yok |
| LLM entegrasyonu placeholder | `cgi_runner.py` → `call_llm()` boş |
| Tek baz bağlam test edildi | Sadece "I feel stuck in my career" |
| Sadece 3 soru tipi | Temporal, Reflective, Ontological — daha fazlası lazım |
| Ablasyon çalışması yok | Hangi bileşen dönüşümü tetikliyor? |
| İnsan değerlendirmesi yok | Metrikler model-internal; insanla korelasyon bilinmiyor |
| Cross-domain test yok | Sadece terapi alanı |

---

## Faz 1 — Konsolidasyon: Temeli Sağlamlaştır

> **Sokratik Soru:** *"Ölçtüğümüz şey gerçekten dönüşüm mü, yoksa istatistiksel gürültü mü?"*

### Hedef

Mevcut ampirik kanıtları istatistiksel olarak sağlamlaştır, `cgi_runner.py`'ye gerçek LLM entegrasyonu yap, korpus işleme standardize et.

### Görevler

#### 1.1 İstatistiksel Güçlendirme (`microgpt.py` + `cgi_semantic.py`)

- [ ] **Çoklu seed çalıştırma**: 10 farklı random seed (42, 123, 256, 512, 1024, 2048, 4096, 7777, 9999, 31415)
- [ ] **Bootstrap resampling**: Her metrik için %95 güven aralığı hesapla
- [ ] **Çoklu baz bağlam**: En az 5 farklı terapötik bağlam testi
  - "I feel stuck in my career"
  - "I can't stop worrying about the future"
  - "My relationship is falling apart"
  - "I don't know who I am anymore"
  - "Everything feels meaningless"
- [ ] **Sonuç formatı**: Her metrik için `ortalama ± std` raporla

#### 1.2 LLM Entegrasyonu (`socratic-lens/cgi_runner.py`)

- [ ] `call_llm()` fonksiyonunu gerçek API çağrısıyla değiştir
- [ ] Desteklenecek sağlayıcılar: Anthropic Claude, OpenAI GPT, yerel model (ollama)
- [ ] Yapılandırılmış çıktı (JSON) doğrulama ve yeniden deneme mekanizması
- [ ] API anahtarı yönetimi (ortam değişkenleri)
- [ ] Rate limiting ve hata yönetimi

#### 1.3 Korpus Standardizasyonu

- [ ] Birleşik korpus formatı tanımla: `{"id": str, "turns": [{"role": str, "content": str}]}`
- [ ] Format dönüştürücüler: CSV, JSON, Parquet, düz metin → standart format
- [ ] Doğrulama şeması (JSON Schema)

#### 1.4 Proje Altyapısı

- [ ] `requirements.txt` — tüm bağımlılıklar (torch, transformers, pyyaml, anthropic/openai)
- [ ] `pyproject.toml` — proje meta verisi
- [ ] Temel birim test altyapısı (`pytest`)
- [ ] CI pipeline taslağı (GitHub Actions)

### Test Senaryoları — Faz 1

| # | Senaryo | Girdi | Beklenen Çıktı | Doğrulama |
|---|---------|-------|-----------------|-----------|
| T1.1 | **Çoklu seed tutarlılığı** | `microgpt.py` × 10 seed | WhatIf'in semantik composite skoru 10 çalıştırmada da en yüksek | `assert all(whatif_score > why_score for each seed)` |
| T1.2 | **Güven aralığı hesaplaması** | 10 seed sonucu | Her metrik için `mean ± 95% CI` | CI'lar sıfırı içermemeli (anlamlı fark) |
| T1.3 | **LLM entegrasyon smoke test** | `cgi_runner.run(sample_corpus)` | 6 zincir hatasız tamamlanır, JSON çıktı geçerli | Her zincir çıktısı şemaya uygun |
| T1.4 | **Korpus format dönüşümü** | CSV + Parquet + JSON girdi | Hepsi aynı standart formata dönüşür | `assert normalize(csv) == normalize(json) == normalize(parquet)` |
| T1.5 | **Farklı baz bağlam robustluğu** | 5 farklı terapötik bağlam | Ontolojik soruların ortalaması > Reflective > Temporal | En az 4/5 bağlamda sıralama korunmalı |

### Çıktılar

- `results/statistical_report.json` — çoklu seed sonuçları + güven aralıkları
- `cgi_runner.py` — çalışan LLM entegrasyonu
- `corpus/schema.json` — standart korpus şeması
- `tests/test_phase1.py` — otomatik testler

---

## Faz 2 — Ölçekleme: Sınırları Genişlet

> **Sokratik Soru:** *"Dönüşüm evrensel mi, yoksa bağlama özgü mü? Terapide işleyen, teknik destekte de işler mi?"*

### Hedef

Daha büyük korpuslar, daha fazla soru tipi, birden fazla domain. CGI teorisinin genelleme kapasitesini test et.

### Görevler

#### 2.1 Soru Tipolojisini Genişlet

Mevcut 3 tip → 7 tip:

| Tip | Örnek | CGI Tahmini |
|-----|-------|-------------|
| Temporal (mevcut) | "Bu ne zaman başladı?" | Düşük yeniden-örgütlenme |
| Reflective (mevcut) | "Neden takılı hissediyorsun?" | Orta yeniden-örgütlenme |
| Ontological (mevcut) | "Ya takılmak seni koruyorsa?" | Yüksek yeniden-örgütlenme |
| **Karşıolgusal** | "Bu sorun hiç var olmasaydı ne farklı olurdu?" | Yüksek (çerçeve değişimi) |
| **İtirafçı** | "Bunun hakkında kime söylemedin?" | Orta-Yüksek (gizli ekseni açar) |
| **Normatif** | "Bu 'yapmalıyım' kimin sesi?" | Yüksek (otorite kaynağını sorgular) |
| **Absürt** | "Takılı kalmakta uzman olsan, ne tavsiye verirdin?" | Test edilecek (paradoksal) |

#### 2.2 Çoklu Domain Testi

| Domain | Veri Kaynağı | Dönüşüm Tanımı |
|--------|-------------|-----------------|
| Terapi (mevcut) | Mental Health Counseling Dataset | Çerçeve değişimi |
| **Teknik destek** | Stack Overflow conversations | Problem çerçevesini yeniden tanımlama |
| **Eğitim** | Socratic teaching transcripts | Kavram atlaması |
| **İş stratejisi** | Business case discussions | Varsayım revizyonu |

#### 2.3 Daha Büyük Korpus

- [ ] Terapi: 30 → 200+ konuşma
- [ ] Her yeni domain: minimum 100 konuşma
- [ ] Stratified sampling iyileştirmesi: konuşma uzunluğu, karmaşıklık, sonuç türüne göre

#### 2.4 Karşılaştırmalı Lens Analizi

- [ ] Aynı korpus üzerinde farklı LLM'lerle lens oluştur (Claude vs GPT vs yerel)
- [ ] Lens uyum oranı hesapla: modeller arası agreement %
- [ ] Domain-specific vs domain-agnostic sinyalleri ayır

### Test Senaryoları — Faz 2

| # | Senaryo | Girdi | Beklenen Çıktı | Doğrulama |
|---|---------|-------|-----------------|-----------|
| T2.1 | **7 soru tipi sıralaması** | 7 tip × semantik model | Ontolojik + Karşıolgusal + Normatif > Temporal | Composite skor sıralaması tutarlı |
| T2.2 | **Cross-domain lens farklılığı** | 4 domain × lens çıktısı | Her domain farklı `decision_question` üretir | `assert lens_therapy != lens_techsupport` |
| T2.3 | **Corpus büyüklüğü etkisi** | Aynı domain: 30, 100, 200 örnekle lens | Lens 100+ örnekte stabilize olur | `jaccard(lens_100, lens_200) > 0.8` |
| T2.4 | **LLM lens uyumu** | Aynı korpus × Claude + GPT + yerel | Agreement > %70 dönüştürücü sınıflandırmada | Cohen's kappa > 0.6 |
| T2.5 | **Absürt soru testi** | "Takılmakta uzman olsan?" tipi | CGI skoru ölçülür, pozisyonu belirlenir | Skor temporal'den yüksek olmalı |

### Çıktılar

- `questions/typology.yaml` — 7 soru tipi tanımları + örnekleri
- `corpus/` — 4 domain korpusu standart formatta
- `results/cross_domain_report.json` — domain karşılaştırma sonuçları
- `results/lens_agreement.json` — LLM'ler arası uyum analizi

---

## Faz 3 — Etkileşimli Sokratik Motor: Soruyu Üret

> **Sokratik Soru:** *"Dönüşümü ölçmek yetmez — onu üretebilir miyiz?"*

### Hedef

Pasif analizden aktif soru üretimine geç. Kullanıcı bir bağlam verir, sistem yüksek CGI skorlu sorular üretir ve geri bildirimle öğrenir.

### Görevler

#### 3.1 Soru Üretim Motoru

```
Kullanıcı Bağlamı → Lens Seçimi → Aday Soru Üretimi → CGI Skorlama → Sıralama → Sunma
```

- [ ] Bağlamdan otomatik domain tespiti (terapi, teknik, eğitim, iş)
- [ ] Domain'e uygun lens yükleme (Faz 2 çıktıları)
- [ ] LLM ile aday soru üretimi (N=10 aday)
- [ ] Her aday için CGI metriklerini hesapla (KL, Spearman, Jaccard, Diversity)
- [ ] En yüksek composite skorlu 3 soruyu sun

#### 3.2 Geri Bildirim Döngüsü

```
Soru Sunuldu → Kullanıcı Değerlendirdi → Lens Güncellendi → Sonraki Sorular Daha İyi
```

- [ ] Kullanıcı her soruyu değerlendirir: `dönüştürücü` / `mekanik` / `emin değilim`
- [ ] Uyumsuzluklar birikir → eşik aşılınca lens yeniden kalibre edilir
- [ ] Kalibrasyon geçmişi saklanır (lens versiyonlama)

#### 3.3 CLI Arayüzü

```bash
# Temel kullanım
python socratic_engine.py --context "Kariyerimde takılı hissediyorum"

# Domain belirterek
python socratic_engine.py --context "..." --domain therapy

# Etkileşimli mod
python socratic_engine.py --interactive
```

#### 3.4 Oturum Yönetimi

- [ ] Konuşma geçmişi takibi (hangi sorular soruldu, kullanıcı tepkisi ne oldu)
- [ ] Bağlam penceresi yönetimi (uzun konuşmalarda ne kadar geçmiş tutulacak)
- [ ] Oturum özeti ve CGI zaman serisi (dönüşüm ne zaman gerçekleşti?)

### Test Senaryoları — Faz 3

| # | Senaryo | Girdi | Beklenen Çıktı | Doğrulama |
|---|---------|-------|-----------------|-----------|
| T3.1 | **Soru üretim kalitesi** | 5 farklı bağlam | Her bağlam için 3 aday soru, hepsi bağlamla ilgili | İnsan değerlendirici %80+ "ilgili" demeli |
| T3.2 | **CGI skor sıralaması** | Üretilen 10 aday | En yüksek skorlu 3'ün composite skoru > medyan | `assert top3_mean > all10_median` |
| T3.3 | **Geri bildirim öğrenme** | 20 soru-değerlendirme çifti | Lens güncellendikten sonra uyum artar | Uyum oranı iterasyon başına artmalı |
| T3.4 | **Domain otomatik tespiti** | Karışık bağlamlar (terapi + teknik + eğitim) | Doğru domain tespiti | `accuracy > 0.85` |
| T3.5 | **Etkileşimli oturum** | 5 turlu konuşma | Sorular ilerledikçe bağlama uyum sağlar | Son turda soru kalitesi ilk turdan yüksek |

### Çıktılar

- `socratic_engine.py` — etkileşimli soru üretim motoru
- `feedback/` — kullanıcı geri bildirim veritabanı
- `lenses/` — versiyonlanmış lens arşivi
- `tests/test_engine.py` — motor birim testleri

---

## Faz 4 — Meta-Öğrenme: Dönüşümü Dönüştür

> **Sokratik Soru:** *"Model dönüşümü ölçüyor — ama modelin ölçümü insanın deneyimiyle örtüşüyor mu?"*

### Hedef

Model-internal metrikler (KL, entropy) ile insan-reported deneyim arasındaki hizalamayı ölç. Kişiselleştirme ve cross-domain transfer öğren.

### Görevler

#### 4.1 İnsan-Model Hizalama Çalışması

- [ ] 50 soru-bağlam çifti hazırla
- [ ] 10+ insan değerlendirici: "Bu soru bakış açınızı değiştirdi mi?" (1-10 ölçek)
- [ ] Her çift için CGI composite skoru hesapla
- [ ] Spearman korelasyonu: `insan_skoru ~ cgi_skoru`
- [ ] Hedef: `rho > 0.5` (orta-güçlü korelasyon)

#### 4.2 Kişiselleştirme

- [ ] Kullanıcı profili: Hangi soru tipleri bu kişide dönüşüm yarattı?
- [ ] Adaptif soru seçimi: Kişiye özel sıralama
- [ ] Gizlilik koruması: Profil verileri yerel, paylaşılmaz

#### 4.3 Cross-Domain Transfer

- [ ] Terapi lens'i eğitim corpus'unda ne kadar işler?
- [ ] Ortak "evrensel dönüşüm sinyalleri" var mı?
- [ ] Transfer skoru: `domain_A_lens → domain_B_corpus` başarı oranı

#### 4.4 Nedensellik Analizi

- [ ] Ablasyon: Sorunun hangi kelimesi dönüşümü tetikliyor?
- [ ] Gradient-based saliency: Baz bağlamın hangi kısmı soruyla en çok etkileşiyor?
- [ ] Minimal sufficient question: Aynı CGI skorunu üreten en kısa soru

### Test Senaryoları — Faz 4

| # | Senaryo | Girdi | Beklenen Çıktı | Doğrulama |
|---|---------|-------|-----------------|-----------|
| T4.1 | **İnsan-model korelasyonu** | 50 çift × 10 değerlendirici | Spearman rho > 0.5 | p-value < 0.01 |
| T4.2 | **Kişiselleştirme etkisi** | Kullanıcı A: 20 etkileşim sonrası | Kişisel sıralama > genel sıralama | `personal_hit_rate > generic_hit_rate` |
| T4.3 | **Cross-domain transfer** | Terapi lens → eğitim corpus | Transfer başarısı > rastgele | `accuracy > 0.5 (chance)` |
| T4.4 | **Ablasyon testi** | "Ya takılmak seni koruyorsa?" → kelime çıkarma | "koruyorsa" çıkarıldığında skor düşer | `score_full > score_ablated` |
| T4.5 | **Minimal soru testi** | Tam soru → progressif kısaltma | Minimum etkili soru uzunluğu belirlenir | Skor eşiğinin altına düşen kelime sayısı |

### Çıktılar

- `alignment/human_study_results.json` — insan-model hizalama verileri
- `profiles/` — anonim kullanıcı profilleri
- `transfer/cross_domain_matrix.json` — domain transfer matrisi
- `analysis/ablation_report.json` — nedensellik analizi

---

## Faz 5 — Açık Araştırma Platformu: Ekosistem Ol

> **Sokratik Soru:** *"Bu araç yalnızca bizim anlamamıza mı hizmet ediyor, yoksa başkalarının da sorgulamasını mümkün kılıyor mu?"*

### Hedef

SocraticGpt'yi araştırmacılar ve uygulayıcılar için açık bir platform haline getir. API, dokümantasyon, topluluk katkısı altyapısı.

### Görevler

#### 5.1 API Katmanı

```python
from socraticgpt import CGIAnalyzer, SocraticEngine, LensBuilder

# Metrik hesaplama
analyzer = CGIAnalyzer(model="distilgpt2")
metrics = analyzer.measure(base_context, question)

# Soru üretimi
engine = SocraticEngine(domain="therapy")
questions = engine.generate(context, top_k=3)

# Lens oluşturma
builder = LensBuilder(llm="claude-opus-4-6")
lens = builder.build(corpus)
```

#### 5.2 Yayın-Hazır Deneysel Framework

- [ ] Tekrarlanabilir deney şablonları
- [ ] Standart raporlama formatı (LaTeX + JSON)
- [ ] Benchmark dataset'ler (terapi, eğitim, teknik, iş)
- [ ] Baseline karşılaştırma araçları

#### 5.3 Topluluk Altyapısı

- [ ] Katkı kılavuzu (CONTRIBUTING.md)
- [ ] Yeni domain/soru tipi ekleme rehberi
- [ ] Lens paylaşım platformu (anonim, versiyonlanmış)
- [ ] Tartışma forumu veya GitHub Discussions entegrasyonu

#### 5.4 Etik Çerçeve

- [ ] Kullanım kılavuzu: Sistem neyi yapabilir, neyi yapmamalı
- [ ] Güvenlik sınırları: Hangi bağlamlarda dönüştürücü soru uygunsuz?
- [ ] Onay mekanizması: Kullanıcı dönüşüme hazır mı?
- [ ] Veri gizliliği politikası

### Test Senaryoları — Faz 5

| # | Senaryo | Girdi | Beklenen Çıktı | Doğrulama |
|---|---------|-------|-----------------|-----------|
| T5.1 | **API smoke test** | `CGIAnalyzer.measure(ctx, q)` | Geçerli metrik dict döner | Tüm metrik anahtarları mevcut, değerler sayısal |
| T5.2 | **Yeni domain ekleme** | Yeni corpus + rehber takibi | 6-zincir pipeline çalışır, lens üretilir | Lens şemasına uygun çıktı |
| T5.3 | **Tekrarlanabilirlik** | Aynı deney × 2 farklı makine | Aynı sonuçlar (deterministic seed ile) | `assert results_A == results_B` |
| T5.4 | **Etik sınır testi** | Kriz bağlamı (intihar riski) | Sistem uyarı verir, dönüştürücü soru üretmez | Güvenlik filtresi aktif |

### Çıktılar

- `socraticgpt/` — Python paketi (pip install edilebilir)
- `benchmarks/` — standart test dataset'leri
- `docs/` — API dokümantasyonu, katkı rehberi, etik çerçeve
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`

---

## Mimari Genel Bakış: Fazlar Arası İlişki

```
Faz 0: TEMELİ AT
  microgpt.py ──────────── "Dönüşümü ölçebildik mi?"
  cgi_semantic.py ──────── "Nadirlliği kontrol edince de işliyor mu?"
  socratic-lens/ ───────── "Bunu herhangi bir korpusa uygulayabilir miyiz?"
       │
       ▼
Faz 1: SAĞLAMLAŞTIR ────── "Ölçümümüz güvenilir mi?"
  İstatistiksel güçlendirme + LLM entegrasyonu + altyapı
       │
       ▼
Faz 2: GENİŞLET ──────── "Başka alanlarda da çalışıyor mu?"
  Çoklu domain + genişletilmiş soru tipleri + karşılaştırmalı lens
       │
       ▼
Faz 3: ÜRET ──────────── "Dönüşümü sadece ölçmek değil, yaratabilir miyiz?"
  Etkileşimli motor + geri bildirim döngüsü + adaptif soru seçimi
       │
       ▼
Faz 4: HIZALA ─────────── "Model ne diyor, insan ne diyor — örtüşüyor mu?"
  İnsan çalışması + kişiselleştirme + nedensellik + cross-domain transfer
       │
       ▼
Faz 5: PAYLAŞ ─────────── "Bunu başkaları da kullanabilir mi?"
  API + benchmark + topluluk + etik çerçeve
```

### Her Fazın Sokratik Meydan Okuması

| Faz | Önceki Fazın Varsayımı | Bu Fazın Sorusu |
|-----|------------------------|-----------------|
| 0→1 | "Metrikler doğru çalışıyor" | "Ama istatistiksel olarak anlamlı mı?" |
| 1→2 | "Terapi alanında çalışıyor" | "Başka alanlarda da geçerli mi?" |
| 2→3 | "Dönüşümü tespit edebiliyoruz" | "Peki üretebilir miyiz?" |
| 3→4 | "Model yüksek skor verdi" | "İnsan da aynı şeyi hissetti mi?" |
| 4→5 | "Bizim için çalışıyor" | "Başkaları da kullanabilir mi?" |

---

## Teknoloji Kararları ve Trade-off'lar

| Karar | Seçilen | Alternatif | Neden |
|-------|---------|------------|-------|
| Autograd motoru | Elle yazılmış (Value class) | PyTorch autograd | Pedagojik şeffaflık — Karpathy ilkesi |
| Semantik model | distilgpt2 (82M) | GPT-2 large (774M) | Yeterli + hızlı + herkesin çalıştırabileceği boyut |
| Pipeline orchestration | Sıralı zincirler (YAML) | DAG framework (Airflow) | Basitlik — gereksiz karmaşıklık ekleme |
| LLM çağrısı | Çoklu sağlayıcı (pluggable) | Tek sağlayıcıya bağlı | Esneklik + erişilebilirlik |
| Lens depolama | JSON/YAML dosyaları | Veritabanı | Versiyonlama (git) + okunabilirlik |
| Metriklerde GPU | Opsiyonel (CPU default) | GPU zorunlu | Erişilebilirlik — herkes çalıştırabilmeli |

---

## Hızlı Başlangıç: İlk Katkı

Projeye katkıda bulunmak isteyen biri için en düşük bariyerli başlangıç noktaları:

### Yeni Başlayanlar İçin

1. **Yeni baz bağlam ekle** — `microgpt.py`'deki training corpus'a yeni bir cümle ekle, metriklerin nasıl değiştiğini gözlemle
2. **Test senaryosu yaz** — Faz 1 test senaryolarından birini `pytest` olarak implemente et
3. **Korpus formatı dönüştürücü** — Yeni bir veri formatı (örn. CSV) → standart format dönüştürücü yaz

### Orta Seviye

4. **Yeni soru tipi** — Faz 2'deki 4 yeni soru tipinden birini `microgpt.py`'ye ekle ve CGI metriklerini ölç
5. **Bootstrap resampling** — `cgi_semantic.py`'ye güven aralığı hesaplaması ekle
6. **Farklı LLM entegrasyonu** — `cgi_runner.py`'ye tercih ettiğin LLM API'sini bağla

### İleri Seviye

7. **Yeni domain** — Eğitim veya teknik destek corpus'u hazırla, socratic-lens pipeline'ını çalıştır
8. **Ablasyon çalışması** — Hangi kelimenin dönüşümü tetiklediğini bul
9. **İnsan çalışması tasarımı** — Faz 4 hizalama deneyinin protokolünü yaz

---

## Zaman Çizelgesi Tahmini

> Karpathy: *"Ne kadar süreceğini söyleme — ne yapılacağını söyle."*
>
> Sokrates: *"Acele eden, cevaba ulaşır. Sabırlı olan, soruya."*

Zaman tahmini vermiyoruz. Her faz, önceki faz tamamlanınca başlar. Her faz bağımsız bir değer üretir — yarım kalan faz bile öğrenme sağlar.

**Tek kural:** Her merge, projeyi bozmamalı. Her faz, önceki fazın testlerini geçmeye devam etmeli.

---

*Bu roadmap, Andrej Karpathy'nin "karmaşıklığı sadeleştirerek anla" felsefesi ile Sokrates'in "soru sorarak dönüştür" yöntemiyle yazıldı. Her faz hem bir mühendislik adımı hem bir felsefi sorgulama.*

*Projenin kendisi, iddiasının kanıtıdır: Doğru soru bağlamı dönüştürür.*
