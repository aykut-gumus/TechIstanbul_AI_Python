# AI Learning - Eposta Spam Tahmini
- Bu proje konsol üzerinden çalışan, eğitim amaçlı bir sınıflandırma uygulamasıdır.
---

## Github Link
[GitHub](https://github.com/aykut-gumus/TechIstanbul_AI_Python)

### Requirements.txt yükle

```sh 
py -m pip install -r requirements.txt
py eposta_spam.py
```
---

### CSV örnek Sütunları
```sh 
- kelime_sayisi
- link_sayisi
- buyuk_harf_orani
- supheli_kelime_sayisi
- gonderici_puani
- ek_var
- spam
```
`spam`:0 = normal, 1 = spam
---


### Örnek Console üzerinden Çözüm sırası
STEP-1
1. CSV dosyasını seç/yükle (1a.Manuel yazsın, 2a.Dinamik seçsin)

STEP-2
2. Veri Temizle & Kaydet
3. Veriyi önizle
4. Veri Kalitesi analizi
5. Doğal Veriyi tekrar yükle

STEP-3
6. Model Eğit (Logistic Regression, Decision Tree, Random Forest)
7. Model sonuçları karşılaştır
8. Classification Report - PDF, txt olarak kaydet
9. Confusion Matrix
10. Yeni e-posta için tahmin
0. Çıkış
---

