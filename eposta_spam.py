# -*- coding utf-8 -*-

"""
Machine Learning Projesi: E-POSTA SPAM Tahmini
Kullanıcı tarafından alınan CSV dosyasını seçerek bu veri üzerinde console/terminal aracılığıyla ML uygulamasını yapacağız.

PROJE AMACI
--------------
Bir e-postanın SPAM olup olmadığını tahmin eden Classification Uygulaması oluşturmaktır.

CSV SÜTUNLARI
--------------
- kelime_sayisi
- link_sayisi
- buyuk_harf_orani
- supheli_kelime_sayisi
- gonderici_puani
- ek_var
- spam

Örnek:
kelime_sayisi, link_sayisi, buyuk_harf_orani, supheli_kelime_sayisi, gonderici_puani, ek_var, spam
36, 2, 0.08, 2, 77, 0, 0

`spam`:0 = normal, 1 = spam

"""

# -----------------------------------------------------------------------------------------

# os / pathlib:

# pandas:

# numpy:

# matplotlib:

# scikit-learn:

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AppState:
    def __init__(self):
        self.csv_path: Optional[Path] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

        self.target_column: Optional[str] = None
        self.target_column: List[str] = None

        self.model_results: List[Dict[str, Any]] = []

        self.preprocessing_completed: bool = False
        self.current_step: int = 1
