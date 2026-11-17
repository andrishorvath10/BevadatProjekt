# Egészségbiztosítási költségek becslése regresszióval

Egyetemi beadandó projekt: egy Kaggle-adathalmazon végigvitt, dokumentált adatelemzési és modellezési folyamat (Jupyter notebook, magyar nyelvű magyarázatokkal és ábrákkal).

**Adathalmaz:** [Medical Cost Personal Datasets – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
1338 biztosított adatai (életkor, nem, BMI, gyermekek száma, dohányzás, régió). A **célváltozó** az éves egészségügyi költség (`charges`, USD).

## Mappaszerkezet

```
BevadatProjekt/
├── data/insurance.csv      # nyers adat (Kaggle)
├── figures/                # a notebook fontosabb ábrái PNG-ként + modell_eredmenyek.csv
├── projekt.ipynb           # a lefuttatott notebook (ez a beadandó)
├── projekt.html            # HTML-export a könnyű bemutatáshoz
├── build_notebook.py       # a notebookot előállító szkript (forrás)
├── requirements.txt
└── README.md
```

## A notebook felépítése

1. Bevezetés és problémafelvetés
2. Adatok betöltése, első áttekintés (méret, típusok, hiányzó értékek, duplikátumok)
3. Feltáró adatelemzés (eloszlások, kategóriák, boxplotok, scatter, korreláció)
4. Szűrés, előfeldolgozás (duplikátum, kiugró értékek, one-hot kódolás, rétegzett 80/20 felosztás)
5. Jellemzők kinyerése (`age_sq`, BMI-kategória, `obese_smoker`, `smoker_bmi`, korcsoport, log-célváltozó)
6. Hipotézisvizsgálat (H1–H4: Welch t, Mann–Whitney, ANOVA, Kruskal–Wallis, Fisher z)
7. Modellezés (7 modell, 5-szörös keresztvalidáció)
8. Kiértékelés és összehasonlítás
9. Konklúzió

## Futtatás

```bash
python3 -m venv .venv            # vagy: uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# a notebook futtatása (a kimenetek a fájlba kerülnek)
jupyter nbconvert --to notebook --execute --inplace projekt.ipynb
# HTML-export
jupyter nbconvert --to html projekt.ipynb

# vagy interaktívan:
jupyter notebook projekt.ipynb
```

A notebookot a repó gyökérmappájából kell futtatni, mert az adatot a `data/insurance.csv` relatív útvonalon keresi. Minden véletlenszerű lépés `random_state=42`-t használ, így az eredmények reprodukálhatók.

A `projekt.ipynb` a `build_notebook.py` szkripttel készült (`python build_notebook.py`). Ha a notebookot közvetlenül Jupyterben szerkeszted, ezt a szkriptet már nem kell újra futtatni, mert felülírná a módosításaidat.

## Fő eredmények

**Hipotézisek** (α = 0,05):

| | Eredmény |
|---|---|
| H1 – a dohányzók költsége magasabb | megerősítve (átlag ~32 000 vs. ~8 400 USD, p < 10⁻¹⁰⁰) |
| H2 – nincs különbség a nemek között | a rangalapú próbával megerősítve (p ≈ 0,69); a nyers átlagkülönbség a dohányzás zavaró hatása |
| H3 – nincs különbség a régiók között | a Kruskal–Wallis-próbával megerősítve (p ≈ 0,20) |
| H4 – a BMI hatása dohányzóknál erősebb | megerősítve (r ≈ 0,81 vs. 0,08; a meredekség ~18-szoros) |

**Modellek** (teszthalmaz: 268 sor; CV: 5-szörös a tanítóhalmazon):

| Modell | CV R² | Teszt R² | Teszt MAE (USD) | Teszt RMSE (USD) |
|---|---|---|---|---|
| Ridge + FE | 0,850 | 0,919 | 2 034 | 3 417 |
| Lineáris regresszió + FE | 0,849 | 0,919 | 2 044 | 3 427 |
| Random Forest | 0,836 | 0,913 | 2 153 | 3 537 |
| Döntési fa (mélység 4) | 0,821 | 0,900 | 2 429 | 3 794 |
| Lineáris regresszió (eredeti jellemzők) | 0,722 | 0,820 | 3 593 | 5 087 |
| Lin. regr. + FE, log(charges) | 0,499 | 0,524 | 3 781 | 8 287 |
| Baseline (átlag) | −0,015 | −0,001 | 9 081 | 12 013 |

**Tanulságok:**
- A költséget főleg a dohányzás, az életkor és a BMI határozza meg. A kulcs a **dohányzás × elhízás kölcsönhatás**: az elhízott dohányzók külön, nagyon magas költségű csoportot alkotnak.
- A feature engineering ugyanazon a lineáris modellen a CV R²-t 0,72-ről 0,85-re emelte. Az így bővített, jól értelmezhető lineáris modell legalább olyan jó, mint a Random Forest.
- A log-célváltozó itt rontott, mert a log-lineáris modell szorzó hatásokat feltételez, és a magas költségű eseteket erősen felülbecsüli.
- A teszt R² minden modellnél kicsit jobb a CV-nél, ezért a reálisabb becslés a CV R² ≈ 0,85.
