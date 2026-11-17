"""A projekt.ipynb notebook előállítása (nbformat segítségével).

Futtatás:  .venv/bin/python build_notebook.py
Utána:     .venv/bin/jupyter nbconvert --to notebook --execute --inplace projekt.ipynb
"""
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


# =====================================================================
# 0. Címlap
# =====================================================================
md(r"""
# Egészségbiztosítási költségek becslése regresszióval

**Beadandó projekt – adatelemzés és gépi tanulás**

**Adatforrás:** [Medical Cost Personal Datasets – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`)

**Tartalom**

1. Bevezetés és problémafelvetés
2. Adatok betöltése és első áttekintés
3. Feltáró adatelemzés (EDA)
4. Szűrés, előfeldolgozás
5. Jellemzők kinyerése (feature engineering)
6. Hipotézisvizsgálat
7. Modellezés
8. Kiértékelés és összehasonlítás
9. Konklúzió
""")

# =====================================================================
# 1. Bevezetés
# =====================================================================
md(r"""
## 1. Bevezetés és problémafelvetés

Az egészségbiztosítók számára alapvető kérdés, hogy egy ügyfél várhatóan mekkora egészségügyi költséget fog okozni. Ha ez jól becsülhető, akkor:

- a díjak igazságosabban és kockázatarányosan állapíthatók meg,
- azonosíthatók azok a tényezők (pl. dohányzás, elhízás), amelyekre prevenciós programokat érdemes építeni,
- a biztosító pontosabban tervezheti a tartalékait.

**Az adathalmaz** 1338 amerikai biztosított adatait tartalmazza. Oszlopok:

| Oszlop | Jelentés | Típus |
|---|---|---|
| `age` | a biztosított életkora (év) | numerikus |
| `sex` | nem (`female` / `male`) | kategorikus |
| `bmi` | testtömegindex (kg/m²) | numerikus |
| `children` | eltartott gyermekek száma | diszkrét numerikus |
| `smoker` | dohányzik-e (`yes` / `no`) | kategorikus |
| `region` | lakóhely régiója az USA-ban (4 kategória) | kategorikus |
| `charges` | **az egészségbiztosító által elszámolt éves orvosi költség (USD)** | numerikus – **célváltozó** |

**A modellezési probléma:** *felügyelt regresszió*. A célváltozó a `charges`, magyarázó változók a többi oszlop. A modellek jóságát a teszthalmazon mért **MAE**, **RMSE** és **R²** mutatókkal mérjük, és egy egyszerű *baseline* (átlagbecslés) modellhez viszonyítjuk.

A hangsúly nem a legbonyolultabb modellen van, hanem azon, hogy minden lépés (tisztítás, jellemzők, modellválasztás) indokolt és ellenőrzött legyen.
""")

# =====================================================================
# 2. Betöltés
# =====================================================================
md(r"""
## 2. Adatok betöltése és első áttekintés

Először betöltjük a szükséges könyvtárakat, és beállítjuk az ábrák egységes stílusát. A `RANDOM_STATE = 42` konstanst minden véletlenszerű lépésnél használjuk, így az eredmények reprodukálhatók.
""")

code(r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_PATH = Path("data/insurance.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.figsize": (8, 4.5), "figure.dpi": 100, "axes.titleweight": "bold"})
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

SMOKER_PALETTE = {"nem": "#4C72B0", "igen": "#C44E52"}


def save_fig(fig, name):
    # Az ábrát PNG-ként is elmentjük a figures/ mappába.
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
""")

md(r"""
Az adatfájlt a Kaggle-ről töltöttük le, és a `data/insurance.csv` helyre tettük. Nézzük meg az első néhány sort.
""")

code(r"""
df_raw = pd.read_csv(DATA_PATH)
df_raw.head()
""")

md(r"""
### Méret, adattípusok, hiányzó értékek
""")

code(r"""
print(f"Sorok száma: {df_raw.shape[0]}, oszlopok száma: {df_raw.shape[1]}\n")
info = pd.DataFrame({
    "típus": df_raw.dtypes.astype(str),
    "hiányzó értékek": df_raw.isna().sum(),
    "egyedi értékek": df_raw.nunique(),
})
info
""")

md(r"""
Az adathalmazban **nincs hiányzó érték**. Három numerikus (`age`, `bmi`, `charges`), egy diszkrét numerikus (`children`) és három szöveges, kategorikus oszlop (`sex`, `smoker`, `region`) van. A kategorikus oszlopokat a modellezéshez kódolni kell (lásd 4. fejezet).
""")

md(r"""
### Leíró statisztikák
""")

code(r"""
df_raw.describe().T
""")

code(r"""
df_raw[["sex", "smoker", "region"]].describe().T
""")

md(r"""
### Duplikátumok
""")

code(r"""
dup_mask = df_raw.duplicated(keep=False)
print(f"Duplikált sorok száma (az ismétlődések): {df_raw.duplicated().sum()}")
df_raw[dup_mask]
""")

md(r"""
**Első benyomások:**

- Az életkor 18 és 64 év között van, tehát csak felnőttek szerepelnek az adatban.
- A BMI átlaga kb. 30,7, ami az elhízás határán van. Ez az amerikai mintában nem meglepő.
- A `charges` átlaga (~13 300 USD) jóval nagyobb a mediánnál (~9 400 USD). Ez **jobbra ferde** eloszlásra utal, amit a következő fejezetben ábrán is megnézünk.
- Egy teljesen azonos sorpár van. Ezt a 4. fejezetben kezeljük.
""")

# =====================================================================
# 3. EDA
# =====================================================================
md(r"""
## 3. Feltáró adatelemzés (EDA)

Az EDA-t a már duplikátummentes adatokon végezzük (az egyetlen ismétlődő sor elhagyása az eredményeken érdemben nem változtat, de így minden ábra pontosan a későbbi modellezési adatokat mutatja). Az ábrákon a jobb olvashatóság kedvéért magyar kategóriacímkéket használunk.
""")

code(r"""
df = df_raw.drop_duplicates().reset_index(drop=True)

# Csak a megjelenítéshez: magyar címkék
df_viz = df.assign(
    dohanyzo=df["smoker"].map({"yes": "igen", "no": "nem"}),
    nem=df["sex"].map({"female": "nő", "male": "férfi"}),
    regio=df["region"].map({"northeast": "északkelet", "northwest": "északnyugat",
                            "southeast": "délkelet", "southwest": "délnyugat"}),
)
print(df.shape)
""")

md(r"""
### 3.1 A célváltozó (`charges`) eloszlása
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sns.histplot(df["charges"], bins=40, kde=True, ax=axes[0], color="#4C72B0")
axes[0].axvline(df["charges"].mean(), color="red", ls="--", label=f"átlag = {df['charges'].mean():,.0f}")
axes[0].axvline(df["charges"].median(), color="black", ls=":", label=f"medián = {df['charges'].median():,.0f}")
axes[0].set(title="Az éves költség (charges) eloszlása", xlabel="Éves költség (USD)", ylabel="Gyakoriság")
axes[0].legend()

sns.histplot(np.log(df["charges"]), bins=40, kde=True, ax=axes[1], color="#55A868")
axes[1].set(title="A log(charges) eloszlása", xlabel="ln(éves költség)", ylabel="Gyakoriság")

plt.tight_layout()
save_fig(fig, "01_charges_eloszlas")
plt.show()

print(f"Ferdeség (charges):      {df['charges'].skew():.3f}")
print(f"Ferdeség (log(charges)): {np.log(df['charges']).skew():.3f}")
""")

md(r"""
**Értelmezés:** A költségek eloszlása erősen **jobbra ferde** (ferdeség ≈ 1,5). A legtöbb ember évi 1 000–15 000 USD között van, de van egy hosszú „farok” 60 000 USD-ig. A logaritmikus transzformáció után az eloszlás közel szimmetrikus (ferdeség ≈ −0,1), viszont több csúcs látszik. Ez arra utal, hogy a mintában **eltérő költségszintű alcsoportok** vannak. A 3.5 fejezetben kiderül, hogy ezeket főleg a dohányzás okozza.
""")

md(r"""
### 3.2 Az életkor és a BMI eloszlása
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sns.histplot(df["age"], bins=range(18, 66, 2), ax=axes[0], color="#8172B3")
axes[0].set(title="Az életkor eloszlása", xlabel="Életkor (év)", ylabel="Gyakoriság")

sns.histplot(df["bmi"], bins=35, kde=True, ax=axes[1], color="#DD8452")
for x, lab in [(18.5, "18,5"), (25, "25"), (30, "30")]:
    axes[1].axvline(x, color="gray", ls="--", lw=1)
    axes[1].text(x + 0.3, axes[1].get_ylim()[1] * 0.92, lab, color="gray")
axes[1].set(title="A BMI eloszlása (szaggatott: WHO-határok)", xlabel="BMI (kg/m²)", ylabel="Gyakoriság")

plt.tight_layout()
save_fig(fig, "02_kor_bmi_eloszlas")
plt.show()
""")

md(r"""
**Értelmezés:** Az életkor nagyjából **egyenletesen** oszlik el 18 és 64 év között. Kivétel a 18–19 évesek csoportja, amely felülreprezentált. A BMI **közel normális eloszlású**, a csúcsa 30 körül van. A biztosítottak több mint fele a WHO szerint elhízott (BMI ≥ 30), és csak kevesen soványak (BMI < 18,5).
""")

md(r"""
### 3.3 A kategorikus változók gyakoriságai
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
cat_plots = [("nem", "Nem"), ("dohanyzo", "Dohányzik"), ("regio", "Régió"), ("children", "Gyermekek száma")]

for ax, (col, title) in zip(axes, cat_plots):
    counts = df_viz[col].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4C72B0")
    ax.set(title=title, xlabel="", ylabel="Darabszám")
    ax.bar_label(ax.containers[0], fontsize=9)
    ax.tick_params(axis="x", rotation=30 if col == "regio" else 0)

plt.tight_layout()
save_fig(fig, "03_kategorikus_gyakorisagok")
plt.show()
""")

md(r"""
**Értelmezés:** A nemek és a négy régió **kiegyensúlyozottan** szerepelnek, a délkeleti régió csak kicsit nagyobb. A dohányzók aránya kb. **20%** (274 fő), tehát ez a csoport kisebb, de egy-egy modell még elég megfigyelésből tud tanulni rajta. A gyermekszám 0 és 5 között van; 4 és 5 gyermek nagyon ritka.
""")

md(r"""
### 3.4 A költségek kategóriánként
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

sns.boxplot(data=df_viz, x="dohanyzo", y="charges", hue="dohanyzo", palette=SMOKER_PALETTE,
            order=["nem", "igen"], legend=False, ax=axes[0])
axes[0].set(title="Költség dohányzás szerint", xlabel="Dohányzik", ylabel="Éves költség (USD)")

sns.boxplot(data=df_viz, x="nem", y="charges", ax=axes[1], color="#8172B3")
axes[1].set(title="Költség nem szerint", xlabel="Nem", ylabel="")

sns.boxplot(data=df_viz, x="regio", y="charges", ax=axes[2], color="#55A868")
axes[2].set(title="Költség régió szerint", xlabel="Régió", ylabel="")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
save_fig(fig, "04_koltseg_kategoriankent")
plt.show()

df_viz.groupby("dohanyzo")["charges"].agg(["count", "mean", "median"])
""")

md(r"""
**Értelmezés:** A **dohányzás** a legerősebb tényező: a dohányzók medián költsége (~34 500 USD) közel ötszöröse a nemdohányzókénak (~7 300 USD). A nemek és a régiók között a mediánok szinte azonosak. A dobozok hasonló szélessége azt mutatja, hogy e két változó önmagában alig befolyásolja a költséget. Ezt a 6. fejezetben statisztikai próbával is ellenőrizzük. A nemdohányzóknál sok felső „kiugró” pont látszik. Ezekkel a 4. fejezetben foglalkozunk.
""")

md(r"""
### 3.5 Az életkor és a BMI kapcsolata a költséggel (dohányzás szerint színezve)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.scatterplot(data=df_viz, x="age", y="charges", hue="dohanyzo", palette=SMOKER_PALETTE,
                hue_order=["nem", "igen"], alpha=0.6, s=25, ax=axes[0])
axes[0].set(title="Életkor és költség", xlabel="Életkor (év)", ylabel="Éves költség (USD)")
axes[0].legend(title="Dohányzik")

sns.scatterplot(data=df_viz, x="bmi", y="charges", hue="dohanyzo", palette=SMOKER_PALETTE,
                hue_order=["nem", "igen"], alpha=0.6, s=25, ax=axes[1])
axes[1].axvline(30, color="gray", ls="--", lw=1)
axes[1].set(title="BMI és költség (szaggatott: BMI = 30)", xlabel="BMI (kg/m²)", ylabel="")
axes[1].legend(title="Dohányzik")

plt.tight_layout()
save_fig(fig, "05_scatter_kor_bmi")
plt.show()
""")

md(r"""
**Értelmezés:**

- **Életkor:** az életkorral nagyjából lineárisan (enyhén gyorsulva) nőnek a költségek. Három párhuzamos „sáv” látszik: az alsó szinte csak nemdohányzókból, a felső csak dohányzókból áll, a középsőben vegyesen vannak.
- **BMI:** nemdohányzóknál a BMI-nek alig van hatása. Dohányzóknál viszont a **BMI = 30 határnál ugrásszerűen megnő** a költség, kb. 20 000 USD-ről 40 000 USD fölé. Ez erős **kölcsönhatás** (interakció) a dohányzás és az elhízás között. Egy egyszerű lineáris modell ezt magától nem tudja megtanulni, ezért az 5. fejezetben külön jellemzőként adjuk hozzá.
""")

md(r"""
### 3.6 Korrelációs hőtérkép

A korrelációhoz a bináris változókat 0/1-es kódolással vesszük be (`smoker_yes`, `sex_male`).
""")

code(r"""
corr_df = df[["age", "bmi", "children", "charges"]].assign(
    smoker_yes=(df["smoker"] == "yes").astype(int),
    sex_male=(df["sex"] == "male").astype(int),
)
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax,
            cbar_kws={"label": "Pearson-korreláció"})
ax.set_title("Korrelációs mátrix")
plt.tight_layout()
save_fig(fig, "06_korrelacios_hoterkep")
plt.show()
""")

md(r"""
**Értelmezés:** A `charges` legerősebben a **dohányzással** korrelál (r ≈ 0,79), ezt az **életkor** (≈ 0,30) és a **BMI** (≈ 0,20) követi. A gyermekszám és a nem korrelációja elhanyagolható. A magyarázó változók egymás között szinte korrelálatlanok, ezért **multikollinearitás nem okoz gondot** az eredeti jellemzőknél. Fontos: a Pearson-korreláció csak lineáris kapcsolatot mér. A dohányzás × BMI kölcsönhatás ebben a mátrixban nem látszik.
""")

# =====================================================================
# 4. Előfeldolgozás
# =====================================================================
md(r"""
## 4. Szűrés, előfeldolgozás

### 4.1 Duplikátumok eltávolítása

Egy teljesen azonos sorpárt találtunk (19 éves, nemdohányzó férfi, azonos BMI-vel és költséggel). Mivel nincs egyedi azonosító, nem tudjuk biztosan, hogy két különböző személyről van-e szó. Ennek az esélye azonban kicsi (a költség 4 tizedesjegyig egyezik), ezért **a duplikátumot eltávolítjuk**. Ezt a lépést a 3. fejezet elején már elvégeztük (`df`).
""")

code(r"""
print(f"Eredeti sorok: {len(df_raw)}, duplikátum nélkül: {len(df)}")
""")

md(r"""
### 4.2 Kiugró értékek vizsgálata

A kiugró értékeket a szokásos **IQR-szabály** szerint keressük (Q3 + 1,5·IQR fölött). Először a teljes mintán, aztán dohányzási csoportonként.
""")

code(r"""
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    return s > upper, upper

mask_all, upper_all = iqr_outliers(df["charges"])
print(f"Teljes minta: felső határ = {upper_all:,.0f} USD, kiugró pontok: {mask_all.sum()} db")
print("Ebből dohányzó:", df.loc[mask_all, "smoker"].value_counts().to_dict())

for grp, sub in df.groupby("smoker"):
    m, up = iqr_outliers(sub["charges"])
    print(f"  Csoporton belül (smoker={grp}): határ = {up:,.0f} USD, kiugró: {m.sum()} db, "
          f"legnagyobb érték: {sub['charges'].max():,.0f} USD")
""")

md(r"""
**Döntés: a kiugró értékeket megtartjuk.** Indoklás:

1. A teljes mintán „kiugrónak” jelölt pontok döntő többsége **dohányzó**. Ezek nem hibás adatok, hanem egy valódi, magas költségű alcsoport tagjai. Ha eltávolítanánk őket, éppen a legfontosabb mintázatot veszítenénk el.
2. Csoportonként vizsgálva a dohányzóknál nincs kiugró pont. A nemdohányzóknál 46 van (kb. 4%), de ezek is reális értékek: a legnagyobb is 40 000 USD alatt van, ami egy kórházi kezeléssel járó év költségének felel meg. Valószínűleg egy-egy súlyosabb betegséghez tartoznak, és az adatban nincs olyan változó, amely alapján hibásnak minősíthetnénk őket.
3. Nincs jele lehetetlen értéknek (negatív költség, irreális életkor vagy BMI).

A ferdeség kezelésére eltávolítás helyett **log-transzformált célváltozót** is kipróbálunk (5. és 7. fejezet).
""")

md(r"""
### 4.3 Kategorikus változók kódolása

A `sex`, `smoker` és `region` változókat **one-hot kódolással** alakítjuk numerikussá. A lineáris modelleknél a `drop="first"` beállítással egy-egy referenciakategóriát elhagyunk, így elkerüljük a tökéletes multikollinearitást („dummy-csapda”). A kódolást nem kézzel, hanem egy `scikit-learn` **Pipeline**-on belül végezzük. Így a kódoló csak a tanítóhalmazon tanul, és a keresztvalidáció minden hajtásában szabályosan ismétlődik (nincs adatszivárgás).

A kódolás eredménye szemléltetésképpen:
""")

code(r"""
pd.get_dummies(df.head(), columns=["sex", "smoker", "region"], drop_first=True, dtype=int)
""")

md(r"""
### 4.4 Tanító- és teszthalmaz

Az adatokat **80/20 arányban** osztjuk fel (`random_state=42`). A felosztást a `smoker` változó szerint **rétegezzük**. Így a ritkább (20%-os) dohányzó csoport aránya mindkét halmazban azonos, és a teszteredmény kevésbé függ attól, hány dohányzó kerül véletlenül a tesztbe. **Minden modell ugyanezt a felosztást használja.** A teszthalmazt csak a végső kiértékelésnél használjuk; a modellek összevetéséhez a tanítóhalmazon 5-szörös keresztvalidációt is futtatunk.
""")

code(r"""
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["smoker"]
)
print(f"Tanító: {len(train_df)} sor, teszt: {len(test_df)} sor")
print(f"Dohányzók aránya – tanító: {(train_df['smoker'] == 'yes').mean():.3f}, "
      f"teszt: {(test_df['smoker'] == 'yes').mean():.3f}")
""")

# =====================================================================
# 5. Feature engineering
# =====================================================================
md(r"""
## 5. Jellemzők kinyerése (feature engineering)

Az EDA alapján a következő új jellemzőket hozzuk létre:

| Új jellemző | Definíció | Indoklás |
|---|---|---|
| `age_sq` | életkor² | Az életkor–költség kapcsolat enyhén gyorsul (3.5 ábra). A négyzetes tag ezt a görbületet egy lineáris modellben is leírhatóvá teszi. |
| `bmi_category` | sovány (<18,5) / normál (18,5–25) / túlsúlyos (25–30) / elhízott (≥30) | A WHO-kategóriák orvosilag értelmezhetők, és a költség a BMI=30 határnál ugrik, nem folytonosan nő. |
| `obese_smoker` | 1, ha BMI ≥ 30 **és** dohányzik | A 3.5 ábra legfeltűnőbb mintázata: az elhízott dohányzók külön, nagyon magas költségű csoportot alkotnak. |
| `smoker_bmi` | dohányzik (0/1) × BMI | Interakció: a BMI meredeksége dohányzóknál sokkal nagyobb (lásd H4 a 6. fejezetben). |
| `age_group` | 18–29 / 30–39 / 40–49 / 50–64 | Könnyen értelmezhető korcsoportok. A lineáris modellnek is segít, ha a hatás nem pontosan lineáris. |
| `log(charges)` | a célváltozó természetes logaritmusa | A célváltozó ferde; a log-skálán a hibák kevésbé függnek a költség nagyságától. Ezt a célváltozó-változatot a 7. fejezetben külön modellel próbáljuk ki. |

Mindegyik jellemző **soronként**, csak az adott sor értékeiből számolható. Ezért nem okoz adatszivárgást, ha a tanító- és teszthalmazra külön-külön alkalmazzuk.
""")

code(r"""
def add_features(data):
    out = data.copy()
    is_smoker = (out["smoker"] == "yes").astype(int)
    out["age_sq"] = out["age"] ** 2
    out["bmi_category"] = pd.cut(
        out["bmi"], bins=[0, 18.5, 25, 30, np.inf], right=False,
        labels=["sovány", "normál", "túlsúlyos", "elhízott"],
    ).astype(str)
    out["obese_smoker"] = ((out["bmi"] >= 30) & (is_smoker == 1)).astype(int)
    out["smoker_bmi"] = is_smoker * out["bmi"]
    out["age_group"] = pd.cut(
        out["age"], bins=[17, 29, 39, 49, 64], labels=["18-29", "30-39", "40-49", "50-64"]
    ).astype(str)
    return out


train_fe = add_features(train_df)
test_fe = add_features(test_df)
df_fe = add_features(df)  # csak az ábrákhoz és a hipotézisvizsgálathoz
train_fe.head()
""")

md(r"""
### Az új jellemzők ellenőrzése ábrán

Nézzük meg, mennyire különülnek el a költségek a dohányzás és az elhízás négy kombinációjában.
""")

code(r"""
group_labels = {
    (0, 0): "nemdohányzó,\nBMI < 30",
    (0, 1): "nemdohányzó,\nBMI ≥ 30",
    (1, 0): "dohányzó,\nBMI < 30",
    (1, 1): "dohányzó,\nBMI ≥ 30",
}
plot_df = df_fe.assign(
    csoport=[group_labels[(int(s == "yes"), int(b >= 30))] for s, b in zip(df_fe["smoker"], df_fe["bmi"])]
)
order = list(group_labels.values())

fig, ax = plt.subplots(figsize=(9, 4.8))
sns.boxplot(data=plot_df, x="csoport", y="charges", order=order, ax=ax,
            hue="csoport", hue_order=order, palette=["#4C72B0", "#64B5CD", "#DD8452", "#C44E52"], legend=False)
ax.set(title="Költség a dohányzás és az elhízás kombinációja szerint", xlabel="", ylabel="Éves költség (USD)")
plt.tight_layout()
save_fig(fig, "07_dohanyzas_x_elhizas")
plt.show()

plot_df.groupby("csoport")["charges"].agg(["count", "median"]).reindex(order)
""")

md(r"""
**Értelmezés:** Az elhízás a nemdohányzóknál alig emeli a medián költséget. A dohányzóknál viszont kb. **kétszeresére** növeli (~20 000 → ~41 000 USD). Ez megerősíti, hogy az `obese_smoker` és a `smoker_bmi` jellemző olyan információt hordoz, amelyet az eredeti változók *additív* kombinációja nem tud leírni.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
age_order = ["18-29", "30-39", "40-49", "50-64"]
sns.boxplot(data=df_fe.assign(dohanyzo=df_viz["dohanyzo"]), x="age_group", y="charges", hue="dohanyzo",
            order=age_order, hue_order=["nem", "igen"], palette=SMOKER_PALETTE, ax=ax)
ax.set(title="Költség korcsoportonként", xlabel="Korcsoport (év)", ylabel="Éves költség (USD)")
ax.legend(title="Dohányzik")
plt.tight_layout()
save_fig(fig, "08_korcsoportok")
plt.show()
""")

md(r"""
**Értelmezés:** Mindkét csoportban egyenletesen nő a költség a korcsoportokkal. A dohányzók és nemdohányzók közötti „ugrás” minden korcsoportban nagyjából azonos nagyságú. A kor és a dohányzás hatása tehát **közel additív**, nem kell kor × dohányzás interakciót bevezetni.
""")

# =====================================================================
# 6. Hipotézisvizsgálat
# =====================================================================
md(r"""
## 6. Hipotézisvizsgálat

Négy hipotézist vizsgálunk a duplikátummentes teljes mintán (n = 1337). A szignifikanciaszint **α = 0,05**.

**A próbák megválasztása:** a `charges` eloszlása erősen ferde, és a csoportonkénti szórások nagyon különbözők. Ezért nem a klasszikus (egyenlő szórású) t-próbát használjuk, hanem:

- a **Welch-féle t-próbát** (nem feltételez egyenlő szórásokat; nagy mintán a centrális határeloszlás-tétel miatt az átlagokra még ferde eloszlás mellett is használható),
- és mellette a **nemparaméteres Mann–Whitney-, illetve Kruskal–Wallis-próbát**, amelyek a rangokat hasonlítják össze, ezért kevésbé érzékenyek a ferdeségre és a kiugró értékekre.

Ha a két megközelítés eltérő eredményt ad, azt külön értelmezzük. Először a normalitást ellenőrizzük.
""")

code(r"""
for grp, sub in df.groupby("smoker"):
    w, p = stats.shapiro(sub["charges"])
    print(f"Shapiro–Wilk (smoker={grp}): W = {w:.3f}, p = {p:.2e}")
""")

md(r"""
A Shapiro–Wilk-próba mindkét csoportban elveti a normalitást (p ≪ 0,05). Ezért a nemparaméteres próbák eredményét tekintjük **elsődlegesnek**, a Welch-próba csak kiegészítő információ.
""")

code(r"""
ALPHA = 0.05


def report(name, test_name, stat, p):
    decision = "H0 elvetése" if p < ALPHA else "H0 nem vethető el"
    return {"hipotézis": name, "próba": test_name, "statisztika": round(float(stat), 3),
            "p-érték": f"{p:.2e}", "döntés (α=0,05)": decision}


results_h = []
""")

md(r"""
### H1: A dohányzók költsége szignifikánsan magasabb

- **H0:** a dohányzók és nemdohányzók költségeloszlása azonos (a dohányzóké nem nagyobb).
- **H1:** a dohányzók költsége nagyobb (egyoldalú próba).
""")

code(r"""
smk = df.loc[df["smoker"] == "yes", "charges"]
non = df.loc[df["smoker"] == "no", "charges"]

t1 = stats.ttest_ind(smk, non, equal_var=False, alternative="greater")
u1 = stats.mannwhitneyu(smk, non, alternative="greater")
results_h += [report("H1 dohányzás", "Welch t (egyoldalú)", t1.statistic, t1.pvalue),
              report("H1 dohányzás", "Mann–Whitney U (egyoldalú)", u1.statistic, u1.pvalue)]
print(f"Átlag: dohányzó {smk.mean():,.0f} USD vs. nemdohányzó {non.mean():,.0f} USD")
pd.DataFrame(results_h[-2:])
""")

md(r"""
**Következtetés:** Mindkét próba p-értéke gyakorlatilag nulla (< 10⁻¹⁰⁰). **H0-t elvetjük:** a dohányzók költsége szignifikánsan és nagyon jelentősen magasabb. Az átlagos különbség kb. 23 600 USD.
""")

md(r"""
### H2: A nemek között nincs szignifikáns különbség

- **H0:** a nők és a férfiak költségeloszlása azonos.
- **H1:** a két nem költségeloszlása különbözik (kétoldalú próba).
""")

code(r"""
male = df.loc[df["sex"] == "male", "charges"]
female = df.loc[df["sex"] == "female", "charges"]

t2 = stats.ttest_ind(male, female, equal_var=False)
u2 = stats.mannwhitneyu(male, female, alternative="two-sided")
results_h += [report("H2 nem", "Welch t (kétoldalú)", t2.statistic, t2.pvalue),
              report("H2 nem", "Mann–Whitney U (kétoldalú)", u2.statistic, u2.pvalue)]
print(f"Átlag: férfi {male.mean():,.0f} USD vs. nő {female.mean():,.0f} USD")
print(f"Medián: férfi {male.median():,.0f} USD vs. nő {female.median():,.0f} USD")
print("Dohányzók aránya nemenként:", df.groupby("sex")["smoker"].apply(lambda s: round((s == "yes").mean(), 3)).to_dict())
pd.DataFrame(results_h[-2:])
""")

code(r"""
# Kontroll: a nemek összehasonlítása külön a nemdohányzók és a dohányzók között
rows = []
for grp, sub in df.groupby("smoker"):
    u = stats.mannwhitneyu(sub.loc[sub["sex"] == "male", "charges"],
                           sub.loc[sub["sex"] == "female", "charges"], alternative="two-sided")
    rows.append({"smoker": grp, "Mann–Whitney p": round(u.pvalue, 3)})
pd.DataFrame(rows)
""")

md(r"""
**Következtetés:** A két próba itt **eltérő eredményt ad**. A Welch-próba az *átlagok* között kis, határon szignifikáns különbséget jelez (p ≈ 0,03; a férfiak átlaga ~1 400 USD-vel magasabb). A rangalapú, elsődlegesnek tekintett Mann–Whitney-próba szerint viszont nincs különbség (p ≈ 0,69), és a mediánok is szinte azonosak. Az átlagkülönbség oka, hogy **a férfiak között több a dohányzó** (~24% vs. ~17%), és a dohányzók magas költségei felhúzzák a férfiak átlagát. Ez a dohányzás *zavaró hatása* (confounding).

A kontrollcella árnyalja a képet. Ha a dohányzást rögzítjük, a dohányzók között nincs szignifikáns különbség (p ≈ 0,11). A nemdohányzók között viszont van egy kicsi, de szignifikáns eltérés (p ≈ 0,02), mégpedig **fordított irányban**: a nemdohányzó nők medián költsége ~650 USD-vel magasabb. Ez a ~7 000–8 000 USD-s szinthez képest kis különbség.

**Összegzés: a teljes mintán H0 nem vethető el.** A nemnek nincs egyirányú, gyakorlatilag jelentős hatása, a nyers átlagkülönbség pedig főleg a dohányzásból ered. A kis, csoporton belüli eltérés miatt a nemet nem hagyjuk ki a modellekből; a modell majd eldönti, mekkora súlyt kap.
""")

md(r"""
### H3: A régiók között nincs szignifikáns különbség

- **H0:** a négy régió költségeloszlása azonos.
- **H1:** legalább egy régió eltér a többitől.

Két próbát futtatunk: egyutas ANOVA (átlagok) és Kruskal–Wallis (rangok).
""")

code(r"""
region_groups = [g["charges"].values for _, g in df.groupby("region")]

f3 = stats.f_oneway(*region_groups)
k3 = stats.kruskal(*region_groups)
lev3 = stats.levene(*region_groups)
results_h += [report("H3 régió", "ANOVA (F)", f3.statistic, f3.pvalue),
              report("H3 régió", "Kruskal–Wallis (H)", k3.statistic, k3.pvalue)]
print(f"Levene-próba a szórások egyezésére: p = {lev3.pvalue:.4f}")
display(df.groupby("region")["charges"].agg(["mean", "median"]))
pd.DataFrame(results_h[-2:])
""")

md(r"""
**Következtetés:** Az ANOVA kis, határon szignifikáns különbséget mutat (p ≈ 0,03). Ezt főleg a délkeleti régió magasabb *átlaga* okozza, ahol a legmagasabb a dohányzók aránya és az átlagos BMI. Az ANOVA feltételei azonban sérülnek: az eloszlás nem normális, a Levene-próba szerint a szórások sem egyenlők. A rangalapú Kruskal–Wallis-próba szerint **nincs szignifikáns különbség** (p ≈ 0,20), és a mediánok is közel vannak egymáshoz. **H0 nem vethető el:** a régió önmagában nem szignifikáns tényező. Ezért a modellekben várhatóan csak kis szerepet kap.
""")

md(r"""
### H4: A BMI hatása dohányzóknál erősebb

- **H0:** a BMI és a költség közötti kapcsolat erőssége a két csoportban azonos.
- **H1:** a kapcsolat dohányzóknál erősebb.

Csoportonként kiszámítjuk a Pearson-korrelációt és a lineáris regressziós meredekséget. A két korreláció különbségét **Fisher-féle z-transzformációval** teszteljük.
""")

code(r"""
h4_rows = {}
for grp, sub in df.groupby("smoker"):
    r, p = stats.pearsonr(sub["bmi"], sub["charges"])
    lr = stats.linregress(sub["bmi"], sub["charges"])
    h4_rows[grp] = {"n": len(sub), "Pearson r": r, "r p-értéke": p,
                    "meredekség (USD / BMI-egység)": lr.slope, "meredekség SE": lr.stderr}
h4 = pd.DataFrame(h4_rows).T
display(h4)

# Két független korreláció összehasonlítása (Fisher z)
z_yes, z_no = np.arctanh(h4.loc["yes", "Pearson r"]), np.arctanh(h4.loc["no", "Pearson r"])
se = np.sqrt(1 / (h4.loc["yes", "n"] - 3) + 1 / (h4.loc["no", "n"] - 3))
z = (z_yes - z_no) / se
p_z = stats.norm.sf(z)  # egyoldalú
results_h.append(report("H4 BMI × dohányzás", "Fisher z (korrelációk, egyoldalú)", z, p_z))
print(f"Fisher z = {z:.2f}, egyoldalú p = {p_z:.2e}")
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(data=df[df["smoker"] == "no"], x="bmi", y="charges", ax=ax, label="nem dohányzik",
            color=SMOKER_PALETTE["nem"], scatter_kws={"alpha": 0.35, "s": 15})
sns.regplot(data=df[df["smoker"] == "yes"], x="bmi", y="charges", ax=ax, label="dohányzik",
            color=SMOKER_PALETTE["igen"], scatter_kws={"alpha": 0.5, "s": 15})
ax.set(title="A BMI hatása csoportonként (illesztett regressziós egyenes)",
       xlabel="BMI (kg/m²)", ylabel="Éves költség (USD)")
ax.legend()
plt.tight_layout()
save_fig(fig, "09_h4_bmi_meredekseg")
plt.show()
""")

md(r"""
**Következtetés:** Nemdohányzóknál a BMI és a költség között nagyon gyenge a kapcsolat (r ≈ 0,08; a meredekség ≈ 83 USD / BMI-egység). Dohányzóknál erős (r ≈ 0,81; a meredekség ≈ 1 470 USD / BMI-egység, kb. **18-szoros**). A Fisher-féle z-próba alapján a különbség erősen szignifikáns, **H0-t elvetjük**. Ez a statisztikai alátámasztása annak, hogy a `smoker_bmi` interakciós tagot bevesszük a modellbe.
""")

md(r"""
### A hipotézisvizsgálat összefoglalása
""")

code(r"""
pd.DataFrame(results_h)
""")

md(r"""
| Hipotézis | Eredmény |
|---|---|
| H1 – a dohányzók költsége magasabb | ✅ megerősítve (H0 elvetve) |
| H2 – nincs különbség a nemek között | ✅ a fő (rangalapú) próbával megerősítve (H0 nem vethető el); a nyers átlagkülönbség a dohányzás zavaró hatása, a nemdohányzók között kis fordított különbség van |
| H3 – nincs különbség a régiók között | ✅ megerősítve a robusztus próbával (H0 nem vethető el) |
| H4 – a BMI hatása dohányzóknál erősebb | ✅ megerősítve (H0 elvetve) |
""")

# =====================================================================
# 7. Modellezés
# =====================================================================
md(r"""
## 7. Modellezés

### 7.1 A modellek és a jellemzőkészletek

Hat modellt hasonlítunk össze, mindegyik **ugyanazon** a tanító/teszt felosztáson fut:

| # | Modell | Jellemzők | Miért? |
|---|---|---|---|
| 1 | **Baseline** (átlag) | – | Viszonyítási alap: mindig a tanítóhalmaz átlagát becsüli. Ennél jobbnak kell lennie minden modellnek. |
| 2 | **Lineáris regresszió** | eredeti 6 változó | Egyszerű, jól értelmezhető kiindulás. |
| 3 | **Lineáris regresszió + FE** | eredeti + kinyert jellemzők | Mutatja, mennyit javít a feature engineering *ugyanazon* a modellen. |
| 4 | **Ridge + FE** | eredeti + kinyert jellemzők | L2-regularizáció: a sok, részben átfedő jellemző miatt stabilabb együtthatókat adhat. |
| 5 | **Lin. regresszió + FE, log(charges)** | eredeti + kinyert jellemzők | A log-célváltozó-változat kipróbálása (a becslést visszatranszformáljuk USD-be). |
| 6 | **Döntési fa** (max. mélység 4) | eredeti 6 változó | Nemlineáris modell, amely a küszöböket (pl. BMI = 30) magától megtanulhatja; a kis mélység miatt még értelmezhető. |
| 7 | **Random Forest** | eredeti 6 változó | Sok fa átlaga: általában pontosabb és kevésbé túltanul, mint egy fa. |

A fa alapú modellek az eredeti jellemzőket kapják. Így az is kiderül, hogy a kézzel készített jellemzőket egy rugalmas modell magától meg tudja-e tanulni.

**Előfeldolgozás a pipeline-ban:**
- lineáris modelleknél: a numerikus jellemzők standardizálása (`StandardScaler`), a kategorikusak one-hot kódolása `drop="first"`-tel,
- fa alapú modelleknél: csak one-hot kódolás (a fák a skálázásra érzéketlenek).
""")

code(r"""
TARGET = "charges"

BASE_NUM = ["age", "bmi", "children"]
BASE_CAT = ["sex", "smoker", "region"]

FE_NUM = BASE_NUM + ["age_sq", "smoker_bmi"]
FE_BIN = ["obese_smoker"]
FE_CAT = BASE_CAT + ["bmi_category", "age_group"]


def linear_preprocessor(num, cat, passthrough=()):
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
        ("bin", "passthrough", list(passthrough)),
    ], verbose_feature_names_out=False)


def tree_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ], verbose_feature_names_out=False)


base_cols = BASE_NUM + BASE_CAT
fe_cols = FE_NUM + FE_BIN + FE_CAT

X_train_fe, X_test_fe = train_fe[fe_cols], test_fe[fe_cols]
y_train, y_test = train_fe[TARGET], test_fe[TARGET]
""")

md(r"""
A modelleket egy szótárban adjuk meg, mindegyikhez a használt jellemzőkkel együtt. A döntési fa mélységét (4) és a Random Forest beállításait (300 fa, levélenként legalább 5 minta) **nem a teszthalmazon hangoltuk**, hanem a túltanulás elkerülését szolgáló, szokásos konzervatív értékek. A Ridge `alpha` értékét a tanítóhalmazon, keresztvalidációval választjuk ki (lásd lent).
""")

code(r"""
# Ridge alpha kiválasztása 5-szörös CV-vel, csak a tanítóhalmazon
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

alphas = [0.01, 0.1, 1, 3, 10, 30, 100]
ridge_cv = []
for a in alphas:
    pipe = Pipeline([("prep", linear_preprocessor(FE_NUM, FE_CAT, FE_BIN)), ("model", Ridge(alpha=a))])
    scores = cross_validate(pipe, X_train_fe, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    ridge_cv.append({"alpha": a, "CV RMSE": -scores["test_score"].mean()})
ridge_cv = pd.DataFrame(ridge_cv)
best_alpha = ridge_cv.loc[ridge_cv["CV RMSE"].idxmin(), "alpha"]
print(f"Legjobb alpha: {best_alpha}")
ridge_cv
""")

code(r"""
models = {
    "Baseline (átlag)": (
        DummyRegressor(strategy="mean"), base_cols),
    "Lineáris regresszió": (
        Pipeline([("prep", linear_preprocessor(BASE_NUM, BASE_CAT)), ("model", LinearRegression())]), base_cols),
    "Lineáris regresszió + FE": (
        Pipeline([("prep", linear_preprocessor(FE_NUM, FE_CAT, FE_BIN)), ("model", LinearRegression())]), fe_cols),
    "Ridge + FE": (
        Pipeline([("prep", linear_preprocessor(FE_NUM, FE_CAT, FE_BIN)), ("model", Ridge(alpha=best_alpha))]), fe_cols),
    "Lin. regr. + FE, log(charges)": (
        TransformedTargetRegressor(
            regressor=Pipeline([("prep", linear_preprocessor(FE_NUM, FE_CAT, FE_BIN)), ("model", LinearRegression())]),
            func=np.log, inverse_func=np.exp), fe_cols),
    "Döntési fa (mélység 4)": (
        Pipeline([("prep", tree_preprocessor(BASE_NUM, BASE_CAT)),
                  ("model", DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE))]), base_cols),
    "Random Forest": (
        Pipeline([("prep", tree_preprocessor(BASE_NUM, BASE_CAT)),
                  ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                  random_state=RANDOM_STATE, n_jobs=-1))]), base_cols),
}
""")

md(r"""
### 7.2 Tanítás, keresztvalidáció és tesztelés

Minden modellre:
1. **5-szörös keresztvalidáció** a tanítóhalmazon (ugyanazok a hajtások): megmutatja, mennyire stabil a teljesítmény a különböző részhalmazokon,
2. **tanítás** a teljes tanítóhalmazon,
3. **kiértékelés** a teszthalmazon, amelyet eddig egyik modell sem látott.

Mérőszámok:
- **MAE** (átlagos abszolút hiba, USD): átlagosan ennyit téved a modell,
- **RMSE** (négyzetes középhiba gyöke, USD): a nagy hibákat jobban bünteti,
- **R²**: a célváltozó szórásának mekkora részét magyarázza a modell (1 = tökéletes, 0 = a baseline szintje).
""")

code(r"""
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rows, fitted, test_preds = [], {}, {}
for name, (model, cols) in models.items():
    cv_res = cross_validate(model, train_fe[cols], y_train, cv=cv,
                            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    model.fit(train_fe[cols], y_train)
    pred_train = model.predict(train_fe[cols])
    pred = model.predict(test_fe[cols])
    fitted[name], test_preds[name] = model, pred
    rows.append({
        "modell": name,
        "CV MAE": -cv_res["test_neg_mean_absolute_error"].mean(),
        "CV RMSE": -cv_res["test_neg_root_mean_squared_error"].mean(),
        "CV R² (átlag)": cv_res["test_r2"].mean(),
        "CV R² (szórás)": cv_res["test_r2"].std(),
        "tanító R²": r2_score(y_train, pred_train),
        "teszt MAE": mean_absolute_error(y_test, pred),
        "teszt RMSE": rmse(y_test, pred),
        "teszt R²": r2_score(y_test, pred),
    })

results = pd.DataFrame(rows).set_index("modell")
print("A modellek tanítása kész.")
""")

# =====================================================================
# 8. Kiértékelés
# =====================================================================
md(r"""
## 8. Kiértékelés és összehasonlítás

### 8.1 Összefoglaló táblázat

A táblázat a teszthalmazon mért RMSE szerint növekvő sorrendben mutatja a modelleket. A legjobb értékek ki vannak emelve.
""")

code(r"""
results_sorted = results.sort_values("teszt RMSE")
fmt = {c: "{:,.0f}" for c in ["CV MAE", "CV RMSE", "teszt MAE", "teszt RMSE"]}
fmt.update({c: "{:.3f}" for c in ["CV R² (átlag)", "CV R² (szórás)", "tanító R²", "teszt R²"]})
(results_sorted.style.format(fmt)
    .highlight_min(subset=["CV MAE", "CV RMSE", "teszt MAE", "teszt RMSE"], color="#c7e9c0")
    .highlight_max(subset=["CV R² (átlag)", "teszt R²"], color="#c7e9c0"))
""")

code(r"""
results_sorted.round(3).to_csv(FIG_DIR / "modell_eredmenyek.csv")
""")

md(r"""
### 8.2 A modellek összehasonlítása oszlopdiagramon
""")

code(r"""
plot_res = results_sorted.reset_index()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(data=plot_res, y="modell", x="teszt RMSE", ax=axes[0], color="#4C72B0")
axes[0].bar_label(axes[0].containers[0], fmt="{:,.0f}", padding=3, fontsize=9)
axes[0].set(title="Teszt RMSE (kisebb a jobb)", xlabel="RMSE (USD)", ylabel="")
axes[0].set_xlim(0, plot_res["teszt RMSE"].max() * 1.18)

r2_long = plot_res.melt(id_vars="modell", value_vars=["CV R² (átlag)", "teszt R²"],
                        var_name="mérés", value_name="R²")
sns.barplot(data=r2_long, y="modell", x="R²", hue="mérés", ax=axes[1], palette=["#8DA0CB", "#55A868"])
for c in axes[1].containers:
    axes[1].bar_label(c, fmt="{:.3f}", padding=3, fontsize=8)
axes[1].set(title="R²: 5-szörös CV átlag és teszt (nagyobb a jobb)", xlabel="R²", ylabel="")
axes[1].set_xlim(-0.15, 1.08)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].legend(title="", loc="lower right")

plt.tight_layout()
save_fig(fig, "10_modell_osszehasonlitas")
plt.show()
""")

md(r"""
**Értelmezés:**

- A **baseline** R²-je ≈ 0 (definíció szerint), az RMSE-je ~12 000 USD. Ehhez viszonyítva a log-célváltozós modell kivételével minden modell sokat javít.
- Az **egyszerű lineáris regresszió** CV R²-je ≈ 0,72, a teszten ≈ 0,82. A dohányzás, a kor és a BMI fő hatása tehát lineárisan is jól leírható.
- A **feature engineering** ugyanazon a lineáris modellen nagy ugrást hoz: a CV R² 0,72-ről **0,85-re**, a teszt R² 0,82-ről **0,92-re** nő, a teszt MAE pedig ~3 600-ról ~2 000 USD-re csökken. Ez főleg az `obese_smoker` és a `smoker_bmi` jellemzőnek köszönhető. A Ridge-regularizáció itt alig változtat, mert kevés a jellemző, és a minta elég nagy.
- A **log-célváltozós** modell meglepően gyenge (R² ≈ 0,5). A log-skálán a lineáris modell *szorzó* hatásokat feltételez, így az idős, elhízott dohányzóknál a hatások összeszorzódnak, és visszatranszformálás után erősen **felülbecsül** (a legnagyobb becslés ~96 000 USD, miközben a legnagyobb valós érték ~53 000 USD). Ezen az adathalmazon a költségek inkább *additívan* épülnek fel, ezért a log-transzformáció itt nem segít.
- A **döntési fa** (teszt R² ≈ 0,90) és a **Random Forest** (≈ 0,91) a kézzel készített jellemzők nélkül is majdnem ugyanezt a szintet éri el. A dohányzás × BMI kölcsönhatást a fák a küszöbökkel maguk is megtanulják. A Random Forest tanító R²-je (0,90) kicsit nagyobb a CV-énél (0,84), ami enyhe túltanulásra utal.
- Minden modell a teszthalmazon kicsit jobban teljesít, mint a CV-ben, vagyis ez a teszthalmaz valamivel „könnyebb” az átlagnál. A **CV-átlagot tekintjük megbízhatóbb becslésnek** (R² ≈ 0,85 a legjobb modellekre). A modellek **sorrendje** viszont a CV-ben és a teszten azonos, és a CV R² szórása kicsi (~0,04). Az összehasonlítás tehát stabil.
""")

md(r"""
### 8.3 A legjobb modell: előrejelzett vs. valós értékek

A legjobb modellt a teszt RMSE alapján választjuk ki automatikusan.
""")

code(r"""
best_name = results_sorted.index[0]
best_pred = test_preds[best_name]
print(f"Legjobb modell (teszt RMSE alapján): {best_name}")

fig, ax = plt.subplots(figsize=(6.5, 6))
colors = test_df["smoker"].map({"yes": SMOKER_PALETTE["igen"], "no": SMOKER_PALETTE["nem"]})
ax.scatter(y_test, best_pred, c=colors, alpha=0.65, s=22, edgecolor="none")
lims = [0, max(y_test.max(), best_pred.max()) * 1.05]
ax.plot(lims, lims, color="black", ls="--", lw=1, label="tökéletes becslés (y = x)")
ax.scatter([], [], c=SMOKER_PALETTE["nem"], label="nem dohányzik")
ax.scatter([], [], c=SMOKER_PALETTE["igen"], label="dohányzik")
ax.set(xlim=lims, ylim=lims, title=f"Előrejelzett vs. valós költség\n({best_name}, teszthalmaz)",
       xlabel="Valós éves költség (USD)", ylabel="Előrejelzett éves költség (USD)")
ax.legend(loc="upper left")
plt.tight_layout()
save_fig(fig, "11_legjobb_modell_pred_vs_valos")
plt.show()
""")

md(r"""
**Értelmezés:** A pontok többsége az átló mentén van, tehát a modell a költségek nagyságrendjét jól eltalálja mindhárom költségsávban. A legtöbb nagy hiba az átló **alatt**, alacsony becslésnél jelentkezik: néhány nemdohányzónak a vártnál sokkal magasabb a költsége. Ezek valószínűleg a 4.2 fejezetben látott, előre nem jelezhető (pl. betegség miatti) esetek. Az adathalmazban nincs olyan változó, amely ezeket megmagyarázná.
""")

md(r"""
### 8.4 Reziduum-elemzés
""")

code(r"""
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].scatter(best_pred, residuals, c=colors, alpha=0.65, s=22, edgecolor="none")
axes[0].axhline(0, color="black", ls="--", lw=1)
axes[0].set(title="Reziduumok az előrejelzett érték függvényében",
            xlabel="Előrejelzett éves költség (USD)", ylabel="Reziduum = valós − becsült (USD)")

sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color="#8172B3")
axes[1].axvline(0, color="black", ls="--", lw=1)
axes[1].set(title="A reziduumok eloszlása", xlabel="Reziduum (USD)", ylabel="Gyakoriság")

plt.tight_layout()
save_fig(fig, "12_reziduumok")
plt.show()

print(f"Reziduumok mediánja: {np.median(residuals):,.0f} USD, átlaga: {residuals.mean():,.0f} USD")
print(f"A tesztesetek {np.mean(np.abs(residuals) < 2000):.0%}-ánál 2 000 USD-nél kisebb a hiba.")
""")

md(r"""
**Értelmezés:** A reziduumok nagy része egy szűk sávban van: a tesztesetek kb. háromnegyedénél a hiba 2 000 USD alatt marad. Az eloszlás azonban **jobbra ferde**: van egy kisebb csoport nagy pozitív reziduummal, ahol a modell alulbecsül. Ezek a fent említett „váratlanul drága” esetek. Mivel a legkisebb négyzetes illesztés ezeket is igyekszik „lefedni”, a tipikus esetre a modell kicsit **felülbecsül**: a reziduumok mediánja kb. −1 400 USD. A reziduumok nem mutatnak erős szisztematikus trendet az előrejelzett érték függvényében, tehát a fő összefüggéseket a modell megtanulta. A megmaradt hiba főleg olyan **zaj**, amelyet a rendelkezésre álló hat változóból nem lehet megmagyarázni.
""")

md(r"""
### 8.5 Jellemzők fontossága (Random Forest)

A Random Forestnél **permutációs fontosságot** számolunk a teszthalmazon. Egy jellemző értékeit véletlenszerűen összekeverjük, és megnézzük, mennyivel romlik az R². Ez megbízhatóbb, mint a fák beépített (impurity-alapú) fontossága, mert nem részesíti előnyben a sok különböző értéket felvevő változókat.
""")

code(r"""
rf = fitted["Random Forest"]
perm = permutation_importance(rf, test_fe[base_cols], y_test, n_repeats=20,
                              random_state=RANDOM_STATE, scoring="r2")
imp = pd.DataFrame({"jellemző": base_cols, "fontosság": perm.importances_mean, "szórás": perm.importances_std})
imp = imp.sort_values("fontosság", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(imp["jellemző"], imp["fontosság"], xerr=imp["szórás"], color="#4C72B0")
ax.invert_yaxis()
ax.set(title="Random Forest – permutációs jellemzőfontosság (teszthalmaz)",
       xlabel="Az R² csökkenése a jellemző összekeverésekor", ylabel="")
plt.tight_layout()
save_fig(fig, "13_rf_jellemzofontossag")
plt.show()
imp.round(4)
""")

md(r"""
**Értelmezés:** A Random Forest szerint a **dohányzás** messze a legfontosabb jellemző, ezt a **BMI** és az **életkor** követi. A BMI fontossága nagyrészt a dohányzással való kölcsönhatásból ered (lásd H4). A nem, a régió és a gyermekszám fontossága elhanyagolható. Ez egybevág a hipotézisvizsgálat eredményeivel (H2, H3).
""")

md(r"""
### 8.6 A lineáris modell együtthatói (Lineáris regresszió + FE)

A regularizálatlan változatot mutatjuk, mert együtthatói közvetlenül értelmezhetők. A választott Ridge-modell (alpha = 1) együtthatói ettől alig térnek el, a teszteredményeik is gyakorlatilag azonosak.

A numerikus jellemzők standardizálva kerültek a modellbe, ezért együtthatóik azt mutatják, hány USD-vel változik a becsült költség, ha a jellemző **1 szórásnyit** nő. A bináris (one-hot) jellemzők együtthatója a referenciakategóriához viszonyított **USD-különbség**. Fontos: a `smoker_yes`, `smoker_bmi` és `obese_smoker` együtt írja le a dohányzás hatását, ezért ezeket nem külön-külön, hanem együtt kell értelmezni. Ugyanez igaz az `age`/`age_sq`/`age_group` hármasra.
""")

code(r"""
lin_fe = fitted["Lineáris regresszió + FE"]
feat_names = lin_fe.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lin_fe.named_steps["model"].coef_, index=feat_names).sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(coefs.index, coefs.values, color=np.where(coefs.values > 0, "#C44E52", "#4C72B0"))
ax.axvline(0, color="black", lw=0.8)
ax.set(title="Lineáris regresszió + FE – együtthatók", xlabel="Együttható (USD)", ylabel="")
plt.tight_layout()
save_fig(fig, "14_linearis_egyutthatok")
plt.show()

print(f"Tengelymetszet: {lin_fe.named_steps['model'].intercept_:,.0f} USD")
""")

code(r"""
# Gyakorlati értelmezés: becsült költségkülönbség egy 45 éves, nő, 0 gyermek, délkeleti, BMI = 33 példán
example = pd.DataFrame([
    {"age": 45, "sex": "female", "bmi": 33.0, "children": 0, "smoker": s, "region": "southeast", "charges": 0}
    for s in ["no", "yes"]
])
example_fe = add_features(example)
pred_ex = lin_fe.predict(example_fe[fe_cols])
print(f"Becsült költség – nemdohányzó: {pred_ex[0]:,.0f} USD, dohányzó: {pred_ex[1]:,.0f} USD "
      f"(különbség: {pred_ex[1] - pred_ex[0]:,.0f} USD)")
""")

md(r"""
**Értelmezés:**

- A legnagyobb együttható az **elhízott dohányzó** indikátoré (kb. +15 000 USD). Ezt a `smoker_bmi` interakció követi (kb. +6 600 USD szórásnyi növekedésenként). A dohányzás hatása tehát szinte teljes egészében ezen a két tagon keresztül jelenik meg, a `smoker_yes` önálló együtthatója ezért közel nulla. Ez jól mutatja, miért kell a három dohányzási tagot **együtt** értelmezni.
- Az **életkor** (`age`, `age_sq`) a második legfontosabb tényező. A korcsoport-dummyk negatív együtthatói csak kis korrekciók a folytonos életkorhatáshoz képest, mert a kettő átfedő információt hordoz.
- A nem, a régió és a nemdohányzók BMI-kategóriái csak néhány száz, legfeljebb ~1 500 USD-s eltolást adnak.
- A fenti példa az együttes hatást mutatja: ugyanannál a 45 éves, elhízott (BMI = 33) nőnél a dohányzás a becsült éves költséget kb. **32 000 USD-vel** növeli.
""")

# =====================================================================
# 9. Konklúzió
# =====================================================================
md(r"""
## 9. Konklúzió

### Mit tanultunk az adatokból?

- Az éves egészségügyi költséget három tényező határozza meg túlnyomó részben: a **dohányzás**, az **életkor** és a **BMI**. A nem, a régió és a gyermekszám hatása a vizsgálatok szerint elhanyagolható (H2, H3).
- A legfontosabb felismerés a **dohányzás × elhízás kölcsönhatás**. A BMI hatása dohányzóknál sokszorosa a nemdohányzókénak (H4), és az elhízott dohányzók külön, nagyon magas költségű csoportot alkotnak.
- A nyers átlagok félrevezetők lehetnek: a nemek közötti átlagkülönbség a dohányzás zavaró hatása, és robusztus próbával eltűnik.

### Melyik modell lett a legjobb és miért?

A legjobb eredményt a **feature engineeringgel bővített lineáris modellek** (Ridge + FE és Lineáris regresszió + FE, gyakorlatilag azonosan) érték el: CV R² ≈ 0,85, teszt R² ≈ 0,92, teszt MAE ≈ 2 000 USD. Szorosan mögöttük a **Random Forest** következik (CV R² ≈ 0,84, teszt R² ≈ 0,91). A különbség kicsi, a pontos számok a 8.1 táblázatban vannak. Az eredeti jellemzős lineáris regresszióhoz képest (teszt R² ≈ 0,82) a jellemzők kinyerése egyértelműen megérte. Ez a projekt fő tanulsága:

- Egy **egyszerű, átlátható lineáris modell** néhány jól megválasztott, az EDA-ból és a hipotézisvizsgálatból levezetett jellemzővel ugyanolyan jó lehet, mint egy „fekete doboz” ensemble modell.
- A lineáris modell előnye az **értelmezhetőség**: az együtthatókból közvetlenül leolvasható például, mennyivel növeli a költséget az elhízott dohányzó státusz. Biztosítási (szabályozott) környezetben ez fontos szempont.
- A Random Forest előnye, hogy **kézi jellemzők nélkül** is megtalálta a kölcsönhatásokat.

### Korlátok

- **Kis és egyszerű adathalmaz:** 1337 sor, 6 magyarázó változó. A diagnózisok, krónikus betegségek és korábbi költségek hiányoznak, pedig ezek magyarázhatnák a „váratlanul drága” nemdohányzó eseteket. A maradék hiba nagy része ilyen, a változókból nem megmagyarázható zaj.
- Az adatok **egyetlen időpontra** és amerikai biztosítottakra vonatkoznak, más ország vagy egészségügyi rendszer esetén nem általánosíthatók.
- A teszthalmaz kicsi (268 sor), így a modellek közötti kis különbségek (néhány század R²-pont) nem tekinthetők érdeminek. A teszt R² minden modellnél kicsit jobb a CV-nél, ezért a valószínűbb általánosító képességet a CV-átlag (≈ 0,85) írja le, nem a teszt R² (≈ 0,92).
- A `bmi` önbevalláson vagy egyszeri mérésen alapulhat, a dohányzás pedig bináris (nincs intenzitás).

### Továbbfejlesztési lehetőségek

- **Hiperparaméter-hangolás** (pl. `GridSearchCV`) a Random Forestre és a döntési fára, illetve **gradient boosting** modellek (pl. `HistGradientBoostingRegressor`) kipróbálása.
- **Kvantilis-regresszió** vagy előrejelzési intervallumok: biztosítási szempontból nemcsak a várható költség, hanem a kockázat (a felső kvantilisek) is érdekes.
- A **„váratlanul drága” esetek** külön vizsgálata, pl. egy osztályozó modellel, amely a nagyon magas költség valószínűségét becsli.
- **Gazdagabb adatok** (betegségtörténet, életmód, korábbi költségek) bevonása.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "projekt.ipynb")
print(f"projekt.ipynb elkészült ({len(cells)} cella).")
