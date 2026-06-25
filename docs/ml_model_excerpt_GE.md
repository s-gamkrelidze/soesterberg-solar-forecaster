# მანქანური სწავლების მოდელის გამოყენება

> ეს ფაილი მზადაა ნაშრომში (Word / Google Docs) ჩასაკოპირებლად: ქვემოთ მოცემული
> ტექსტი ჩასვით აბზაცად, ხოლო კოდის ბლოკი — monospace შრიფტით ან ეკრანის ანაბეჭდად (screenshot).

---

## მოდელის აღწერა

ენერგიის გამომუშავების საპროგნოზოდ გამოყენებულია **ზედამხედველობითი (supervised) მანქანური
სწავლების** მოდელი — **LightGBM** (გრადიენტული ბუსტინგი გადაწყვეტილების ხეებზე, *gradient boosting
over decision trees*). ამოცანა **რეგრესიაა**: უწყვეტი სამიზნე ცვლადის — სიმძლავრის კოეფიციენტის
(`capacity_factor`, ID003 ინვერტორის რეალურად გაზომილი გამომავალი) — პროგნოზირება.

მოდელი იყენებს **კვანტილურ რეგრესიას**: ერთი წერტილოვანი პროგნოზის ნაცვლად ვწვრთნით სამ მოდელს —
**q05, q50 და q95**. q50 იძლევა ცენტრალურ (მედიანურ) პროგნოზს, ხოლო q05 და q95 ქმნიან
**განუსაზღვრელობის დიაპაზონს** (prediction band), რაც პროგნოზის სანდოობის შეფასების საშუალებას იძლევა.

შესასვლელად მოდელს მიეწოდება **დაახლოებით 50 ნიშანი (feature)**: ფიზიკური სიდიდეები (მზის რადიაცია,
მზის სიმაღლე, ჰაერის მასა), clear-sky დეკომპოზიცია, ამინდის პარამეტრები (ღრუბლიანობა, ტემპერატურა,
ტენიანობა, ქარი), დროის ციკლური (Fourier) ნიშნულები და ისტორიული ლაგ-მახასიათებლები. სამიზნე
ცვლადის (`y`) არსებობა და მისი დაწყვილება შესასვლელ ნიშნებთან (`X`) სწორედ ის თვისებაა, რაც ამ მოდელს
ზედამხედველობით სწავლებად აქცევს.

**შედეგები (ტესტის ამონაკრები):** R² = **0.83**, MAE = **0.0513** CF (≈ ±0.115 kWh/სთ 2.25 kW AC სისტემაზე),
RMSE = **0.0996**.

---

## კოდის ფრაგმენტი — მოდელის სწავლება და შეფასება

```python
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# სამიზნე ცვლადი — სიმძლავრის კოეფიციენტი (ID003 ინვერტორის რეალური გაზომვა)
TARGET = "capacity_factor"

# შესასვლელი ნიშნები (features) — სულ ~50: ფიზიკა, ამინდი, clear-sky, დრო, ლაგები
FEATURES = [
    "solar_radiation", "solar_elevation", "air_mass",      # ფიზიკა
    "clearsky_ghi", "clearsky_index",                      # clear-sky დეკომპოზიცია
    "clouds", "temp", "humidity", "wind_speed",            # ამინდი
    "hour_sin", "hour_cos", "month_sin", "month_cos",      # დროის ციკლური ნიშნულები
    "solar_lag_1h", "solar_lag_24h", "cloud_lag_1h",       # ისტორიული ლაგები
    # ... სულ ~50 ნიშანი
]

# სამი კვანტილური მოდელი: q05 (ქვედა), q50 (ცენტრალური), q95 (ზედა)
QUANTILES = {"q05": 0.05, "q50": 0.5, "q95": 0.95}

models = {}
for name, q in QUANTILES.items():
    model = LGBMRegressor(
        objective        = "quantile",   # კვანტილური რეგრესია
        alpha            = q,            # სამიზნე კვანტილი
        n_estimators     = 2000,
        max_depth        = 6,
        num_leaves       = 31,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
    )
    # სწავლება: X — ნიშნები, y — რეალური გაზომილი სამიზნე (ზედამხედველობითი)
    model.fit(
        X_train, y_train,
        eval_set = [(X_calib, y_calib)],
        callbacks = [early_stopping(stopping_rounds=50)],  # overfitting-ის თავიდან ასაცილებლად
    )
    models[name] = model

# პროგნოზი და შეფასება ტესტის ამონაკრებზე
y_pred = models["q50"].predict(X_test).clip(0, 1)
mae  = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
```

---

*წყარო: `src/models/train_pv_real.py` (შემოკლებული — გამოტოვებულია მონაცემთა ჩატვირთვა,
conformal კალიბრაცია და შენახვა, რაც მოდელის სწავლების არსს არ ეხება).*
