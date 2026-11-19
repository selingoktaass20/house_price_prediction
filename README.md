# Istanbul House Price Prediction

An end-to-end machine learning project that predicts house prices in Istanbul using Turkish real estate sales data. The project covers data filtering, categorical encoding, skewness transformation, model training, and performance comparison.

## Dataset
- **Source:** Turkish House Sales Dataset (`processed_turkish_house_sales.csv`)
- **Scope:** Filtered specifically for **Istanbul** listings (1,099 records after cleaning).
- **Features:** Seller type (`satici_tip`), area (`Metrekare`), room count (`Oda_Sayisi`), district (`Ilce`), neighborhood (`Mahalle`), and target price (`fiyat`).

## Technologies & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-Learn, XGBoost
- **Environment:** Jupyter Notebook

## Methodology
1. **Data Filtering & Cleaning:**
   - Filtered the dataset for Istanbul properties and dropped non-informative columns (`il`, `Tarih`).
   - Removed rare room count categories (fewer than 5 listings) and grouped rare neighborhoods (fewer than 4 listings) under `"diger"`.
2. **Feature Engineering & Encoding:**
   - Binary encoded `satici_tip` (`Sahibinden`: 0, `Emlak-ofisinden`: 1).
   - Applied One-Hot Encoding (`pd.get_dummies`) to `Oda_Sayisi`, `Ilce`, and `Mahalle`.
   - Applied log transformation (`np.log1p`) to `Metrekare` and `fiyat` to reduce right skewness.
3. **Modeling & Evaluation:**
   - Split the data into 80% training and 20% testing sets (`random_state=42`).
   - Trained and compared `RandomForestRegressor` and `XGBRegressor` (`n_estimators=200`, `max_depth=8`) using MSE and R² metrics on log-transformed prices.

## Results
| Model | MSE (Log Scale) | R² Score |
| :--- | :---: | :---: |
| Random Forest Regressor | `0.2045` | `0.5238` |
| XGBoost Regressor | `0.2126` | `0.5051` |

*Random Forest Regressor achieved slightly better performance than XGBoost on this dataset.*

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/selingoktaass20/house_price_prediction.git
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn xgboost
   ```
3. Open and run `price_prediction.ipynb` in Jupyter Notebook or VS Code.
