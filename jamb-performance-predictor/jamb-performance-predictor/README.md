# JAMB Performance Predictor

A web app that estimates whether a Nigerian secondary-school student is on track to score **200 or above in JAMB**, and shows which factors are helping or holding them back.

Built from my B.Sc. thesis, *Predicting Student's Academic Performance Using Machine Learning Models* (Caleb University, Imota, 2025).

**Live app:** _add your Streamlit link here after deploying_

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment the next line -->
<!-- ![App screenshot](docs/screenshot.png) -->

## What it does

- **Predict:** enter study habits, school details and home background; get a pass probability, a plain-language verdict, and a chart of the factors pushing the result up or down compared with an average student.
- **Model comparison:** Logistic Regression, K-Nearest Neighbors, Random Forest and Support Vector Machine compared with RMSE, MAE and R².
- **About:** problem, data, method, limitations and future work.

## Thesis results

Test set, 80/20 split, `random_state=42` (Table 4.1):

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| **Logistic Regression** | **39.75** | **31.89** | **0.3452** |
| K-Nearest Neighbors | 44.09 | 35.65 | 0.1944 |
| Random Forest | 40.63 | 32.97 | 0.3161 |
| Support Vector Machine | 47.70 | 37.12 | 0.0571 |

Random Forest fits the training data closely (R² 0.90) but drops to 0.32 on unseen data, so Logistic Regression was chosen for the app.

## Run it locally

```bash
git clone <your-repo-url>
cd jamb-performance-predictor
pip install -r requirements.txt
```

1. Download `jamb_exam_results.csv` from [Kaggle](https://www.kaggle.com/datasets/idowuadamo/students-performance-in-2024-jamb) into `dataset/`.
2. Train and save the model: `python train.py`
3. Launch the app: `streamlit run app.py`

Other scripts:

| Command | What it does |
| --- | --- |
| `python compare.py` | Re-runs the four-model comparison and saves `results/model_comparison.csv` |
| `python compare_chart.py` | Draws the comparison chart (thesis Figure 4.2) |

## Deploy for free (Streamlit Community Cloud)

1. Run `python train.py` locally, then commit the generated `logreg_model.pkl`, `scaler.pkl`, `feature_columns.pkl` and `model_meta.json`.
2. Push the repository to GitHub (public).
3. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
4. Choose **Create app**, select this repository, branch `main`, main file `app.py`, then **Deploy**.
5. Paste the live URL at the top of this README.

Free apps go to sleep after a period of inactivity and wake with one click.

## Project structure

```
app.py              Streamlit web app
train.py            Trains and saves the pass/fail model
compare.py          Four-model comparison (thesis Table 4.1)
compare_chart.py    Comparison chart (thesis Figure 4.2)
plots.py            Shared matplotlib figures
project_info.py     Thesis results, labels and sample profiles
dataset/            Place the Kaggle CSV here (git-ignored)
results/            Generated comparison table and chart
.streamlit/         Theme configuration
```

## Method in brief

1. Fill missing values (mean for numeric, mode for categorical).
2. One-hot encode categorical features; standardise numeric features.
3. Define pass as a JAMB score of 200 or above.
4. Split 80/20, train Logistic Regression, evaluate on the held-out 20%.

## Limitations

- One exam board and one year of data, so results may not transfer to other populations.
- The model finds statistical patterns, not causes.
- Deep learning was out of scope.
- Future work: hybrid models, larger and more varied data, and explainable AI (for example SHAP) inside the app.

## Author

**Olesin Ibukunoluwa Moyosore** · B.Sc. Computer Science, Caleb University · [GitHub](https://github.com/Moyohor)
