"""JAMB Performance Predictor: Streamlit app for the thesis
"Predicting Student's Academic Performance Using Machine Learning Models".

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from plots import comparison_figure, drivers_figure
from project_info import (AT_RISK_SAMPLE, AUTHOR, DATASET_URL, FEATURES, GITHUB_PROFILE,
                          GROUPS, INSTITUTION, PASS_MARK, STRONG_SAMPLE, SUPERVISOR,
                          THESIS_METRICS, TITLE, YEAR)
from train import ARTIFACTS, DATA_PATH

ROOT = Path(__file__).parent

st.set_page_config(page_title="JAMB Performance Predictor", page_icon="🎓", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif;
}
h1, h2, h3, h4, .hero h1, .result-prob, .verdict {
  font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif !important;
  letter-spacing: -0.01em;
  color: #13241F;
}
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

.hero { padding: 0.6rem 0 1.4rem 0; max-width: 720px; }
.hero h1 { font-size: 2.6rem; line-height: 1.08; margin: 0 0 0.6rem 0; font-weight: 700; }
.hero p { font-size: 1.1rem; line-height: 1.55; color: #3C5049; margin: 0; }

.result {
  background: #FFFFFF; border: 1px solid #D5DEDA; border-left: 8px solid var(--tone);
  border-radius: 6px; padding: 1.4rem 1.6rem 1.5rem 1.6rem; color: #13241F;
}
.result-label { font-size: 0.98rem; color: #3C5049; }
.result-prob { font-size: 4.2rem; font-weight: 700; line-height: 1.05; color: var(--tone) !important; margin: 0.15rem 0 0.7rem 0; }
.meter { position: relative; height: 10px; background: #E6EDEA; border-radius: 5px; margin-bottom: 1rem; }
.meter span { position: absolute; left: 0; top: 0; bottom: 0; background: var(--tone); border-radius: 5px; transition: width .35s ease; }
.meter i { position: absolute; top: -5px; bottom: -5px; width: 2px; background: #13241F; opacity: .55; }
.verdict { font-size: 1.45rem; font-weight: 700; color: var(--tone) !important; margin-bottom: 0.25rem; }
.result p { margin: 0; color: #3C5049; line-height: 1.5; }

.note { font-size: 0.88rem; color: #5E716B; line-height: 1.5; }
.pill-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .8rem 0; }
.pill { background: #E6EDEA; color: #13241F; border-radius: 999px; padding: .15rem .7rem; font-size: .88rem; }
.pill.good { background: #D7EBE2; color: #07513B; }
.pill.bad { background: #F6DAD7; color: #8A1C15; }

[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D5DEDA; border-radius: 6px; padding: .8rem 1rem; }
button[data-baseweb="tab"] { font-size: 1.02rem; }
button:focus-visible { outline: 3px solid #0B6B4F !important; outline-offset: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

GREEN, AMBER, RED = "#0B6B4F", "#B7791F", "#B3261E"


# --------------------------------------------------------------------------- model loading
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved model. If it is missing but the dataset is present, train it first."""
    if not all((ROOT / f).exists() for f in ARTIFACTS):
        if not (ROOT / DATA_PATH).exists():
            return None
        from train import train
        train(ROOT / DATA_PATH, ROOT)
    return {
        "model": joblib.load(ROOT / "logreg_model.pkl"),
        "scaler": joblib.load(ROOT / "scaler.pkl"),
        "columns": joblib.load(ROOT / "feature_columns.pkl"),
        "meta": json.loads((ROOT / "model_meta.json").read_text()),
    }


with st.spinner("Preparing the model..."):
    art = load_artifacts()

st.markdown(
    """
<div class="hero">
  <h1>JAMB performance predictor</h1>
  <p>Describe a student's study habits, school and home background. The model estimates
  whether they are on track to score 200 or above in JAMB, and shows which factors
  are helping or holding them back.</p>
</div>
""",
    unsafe_allow_html=True,
)

if art is None:
    st.error("The trained model is not in this project yet.")
    st.markdown(
        f"""
1. Download `jamb_exam_results.csv` from [Kaggle]({DATASET_URL}) and save it as `{DATA_PATH}`.
2. Run `python train.py` to create the model files.
3. Run `streamlit run app.py` again (or commit the generated `.pkl` and `model_meta.json` files before deploying).
"""
    )
    st.stop()

meta = art["meta"]
NUM = meta["numeric"]
CAT = meta["categorical"]
INPUT_GROUPS = {g: [f for f in fs if f in NUM or f in CAT] for g, fs in GROUPS.items()}
INPUT_FEATURES = [f for fs in INPUT_GROUPS.values() for f in fs]


def default_for(feature):
    if feature in NUM:
        spec = NUM[feature]
        return int(round(spec["median"])) if spec["is_int"] else float(spec["median"])
    return CAT[feature]["mode"]


def coerce(feature, value):
    """Make a sample value valid for the widget (clip numbers, fall back for unseen categories)."""
    if feature in NUM:
        spec = NUM[feature]
        v = min(max(float(value), spec["min"]), spec["max"])
        return int(round(v)) if spec["is_int"] else v
    return value if value in CAT[feature]["options"] else CAT[feature]["mode"]


def apply_profile(profile):
    for f in INPUT_FEATURES:
        st.session_state[f"in_{f}"] = coerce(f, profile.get(f, default_for(f)))


for f in INPUT_FEATURES:
    st.session_state.setdefault(f"in_{f}", default_for(f))


def render_input(feature):
    label, help_text = FEATURES[feature]
    key = f"in_{feature}"
    if feature in NUM:
        spec = NUM[feature]
        if spec["is_int"]:
            st.slider(label, int(spec["min"]), int(spec["max"]), key=key, help=help_text)
        else:
            st.number_input(label, float(spec["min"]), float(spec["max"]), step=0.1, key=key, help=help_text)
    else:
        st.selectbox(label, CAT[feature]["options"], key=key, help=help_text)


def predict(values):
    row = pd.DataFrame([values])
    encoded = pd.get_dummies(row).reindex(columns=art["columns"], fill_value=0).astype(float)
    scaled = art["scaler"].transform(encoded)
    prob = float(art["model"].predict_proba(scaled)[0, 1])
    contrib = pd.Series(art["model"].coef_[0] * scaled[0], index=art["columns"])
    return prob, contrib


def factor_of(column):
    for c in CAT:
        if column.startswith(c + "_"):
            return c
    return column


def verdict_for(p):
    if p >= 0.65:
        return "Likely to reach 200", GREEN
    if p >= 0.35:
        return "Borderline", AMBER
    return "Likely to fall below 200", RED


tab_predict, tab_models, tab_about = st.tabs(["Predict", "Model comparison", "About the project"])

# --------------------------------------------------------------------------- predict
with tab_predict:
    b1, b2, b3, _ = st.columns([1.2, 1.2, 1, 3])
    b1.button("Load strong sample", on_click=apply_profile, args=(STRONG_SAMPLE,), use_container_width=True)
    b2.button("Load at-risk sample", on_click=apply_profile, args=(AT_RISK_SAMPLE,), use_container_width=True)
    b3.button("Reset", on_click=apply_profile, args=({},), use_container_width=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        for group, feats in INPUT_GROUPS.items():
            if not feats:
                continue
            with st.container(border=True):
                st.markdown(f"#### {group}")
                for feat in feats:
                    render_input(feat)

    values = {f: st.session_state[f"in_{f}"] for f in INPUT_FEATURES}
    prob, contrib = predict(values)
    label, tone = verdict_for(prob)

    with right:
        st.markdown(
            f"""
<div class="result" style="--tone:{tone}">
  <div class="result-label">Chance of scoring {PASS_MARK} or above</div>
  <div class="result-prob">{prob:.0%}</div>
  <div class="meter"><span style="width:{prob * 100:.0f}%"></span><i style="left:50%"></i></div>
  <div class="verdict">{label}</div>
  <p>The line on the bar marks 50%, the point where the model switches from predicting a fail to a pass.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        impact = contrib.groupby(factor_of).sum()
        impact = impact.reindex(impact.abs().sort_values(ascending=False).index)
        top = impact.head(6)
        chart_df = pd.DataFrame({"Factor": [FEATURES.get(i, (i,))[0] for i in top.index], "Impact": top.values})

        st.markdown("#### What is driving this result")
        st.caption("Each bar compares this student with an average student in the dataset.")
        st.pyplot(drivers_figure(chart_df), use_container_width=True)

        helping = [FEATURES[i][0] for i in impact.index if impact[i] > 0.05][:3]
        holding = [FEATURES[i][0] for i in impact.index[::-1] if impact[i] < -0.05][:3]
        pills = "".join(f'<span class="pill good">{x}</span>' for x in helping)
        pills += "".join(f'<span class="pill bad">{x}</span>' for x in holding)
        if pills:
            st.markdown(
                f'<div class="pill-row">{pills}</div>'
                '<div class="note">Green: helping the student. Red: holding the student back.</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="note">This is an educational demonstration trained on one public dataset. It shows '
        "statistical patterns, not cause and effect, and it is not an official JAMB tool.</p>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- model comparison
with tab_models:
    st.markdown("### Which model predicts best?")
    st.write(
        "Four supervised models were trained on the same 80% of the data and scored on the remaining 20%. "
        "Logistic regression generalised best, so it powers this app."
    )
    thesis_df = pd.DataFrame(THESIS_METRICS)
    st.dataframe(
        thesis_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "RMSE": st.column_config.NumberColumn("RMSE (lower is better)", format="%.2f"),
            "MAE": st.column_config.NumberColumn("MAE (lower is better)", format="%.2f"),
            "R2": st.column_config.NumberColumn("R² (higher is better)", format="%.4f"),
        },
    )
    st.pyplot(comparison_figure(thesis_df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Logistic regression** has the highest test R² (0.3452) and the lowest RMSE and MAE, "
            "so it explains about 34.5% of the variation in performance and generalises best.\n\n"
            "**Support vector machine** scores lowest on every metric, which points to under-fitting."
        )
    with c2:
        st.markdown(
            "**Random forest** fits the training set very closely (R² 0.9010) but drops to 0.3161 on "
            "unseen data, a sign of over-fitting.\n\n"
            "**K-nearest neighbors** also declines sharply on the test set (R² 0.1944)."
        )
    st.caption(
        "Results from Table 4.1 of the thesis, produced by compare.py on the raw JAMB score. "
        "Run `python compare.py` to reproduce them."
    )

    st.markdown("### The model running in this app")
    m = meta["metrics"]
    st.write(
        f"The app classifies students as pass or fail at a score of {meta['pass_mark']}. "
        f"Trained on {meta['train_rows']:,} students and tested on {meta['test_rows']:,} unseen students."
    )
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
    cols[1].metric("Precision", f"{m['precision']:.1%}")
    cols[2].metric("Recall", f"{m['recall']:.1%}")
    cols[3].metric("F1 score", f"{m['f1']:.3f}")
    cols[4].metric("ROC AUC", f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a")
    cm = pd.DataFrame(
        m["confusion_matrix"],
        index=["Actually failed", "Actually passed"],
        columns=["Predicted fail", "Predicted pass"],
    )
    st.dataframe(cm, use_container_width=True)
    st.caption(f"Model trained {meta['trained_at']}. {meta['pass_rate']:.0%} of students in the dataset scored {meta['pass_mark']} or above.")

# --------------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"### {TITLE}")
    st.write(
        f"Final-year project by **{AUTHOR}**, supervised by {SUPERVISOR}. {INSTITUTION}, submitted in {YEAR} "
        "for the B.Sc. degree in Computer Science."
    )
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("#### The problem")
        st.write(
            "Many students leave higher education early, and warning signs often go unnoticed. Predictive "
            "models can flag students who may need support before results are in, so schools can step in "
            "with tutoring or mentoring."
        )
        st.markdown("#### What the project does")
        st.markdown(
            "- Reviews existing literature on machine learning for student performance\n"
            "- Prepares the 2024 JAMB student dataset from Kaggle\n"
            "- Trains and compares logistic regression, k-nearest neighbors, random forest and support vector machine\n"
            "- Evaluates them with RMSE, MAE and R²\n"
            "- Identifies which factors contribute most to performance\n"
            "- Delivers the best model as this web app"
        )
    with a2:
        st.markdown("#### Data and method")
        st.markdown(
            f"- [JAMB 2024 student performance dataset]({DATASET_URL}): about 5,000 students, 17 columns\n"
            f"- Missing values filled with the mean (numeric) or mode (categorical)\n"
            "- Categories one-hot encoded and numeric features standardised\n"
            f"- Pass defined as a JAMB score of {PASS_MARK} or above\n"
            "- 80/20 train and test split"
        )
        st.markdown("#### Tools")
        st.write("Python, scikit-learn, pandas, matplotlib, joblib, Streamlit and PyCharm.")
        st.markdown("#### Limitations and next steps")
        st.markdown(
            "- One exam board and one year of data, so results may not carry over to other populations\n"
            "- Deep learning models were out of scope\n"
            "- Future work: hybrid models, larger and more varied datasets, explainable AI in the app"
        )
    st.link_button("More projects on GitHub", GITHUB_PROFILE)
