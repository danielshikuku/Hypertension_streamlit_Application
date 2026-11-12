import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Hypertension Risk Analysis", layout="wide")

NUM_COLS = ["Age", "BMI", "Waist_Circumference_cm", "Sleep_Hours",
            "Creatinine_mg_dL", "Fasting_Glucose_mg_dL", "Cholesterol_mg_dL",
            "Systolic_BP_mmHg", "Diastolic_BP_mmHg"]
PREDICTOR_NUM = NUM_COLS[:-2]          
CAT_COLS = ["Gender", "Salt_Intake", "Physical_Activity", "Alcohol_Consumption",
            "Smoking_Status", "Stress_Level", "Family_History_HTN"]
MEDIAN_COLS = ["Sleep_Hours", "Waist_Circumference_cm", "Creatinine_mg_dL",
               "Cholesterol_mg_dL", "Fasting_Glucose_mg_dL"]


@st.cache_data
def load_raw():
    return pd.read_csv("hypertension_dataset_5000.csv")


@st.cache_data
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna("Unknown")
    for c in ["Salt_Intake", "Stress_Level"]:
        df[c] = df[c].fillna(df[c].mode()[0])
    for c in MEDIAN_COLS:
        df[c] = df[c].fillna(df[c].median())

    
    df["Age_band"] = pd.cut(df["Age"], [0, 30, 45, 60, 200], right=False,
                            labels=["Under 30", "30-44", "45-59", "60+"])
    df["BMI_Group"] = pd.cut(df["BMI"], [0, 18.5, 25, 30, 100], right=False,
                             labels=["Underweight", "Normal", "Overweight", "Obese"])
    df["HTN"] = (df["Hypertension"] == "Yes").astype(int)
    return df


raw = load_raw()
df = clean(raw)


st.sidebar.title("Hypertension Analysis")
page = st.sidebar.radio("Go to", [
    "Overview", "Data Cleaning", "Distributions",
    "Risk Factors", "Correlations", "Modelling", "Findings & Limitations", "Conclution & Next step"])
st.sidebar.caption("Dataset: 5,000 patients")

if page == "Overview":
    st.title("HYPERTENSION RISK ANALYSIS")
    st.header("Hypertension Risk Analytics and Decision-Support Application")
    st.subheader("Hypertension in the UK")
    st.write("Hypertension is a major public health concern because its development is influenced by a combination of demographic, behavioral and clinical factors. Awareness of the relationship between these factors and hypertension can be useful in identifying trends that warrant consideration in primary health care and in health education and prevention. This project consequently analyzes the available data to investigate disparities in hypertension among population groups and to investigate potential associations between lifestyle habits, body measurements, family history and clinical indicators. Rather than assuming that individual factors directly cause hypertension, the analysis seeks to establish what associations and patterns are evident in the data and where further investigation may be needed. ")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Hypertension prevalence", f"{df['HTN'].mean():.1%}")
    c4.metric("Mean age", f"{df['Age'].mean():.1f}")

    st.subheader("Preview")
    st.dataframe(df.head(20), width="stretch")

    with st.expander("Summary statistics"):
        st.dataframe(df.describe(include="all").T, width="stretch")

elif page == "Data Cleaning":
    st.title("Data Cleaning")

    st.subheader("1. Missing values before cleaning")
    miss = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "unique": raw.nunique(),
        "missing": raw.isnull().sum(),
        "missing_%": (raw.isnull().mean() * 100).round(1),
    })
    st.dataframe(miss, width="stretch")
    fig = px.bar(miss[miss.missing > 0].reset_index(), x="index", y="missing",
                 labels={"index": "Column"}, title="Columns with missing values")
    st.plotly_chart(fig, width="stretch")

    st.subheader("2. How each column was handled")
    st.table(pd.DataFrame({
        "Column": ["Alcohol_Consumption", "Salt_Intake, Stress_Level",
                   "Sleep, Waist, Creatinine, Cholesterol, Glucose"],
        "Method": ["Filled with 'Unknown'", "Filled with mode", "Filled with median"]}))

    st.subheader("3. Duplicates, outliers and invalid values")
    c1, c2 = st.columns(2)
    c1.metric("Duplicate rows", int(df.duplicated().sum()))
    c2.metric("Missing after cleaning", int(df.isnull().sum().sum()))

    out = {}
    for col in NUM_COLS:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        out[col] = int(((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum())
    st.write("**Outliers (1.5 × IQR rule)**")
    st.dataframe(pd.Series(out, name="outliers"), width="stretch")

    col = st.selectbox("Inspect a column with a boxplot", NUM_COLS)
    st.plotly_chart(px.box(df, x=col, title=f"Outlier detection: {col}"),
                    width="stretch")

    st.subheader("4. New features")
    st.write("`Age_band`, `BMI_Group` and `HTN` (0/1 target) were created.")
    st.dataframe(df[["Age", "Age_band", "BMI", "BMI_Group", "Hypertension", "HTN"]].head(),
                 width="stretch")

    st.download_button("Download cleaned CSV", df.to_csv(index=False),
                       "cleaned_hypertension_dataset_5000.csv", "text/csv")

elif page == "Distributions":
    st.title("Distributions")

    st.subheader("Numeric variables")
    col = st.selectbox("Variable", NUM_COLS)
    bins = st.slider("Bins", 10, 80, 30)
    color = st.checkbox("Split by hypertension status")
    fig = px.histogram(df, x=col, nbins=bins, marginal="box",
                       color="Hypertension" if color else None, barmode="overlay")
    st.plotly_chart(fig, width="stretch")
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean", f"{df[col].mean():.2f}")
    s2.metric("Median", f"{df[col].median():.2f}")
    s3.metric("Skewness", f"{df[col].skew():.2f}")

    st.divider()
    st.subheader("Categorical variables (counts, not risk)")
    cat = st.selectbox("Variable ", CAT_COLS + ["Hypertension"])
    counts = df[cat].value_counts().reset_index()
    counts.columns = [cat, "count"]
    st.plotly_chart(px.bar(counts, x=cat, y="count", text="count"),
                    width="stretch")
    st.info("A tall bar only means many patients are in that group. It does not "
            "mean that group has a higher risk.")

elif page == "Risk Factors":
    st.title("Which factors go with hypertension?")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Categorical rates", "Numeric comparison", "Age × BMI", "Statistical tests"])

    with tab1:
        factor = st.selectbox("Factor", CAT_COLS + ["Age_band", "BMI_Group"])
        rate = (df.groupby(factor, observed=True)["HTN"].mean()
                  .mul(100).round(1).sort_values().reset_index())
        rate.columns = [factor, "Hypertension rate (%)"]
        fig = px.bar(rate, y=factor, x="Hypertension rate (%)", orientation="h",
                     text="Hypertension rate (%)")
        fig.add_vline(x=df["HTN"].mean()*100, line_dash="dash",
                      annotation_text="Overall rate")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        num = st.selectbox("Variable", PREDICTOR_NUM)
        st.plotly_chart(px.box(df, x="Hypertension", y=num, color="Hypertension"),
                        width="stretch")

    with tab3:
        pivot = (df.pivot_table(index="BMI_Group", columns="Age_band",
                                values="HTN", aggfunc="mean", observed=True) * 100)
        st.plotly_chart(px.imshow(pivot.round(0), text_auto=True, aspect="auto",
                                  color_continuous_scale="Reds",
                                  title="Hypertension rate (%) by BMI group and age band"),
                        width="stretch")
        st.caption("Risk compounds: higher BMI means higher risk at every age, and "
                   "higher age means higher risk at every BMI.")

    with tab4:
        st.write("**Chi-square tests (categorical vs hypertension)**")
        rows = []
        for c in CAT_COLS + ["Age_band", "BMI_Group"]:
            ct = pd.crosstab(df[c], df["Hypertension"])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
            rows.append({"Factor": c, "chi2": round(chi2, 1), "p-value": f"{p:.3g}",
                         "Cramér's V": round(v, 2)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.write("**Welch t-tests (numeric vs hypertension)**")
        rows = []
        for c in PREDICTOR_NUM:
            a, b = df.loc[df.HTN == 1, c], df.loc[df.HTN == 0, c]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)  # Cohen's d
            rows.append({"Variable": c, "Mean (Yes)": round(a.mean(), 1),
                         "Mean (No)": round(b.mean(), 1), "p-value": f"{p:.3g}",
                         "Cohen's d": round(d, 2)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.warning("Significance and effect size are different questions, and "
                   "association is not causation. These tests look at one factor at "
                   "a time, so age, BMI and other factors overlap.")

elif page == "Correlations":
    st.title("Correlation Heatmap")
    cols = st.multiselect("Variables", NUM_COLS, default=NUM_COLS)
    corr = df[cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                              color_continuous_scale="RdBu_r", aspect="auto"),
                    width="stretch")
    st.write("Age is the strongest correlate of systolic BP (about 0.54). Waist, creatinine, glucose, and BMI are around 0.3 to 0.4. Sleep is mildly negative.")

elif page == "Modelling":
    st.title("Modelling")

    X = pd.get_dummies(df[PREDICTOR_NUM + CAT_COLS], drop_first=True)

    @st.cache_resource
    def train_models():
        # A. Regression: systolic BP
        Xtr, Xte, ytr, yte = train_test_split(X, df["Systolic_BP_mmHg"],
                                              test_size=0.2, random_state=42)
        lin = LinearRegression().fit(Xtr, ytr)
        reg = {"r2": r2_score(yte, lin.predict(Xte)),
               "mae": mean_absolute_error(yte, lin.predict(Xte)),
               "coef": pd.Series(lin.coef_, X.columns).sort_values()}
        # B. Classification: hypertension
        Xtr, Xte, ytr, yte = train_test_split(X, df["HTN"], test_size=0.2,
                                              stratify=df["HTN"], random_state=42)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        fpr, tpr, _ = roc_curve(yte, proba)
        return reg, clf, {"auc": roc_auc_score(yte, proba),
                          "report": classification_report(yte, pred, output_dict=True),
                          "cm": confusion_matrix(yte, pred), "fpr": fpr, "tpr": tpr}

    reg, clf, res = train_models()
    t1, t2, t3 = st.tabs(["Predict systolic BP", "Predict hypertension", "Try a patient"])

    with t1:
        c1, c2 = st.columns(2)
        c1.metric("R²", f"{reg['r2']:.2f}")
        c2.metric("MAE (mmHg)", f"{reg['mae']:.1f}")
        st.plotly_chart(px.bar(reg["coef"], orientation="h", height=600,
                               labels={"value": "Coefficient", "index": ""}),
                        width="stretch")
        st.write("Findings: Systolic blood pressure is the primary measure used to define hypertension, thus predicting it reflects the closeness of lifestyle and clinical factors to the risk of hypertension. The regression explained approximately 58% of the variance in systolic BP and had an average prediction error of approximately 8 mmHg (MAE = 8.3). This is consistent with the findings of the Risk Factors study which showed that higher age, BMI, waist circumference and glucose were associated with higher rates of blood pressure and hypertension.")

    with t2:
        rep = res["report"]
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{res['auc']:.2f}")
        c2.metric("Recall (hypertensive)", f"{rep['1']['recall']:.0%}")
        c3.metric("Precision (hypertensive)", f"{rep['1']['precision']:.0%}")
        left, right = st.columns(2)
        cm = pd.DataFrame(res["cm"], index=["Actual No", "Actual Yes"],
                          columns=["Pred No", "Pred Yes"])
        left.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                    title="Confusion matrix"), width="stretch")
        roc = px.line(x=res["fpr"], y=res["tpr"], title="ROC curve",
                      labels={"x": "False positive rate", "y": "True positive rate"})
        roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line_dash="dash")
        right.plotly_chart(roc, width="stretch")
        st.caption("During my modelling, I spotted hypertension. I classified 90% of the 1,000 test patients correctly, and I found 91% of the people who truly have hypertension. When I got that someone had hypertension, it was right 82% of the time, so about 1 in 5 of my 'Yes' calls was a false alarm. The AUC of 0.973 means that if I compared a patient with hypertension to one without, the model gives me the hypertensive patient the higher risk score about 97% of the time.")

    with t3:
        st.write("Enter patient details (blood pressure is **not** used by this model).")
        a, b, c = st.columns(3)
        inp = {
            "Age": a.slider("Age", 18, 90, 45),
            "BMI": a.slider("BMI", 16.0, 44.0, 27.0),
            "Waist_Circumference_cm": a.slider("Waist (cm)", 54, 127, 89),
            "Sleep_Hours": b.slider("Sleep hours", 3.5, 10.5, 7.0),
            "Creatinine_mg_dL": b.slider("Creatinine", 0.5, 1.7, 0.94),
            "Fasting_Glucose_mg_dL": b.slider("Glucose", 60, 175, 93),
            "Cholesterol_mg_dL": c.slider("Cholesterol", 105, 290, 187),
        }
        for k in CAT_COLS:
            inp[k] = c.selectbox(k, sorted(df[k].unique()))
        row = pd.get_dummies(pd.DataFrame([inp]))
        row = row.reindex(columns=X.columns, fill_value=0)
        prob = clf.predict_proba(row)[0, 1]
        st.metric("Predicted probability of hypertension", f"{prob:.0%}")
        st.progress(float(prob))
        st.caption("This is for clinical advice and should not be implemented without medical review as some adjustments are needed in this report  and next steps include confirming data and definitions, re-running analytical tasks, testing the application and having it clinically reviewed.")

elif page == "Findings & Limitations":
    st.title("Findings")
    st.markdown("""
1. **Age is the strongest factor.** Age was the most influential variable. Hypertension prevalence increases with age as follows: 14% (less than 30), 22% (30-44), 38% (45-59), 58% (more than 60). Patients aged more than 60 years are four times more likely to be hypertensive than those who are less than 30 years old. The average age of hypertensive patients is 57 years compared to 44 years for other patients. Age was the largest predictor in both statistical tests and models.
2. **Body mass index.** Body Mass Index makes a large difference. Obese patients (48%) are twice as likely to be hypertensive as normal-weight patients (21%). Moreover, waist circumference in hypertensive patients is also larger (94 cm compared to 87 cm in the others) and their body mass index is higher (29.2 versus 27.2).
3. **Salt and physical activity** Salt and physical activity make a large difference. Very high salt consumption causes 54% of hypertension cases compared to 18% for low salt consumption. Also, 49% of hypertensive patients are physically inactive compared to 19% of active patients.
4. **Risk compounds.** This is clearly shown in the table of age and BMI. Only 8% of people under 30 who are of normal weight have hypertension but 72% of people 60+ who are obese do. The risk is higher at each age with a higher BMI and the risk is higher at each BMI with a higher age.
5. **Men (43%) and those with a family history (44%)** have higher rates than women (27%) and those without a family history (30%).
6. **Weaker factors:** heavy drinking, very high stress and smoking history. Former smokers are older on average, so age probably explains their higher rate (confounding).

**Model:** a logistic regression without blood pressure identified about 79% of hypertensive patients (AUC ≈ 0.85), with age as the strongest predictor.
""")
    st.title("How these findings can help a clinical officer")
    st.markdown("""
1. This finding can help clinical officers in target screening. Age is the strongest factor, with prevalence of 14% under 30 and 58% over 60. This supports the argument for focusing blood pressure checks on older adults, and the age-by-BMI pattern (8% in young, normal-weight patients compared with 72% in obese patients over 60) shows where screening is most likely to identify undiagnosed cases.
2. This can help clinical officers in target screening.
3. This can help clinical officers in the identification of modifiable risks. Excess weight, high salt intake and physical inactivity are all modifiable risk factors, associated with much higher rates. This provides clinicians with real lifestyle targets to counsel on weight control, salt reduction and more activity. These are the areas where primary care and health education can help.
4. This can help clinical officers to identify the cumulative risk. Risk increases when age and BMI come together, so patients with multiple risk factors require more attention than any single factor alone would suggest. This enables clinicians to prioritize follow-up rather than dealing with each factor in isolation.
5. This can help clinical officer in non-modifiable factors for early detection. Male sex and family history are not modifiable but define individuals who might benefit from earlier or more frequent scree
6. This can help clinical offer not to over-interpret. Heavy drinking, high stress and former smoking had higher rates but small effects, the former-smoker result being possibly explained by age. This is a reminder for clinicians not to put too much weight on weak or confounded factors.
""")
    st.header("Limitations")
    st.markdown("""
    The raw dataset had missing values in eight columns, which I filled in during cleaning and this matters because median and mode filling keeps my sample size but slightly shrinks the natural variation and weakens real relationships for those variables. Alcohol has the biggest problem, with more than a quarter of its values unknown
- **Imputation:** about 27% of alcohol data was missing and coded as "Unknown". Other columns (1–4% missing) were filled with the median or mode. This shrinks variation and creates artificial peaks at the median value.
- **Unknown alcohol group** is not a true drinking category, so don't read its low hypertension rate as "healthier".
- **Association is not causation**, and the tests look at one factor at a time.
""")  
elif page == "Conclution & Next step":
    st.title("Conclusion")
    st.write("Higher hypertension prevalence appeared among older participants, males, sedentary participants, those reporting high salt intake or very high stress, obese participants and those with a family history. The hypertension group also had higher fasting glucose and cholesterol averages, but there was minimal difference in creatinine. These findings could be used to guide population-level prevention and health education, but not for diagnosis or causal inferences. Preprocessing retained records while making missingness visible; engineered features supported meaningful comparisons. The project contributes an exploration and communication design for clinical staff. Limitations are lack of alcohol data, questionable waist measurements, possible confounding, and uncertain representativeness and unresolved prevalence discrepancy.")
    st.header("Next step")
    st.write("Next steps include confirming data and definitions, re-running analytical tasks, testing the application, and having it clinically reviewed.")