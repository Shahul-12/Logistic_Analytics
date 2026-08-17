# Logistic_Analytics Understanding Delays, Cost & Customer Satisfaction

An end-to-end data analytics project analyzing 25,000 delivery records to identify the 
true drivers of delivery delays and failures, model delivery cost, and build a predictive 
risk-flagging system — with a strong emphasis on statistical rigor over surface-level 
observations.

## Project Overview

E-commerce logistics companies lose money and customer trust through delayed and failed 
deliveries. This project analyzes operational data to answer three questions:

1. **What actually causes delivery delays and failures** — partner performance, weather, 
   region, or something else?
2. **What determines delivery cost**, and is pricing consistent?
3. **Can at-risk deliveries be predicted in advance**, before they happen?

##  Key Findings

| Finding | Detail |
|---|---|
| **Weather dominates delay/failure risk** | Stormy conditions show a 32% delay rate vs. 14% in clear weather — more than double |
| **Partner performance is a myth in this data** | All 9 delivery partners cluster within a 3-point delay-rate band, despite appearing "statistically significant" |
| **Cost is almost perfectly formulaic** | cost ≈ ₹39 base + ₹5/km + ₹3/kg (R² = 0.991) |
| **Ratings directly track outcomes** | Delivered orders average 4-5★, delayed 2-3★, failed 1-2★ — minimal overlap |
| **A weather-based model can flag ~1 in 3 at-risk deliveries** in advance | Achieved by correcting a class imbalance problem in the classifier |

##  Methodology

### 1. Data Cleaning
- Identified and fixed a datetime-encoding bug where numeric hour values had been 
  mis-parsed as nanosecond-precision timestamps
- Uncovered a mislabeled categorical field: delayed (yes/no) was masking two distinct 
  outcomes — "delayed" and "failed" — validated via crosstab inspection

### 2. Exploratory Data Analysis
- Used pd.crosstab(normalize='index') to compare outcome rates fairly across groups 
  of different sizes
- Applied **chi-square hypothesis testing** to validate whether observed differences 
  (weather, partner) were statistically real or sampling noise
- Explicitly distinguished **statistical significance from practical significance** — 
  partner showed a technically significant p-value (0.044) but a negligible effect size, 
  while weather showed both significance and a large real-world effect

### 3. Regression Modeling
- Built a LinearRegression model predicting delivery_cost from distance and package 
  weight
- Achieved **R² = 0.991**, MAE = ₹37.14
- Recovered an interpretable pricing formula directly from model coefficients

### 4. Classification Modeling
- Built a DecisionTreeClassifier to predict delivery outcome (delivered/delayed/failed) 
  from weather, distance, weight, region, and vehicle type
- **Diagnosed a class imbalance problem**: baseline model reached 73% accuracy but only 
  0–4% recall on minority classes (delayed, failed) — it was simply predicting the 
  majority class
- Applied class_weight='balanced' to correct this, improving delayed/failed recall to 
  ~33% each, at the cost of overall accuracy (73% → 54%) — a deliberate trade-off in 
  favor of real-world usefulness
- Validated with confusion matrix and feature importance analysis, which independently 
  confirmed the EDA findings (weather + distance dominate; region/vehicle type are 
  near-irrelevant)

##  Visualizations

- Delivery outcome % by weather condition (stacked bar)
- Overall outcome distribution (count plot)
- Distance vs. delivery cost, colored by outcome (scatter)
- Delivery rating distribution by outcome (boxplot)
- Delay/failure rate by delivery partner (bar)
- Confusion matrix and feature importance (classification model)

##  Business Recommendations

1. **Build a weather-triggered risk alert system** for operations teams to proactively 
   reroute or notify customers ahead of storms/rain
2. **Stop scoring delivery partners on delay rate** — the data shows this isn't a 
   meaningful differentiator and may unfairly penalize partners for external conditions
3. **Use predicted delivery outcome as a leading indicator for customer satisfaction**, 
   enabling proactive service outreach on high-risk orders
4. **Audit pricing outliers** against the recovered cost formula to catch pricing errors 
   or standardize surcharges
5. **Prioritize weather-resilience investment** (backup vehicles, protective packaging) 
   over regional/vehicle-based interventions, which show minimal impact

##  Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** SciPy (chi-square testing)
- **Machine Learning:** Scikit-learn (Linear Regression, Decision Tree Classifier)
