# German Credit Dataset (UCI Statlog)

## 1. Dataset Overview

This directory contains the **Statlog (German Credit Data)** dataset, an established benchmark dataset for credit risk modeling and algorithmic fairness research.

* **Source**: [UCI Machine Learning Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata)
* **Donor**: Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg (1994)
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Number of Instances**: 1,000
* **Number of Features**: 20 (7 numerical, 13 categorical)
* **Target Variable**: `credit_risk` (Binary credit classification)

---

## 2. Target Variable Mapping

The dataset contains an original binary target classifying credit applicants:

| Original Raw Value (`credit_risk` in CSV) | Meaning | Model Internal Target (`y`) | Semantic Interpretation |
| :--- | :--- | :--- | :--- |
| `1` | Good Credit | `0` | Low credit risk / Approved |
| `2` | Bad Credit | `1` | High credit risk / Default |

**Note on Polarity**: The internal target is mapped such that `1` represents the positive/adverse outcome (bad credit / high risk), and output probabilities $P(y=1)$ represent the predicted probability of bad credit.

---

## 3. Feature Descriptions

The dataset consists of 20 input attributes:

### Numerical Features (7)

1. `duration_months`: Duration of credit in months (integer)
2. `credit_amount`: Credit amount in Deutsche Mark (DM) (integer)
3. `installment_rate`: Installment rate in percentage of disposable income (integer: 1–4)
4. `present_residence`: Present residence duration in years (integer: 1–4)
5. `age`: Age in years (integer)
6. `existing_credits`: Number of existing credits at this bank (integer: 1–4)
7. `num_dependents`: Number of people liable for maintenance (integer: 1–2)

### Categorical Features (13)

1. `status_checking_account`: Status of existing checking account
   * `A11`: $< 0$ DM
   * `A12`: $0 \le \dots < 200$ DM
   * `A13`: $\ge 200$ DM / salary assignments for at least 1 year
   * `A14`: no checking account
2. `credit_history`: Credit history / repayment track record
   * `A30`: no credits taken / all credits paid back duly
   * `A31`: all credits at this bank paid back duly
   * `A32`: existing credits paid back duly till now
   * `A33`: delay in paying off in the past
   * `A34`: critical account / other credits existing (not at this bank)
3. `purpose`: Credit purpose
   * `A40`: car (new), `A41`: car (used), `A42`: furniture/equipment, `A43`: radio/television, `A44`: domestic appliances, `A45`: repairs, `A46`: education, `A47`: vacation, `A48`: retraining, `A49`: business, `A410`: others
4. `savings_account`: Savings account / bonds balance
   * `A61`: $< 100$ DM, `A62`: $100 \le \dots < 500$ DM, `A63`: $500 \le \dots < 1000$ DM, `A64`: $\ge 1000$ DM, `A65`: unknown / no savings account
5. `present_employment`: Present employment duration
   * `A71`: unemployed, `A72`: $< 1$ year, `A73`: $1 \le \dots < 4$ years, `A74`: $4 \le \dots < 7$ years, `A75`: $\ge 7$ years
6. `personal_status_sex`: Personal status and sex (combined categorical attribute)
   * `A91`: male : divorced/separated
   * `A92`: female : divorced/separated/married
   * `A93`: male : single
   * `A94`: male : married/widowed
   * `A95`: female : single
7. `other_debtors`: Other debtors / guarantors
   * `A101`: none, `A102`: co-applicant, `A103`: guarantor
8. `property`: Property owned
   * `A121`: real estate, `A122`: building society savings / life insurance, `A123`: car or other, `A124`: unknown / no property
9. `other_installment_plans`: Other installment plans
   * `A141`: bank, `A142`: stores, `A143`: none
10. `housing`: Housing status
    * `A151`: rent, `A152`: own, `A153`: for free
11. `job`: Employment type / qualification level
    * `A171`: unemployed / unskilled non-resident, `A172`: unskilled resident, `A173`: skilled employee / official, `A174`: management / self-employed / highly qualified employee / officer
12. `telephone`: Telephone registration
    * `A191`: none, `A192`: yes, registered under customer name
13. `foreign_worker`: Foreign worker indicator
    * `A201`: yes, `A202`: no

---

## 4. Preprocessing Assumptions

1. **No Missing Values**: The original dataset has no missing values across all 1,000 instances.
2. **Sensitive Attribute Preservation**: The raw `personal_status_sex` attribute is preserved directly in feature matrices without modification. Fairness transformations and sensitive group analysis belong strictly to downstream fairness modules.
3. **Encoding**: Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore")` inside the model pipeline.
4. **Deterministic Partitioning**: Train/test splits use an 80/20 stratified split (`test_size=0.2`, `stratify=y`, `random_state=42`).
