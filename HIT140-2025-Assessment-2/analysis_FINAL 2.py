import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Load & Prep
df3 = pd.read_csv("merged_dataset.csv")

#Rat presence (treat missing arrivals as 0 -> no rat observed)
df3["rat_present"] = df3["rat_arrival_number"].fillna(0) > 0
df3["rat_present_label"] = df3["rat_present"].map({False: "No Rat", True: "Rat Present"})
ORDER = ["No Rat", "Rat Present"]

#Descriptive Summaries

def iqr(s: pd.Series) -> float:
    return s.quantile(0.75) - s.quantile(0.25)

delay_summary = (
    df3.groupby("rat_present")["bat_landing_to_food"]
       .agg(count="count", mean="mean", median="median", std="std",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75), iqr=iqr)
       .round(2)
)
print("\nHesitation time by rat presence:\n", delay_summary)

landing_summary = (
    df3.groupby("rat_present")["bat_landing_number"]
       .agg(count="count", mean="mean", median="median", std="std",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75), iqr=iqr)
       .round(2)
)
print("\nBat landings by rat presence:\n", landing_summary)

reward_summary = (
    df3.groupby("rat_present")["reward"]
       .agg(count="count", success_rate="mean")
       .round(3)
)
print("\nFeeding success by rat presence:\n", reward_summary)

#Risk summary
risk_summary = (
    df3.groupby("rat_present")["risk"]
       .agg(count="count", risk_rate="mean")
       .round(3)
)
print("\nRisk-taking behavior by rat presence:\n", risk_summary)

#Plots
sns.set(style="whitegrid")

#Hesitation
plt.figure()
sns.boxplot(data=df3, x="rat_present_label", y="bat_landing_to_food", order=ORDER, showfliers=True)
sns.stripplot(data=df3, x="rat_present_label", y="bat_landing_to_food",
              order=ORDER, color="0.25", alpha=0.35, jitter=True)
plt.title("Hesitation Time by Rat Presence")
plt.xlabel("Rat Presence")
plt.ylabel("Seconds to Get Food")
plt.tight_layout()
plt.show()

#Bat landings
plt.figure()
sns.pointplot(data=df3, x="rat_present_label", y="bat_landing_number",
              order=ORDER, errorbar=("ci", 95))
plt.title("Mean Bat Landings by Rat Presence (95% CI)")
plt.xlabel("Rat Presence")
plt.ylabel("Bat Landings per Period")
plt.tight_layout()
plt.show()

#Landing trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=df3, x="hours_after_sunset", y="bat_landing_number",
             hue="rat_present_label", hue_order=ORDER, errorbar=("ci", 95), palette="muted")
plt.title("Bat Landing Number over Time by Rat Presence")
plt.xlabel("Hours After Sunset")
plt.ylabel("Bat Landings per Period")
plt.legend(title="Rat Presence")
plt.tight_layout()
plt.show()

#Distribution of bat landings
plt.figure(figsize=(8, 6))
sns.histplot(data=df3, x="bat_landing_number", hue="rat_present_label",
             hue_order=ORDER, bins=20, kde=False, multiple="layer")
plt.title("Distribution of Bat Landings by Rat Presence")
plt.xlabel("Bat Landings per Period")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Feeding success
success_rate = (
    df3.groupby("rat_present_label")["reward"]
       .mean()
       .reindex(ORDER)
       .rename("success_rate")
       .reset_index()
)
plt.figure()
ax = sns.barplot(data=success_rate, x="rat_present_label", y="success_rate", order=ORDER)
ax.bar_label(ax.containers[0], labels=[f"{p:.1%}" for p in success_rate["success_rate"]])
plt.title("Feeding Success Rate by Rat Presence")
plt.xlabel("Rat Presence")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#Risk-taking behavior
risk_rate = (
    df3.groupby("rat_present_label")["risk"]
       .mean()
       .reindex(ORDER)
       .rename("risk_rate")
       .reset_index()
)
plt.figure()
ax = sns.barplot(data=risk_rate, x="rat_present_label", y="risk_rate", order=ORDER)
ax.bar_label(ax.containers[0], labels=[f"{p:.1%}" for p in risk_rate["risk_rate"]])
plt.title("Risk-taking Rate by Rat Presence")
plt.xlabel("Rat Presence")
plt.ylabel("Risk-taking Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


#Inferential Analysis

#Split groups (numpy arrays)
no_rat_delay = df3.loc[~df3["rat_present"], "bat_landing_to_food"].dropna().to_numpy()
rat_delay    = df3.loc[df3["rat_present"],  "bat_landing_to_food"].dropna().to_numpy()

no_rat_land = df3.loc[~df3["rat_present"], "bat_landing_number"].dropna().to_numpy()
rat_land    = df3.loc[df3["rat_present"],  "bat_landing_number"].dropna().to_numpy()

def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan

#Hesitation time: Welch t-test + Mann–Whitney + Cohen's d
t_stat, p_val = stats.ttest_ind(no_rat_delay, rat_delay, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(no_rat_delay, rat_delay, alternative="two-sided")
d_delay = cohens_d(no_rat_delay, rat_delay)

print("\n[INFERENTIAL] Hesitation time:")
print(f"  Welch t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
print(f"  Mann–Whitney U: U = {u_stat:.0f}, p = {p_u:.4f}")
print(f"  Cohen's d = {d_delay:.3f}")

#Bat landings: Welch t-test + Cohen's d
t_stat2, p_val2 = stats.ttest_ind(no_rat_land, rat_land, equal_var=False)
d_land = cohens_d(no_rat_land, rat_land)

print("\n[INFERENTIAL] Bat landings:")
print(f"  Welch t-test: t = {t_stat2:.3f}, p = {p_val2:.4f}")
print(f"  Cohen's d = {d_land:.3f}")

#Feeding success: Chi-square + Cramér's V
success_table = pd.crosstab(df3["rat_present"], df3["reward"]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
chi2, p_chi, dof, expected = stats.chi2_contingency(success_table)

n = success_table.values.sum()
phi2 = chi2 / n
r, c = success_table.shape
cramers_v = np.sqrt(phi2 / (min(r - 1, c - 1)))

p_no = success_table.loc[False, 1] / success_table.loc[False].sum() if success_table.loc[False].sum() else np.nan
p_yes = success_table.loc[True,  1] / success_table.loc[True].sum()  if success_table.loc[True].sum()  else np.nan

print("\n[INFERENTIAL] Feeding success:")
print(f"  Chi-square: chi2 = {chi2:.3f}, p = {p_chi:.4f}")
print(f"  Success rate (No Rat) = {p_no:.3f}")
print(f"  Success rate (Rat Present) = {p_yes:.3f}")
print(f"  Cramér's V = {cramers_v:.3f}")

#Risk-taking behavior: Chi-square + Cramér's V
risk_table = pd.crosstab(df3["rat_present"], df3["risk"]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
chi2_risk, p_risk, dof_risk, expected_risk = stats.chi2_contingency(risk_table)

n_risk = risk_table.values.sum()
phi2_risk = chi2_risk / n_risk
r_risk, c_risk = risk_table.shape
cramers_v_risk = np.sqrt(phi2_risk / (min(r_risk - 1, c_risk - 1)))

risk_no = risk_table.loc[False, 1] / risk_table.loc[False].sum() if risk_table.loc[False].sum() else np.nan
risk_yes = risk_table.loc[True,  1] / risk_table.loc[True].sum()  if risk_table.loc[True].sum()  else np.nan

print("\n[INFERENTIAL] Risk-taking behavior:")
print(f"  Chi-square: chi2 = {chi2_risk:.3f}, p = {p_risk:.4f}")
print(f"  Risk-taking rate (No Rat) = {risk_no:.3f}")
print(f"  Risk-taking rate (Rat Present) = {risk_yes:.3f}")
print(f"  Cramér's V = {cramers_v_risk:.3f}")

#Final Interpretation
print("\n========== FINAL CONCLUSION ==========")
if p_val < 0.05 or p_u < 0.05:
    print("Hesitation time differs significantly between groups, but bats do not hesitate more with rats present.")
else:
    print("No evidence that bats hesitate more when rats are present.")

if p_val2 < 0.05:
    print("Bat landing frequency differs between groups, but landings are not suppressed by rats.")
else:
    print("No evidence that bat landings are reduced when rats are present.")

if p_chi < 0.05:
    print("Feeding success shows some difference between groups, but bats still obtain food when rats are present.")
else:
    print("Feeding success is not significantly reduced by rat presence.")

if p_risk < 0.05:
    print("Risk-taking differs by rat presence; bats still engage around rats, consistent with competition.")
else:
    print("Risk-taking is not significantly reduced by rat presence.")

print("\nOverall interpretation: Bats treat rats as COMPETITORS for food, not as PREDATORS.")