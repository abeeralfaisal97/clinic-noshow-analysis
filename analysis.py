"""
No-show analysis for a (simulated) dental clinic.

Questions
1. What is the no-show rate, and when/for whom is it highest?
2. Do reminders reduce no-shows? (chi-square test)
3. Which factors matter most when considered together? (logistic regression)
4. When are the peak hours, and how is work spread across doctors?

Run:  python analysis.py        (needs data/appointments_synthetic.csv)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

sns.set_theme(style="whitegrid")
DAY_ORDER = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
df = pd.read_csv("data/appointments_synthetic.csv", parse_dates=["date"])
print(f"Appointments: {len(df):,}")
print(df["status"].value_counts().to_string(), "\n")

# Cancelled in advance = the slot could be re-booked, so it is not a no-show.
# Analyse no-shows among appointments that were meant to happen.
d = df[df["status"] != "Cancelled"].copy()
d["no_show"] = (d["status"] == "No-show").astype(int)
overall = d["no_show"].mean()
print(f"Overall no-show rate (excluding cancelled): {overall:.1%}\n")

d["lead_group"] = pd.cut(d["lead_days"], bins=[-1, 2, 7, 14, 45],
                         labels=["0-2 days", "3-7 days", "8-14 days", "15+ days"])
d["period"] = np.where(d["hour"] < 13, "Morning", "Evening")

# ---------------------------------------------------------------------------
# 2. No-show rate by group
# ---------------------------------------------------------------------------
def rate_by(col):
    t = d.groupby(col, observed=True)["no_show"].agg(rate="mean", n="size")
    return t

by_day = rate_by("day_of_week").loc[DAY_ORDER]
by_lead = rate_by("lead_group")
by_reminder = rate_by("reminder_sent")
by_new = rate_by("new_patient")
by_type = rate_by("appointment_type").sort_values("rate", ascending=False)

for name, t in [("Weekday", by_day), ("Booking lead time", by_lead),
                ("Reminder sent", by_reminder), ("New patient", by_new),
                ("Appointment type", by_type)]:
    print(f"No-show rate by {name}:")
    print((t.assign(rate=t["rate"].map("{:.1%}".format))).to_string(), "\n")

# ---------------------------------------------------------------------------
# 3. Hypothesis tests (chi-square test of independence)
# ---------------------------------------------------------------------------
def chi_test(col):
    table = pd.crosstab(d[col], d["no_show"])
    chi2, p, dof, _ = chi2_contingency(table)
    return chi2, p

print("Chi-square tests (H0: no-show is independent of the factor)")
for col in ["reminder_sent", "new_patient", "lead_group", "day_of_week"]:
    chi2, p = chi_test(col)
    print(f"  {col:15s} chi2 = {chi2:7.1f}   p = {p:.2e}")
print()

# ---------------------------------------------------------------------------
# 4. Logistic regression (factors considered together)
# ---------------------------------------------------------------------------
X = pd.DataFrame({
    "lead_days_per_week": d["lead_days"] / 7,
    "new_patient": d["new_patient"].astype(int),
    "reminder_sent": d["reminder_sent"].astype(int),
    "thursday": (d["day_of_week"] == "Thursday").astype(int),
    "early_or_late_hour": d["hour"].isin([9, 20]).astype(int),
    "consultation": (d["appointment_type"] == "Consultation").astype(int),
    "root_canal": (d["appointment_type"] == "Root canal").astype(int),
    "age_18_34": (d["age_group"] == "18-34").astype(int),
})
y = d["no_show"]

model = LogisticRegression(max_iter=1000).fit(X, y)
odds = pd.Series(np.exp(model.coef_[0]), index=X.columns).sort_values()
print("Odds ratios (above 1 = more no-shows, below 1 = fewer):")
print(odds.round(2).to_string(), "\n")

# What if every appointment got a reminder?
X_all_reminded = X.copy()
X_all_reminded["reminder_sent"] = 1
p_now = model.predict_proba(X)[:, 1].sum()
p_all = model.predict_proba(X_all_reminded)[:, 1].sum()
avoided = p_now - p_all
print(f"Reminder coverage now: {d['reminder_sent'].mean():.0%}")
print(f"Expected no-shows now: {p_now:.0f}; with reminders for all: {p_all:.0f}")
print(f"Estimated no-shows avoided per year: {avoided:.0f} "
      f"({avoided / p_now:.0%} of all no-shows)\n")

# ---------------------------------------------------------------------------
# 5. Workload: peak hours and doctors
# ---------------------------------------------------------------------------
heat = df.pivot_table(index="day_of_week", columns="hour", values="appointment_id",
                      aggfunc="count").loc[DAY_ORDER]
per_doc = df.groupby("doctor_id")["appointment_id"].count()
print("Appointments per doctor (year):")
print(per_doc.to_string(), "\n")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def bar(ax, t, title, xlabel):
    sns.barplot(x=t.index.astype(str), y=t["rate"] * 100, ax=ax, color="#3b78b5")
    ax.axhline(overall * 100, color="#c0392b", ls="--", lw=1.2, label=f"Overall {overall:.1%}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("No-show rate (%)")
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bar(axes[0], by_lead, "No-show rate by booking lead time", "Days between booking and visit")
r = by_reminder.copy()
r.index = ["No reminder", "Reminder sent"]
bar(axes[1], r, "No-show rate by reminder", "")
plt.tight_layout()
plt.savefig("figures/01_leadtime_and_reminder.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
bar(ax, by_day, "No-show rate by weekday", "")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figures/02_weekday.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(heat, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Appointments (year)"})
ax.set_title("When is the clinic busiest? Appointments by weekday and hour")
ax.set_xlabel("Start hour")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("figures/03_peak_hours.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#27ae60" if v < 1 else "#c0392b" for v in odds.values]
ax.barh(odds.index, odds.values, color=colors)
ax.axvline(1, color="black", lw=1)
ax.set_title("What drives no-shows? (odds ratios, logistic regression)")
ax.set_xlabel("Odds ratio (1 = no effect)")
plt.tight_layout()
plt.savefig("figures/04_odds_ratios.png", dpi=150)
plt.close()
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=per_doc.index, y=per_doc.values, ax=ax, color="#3b78b5")
ax.set_title("Appointments per doctor (year)")
ax.set_xlabel("")
ax.set_ylabel("Appointments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/05_doctor_workload.png", dpi=150)
plt.close()

print("Figures saved in figures/")
