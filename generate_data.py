"""
Generate a SYNTHETIC appointment dataset for a small dental clinic.

IMPORTANT: No real patient data is used anywhere in this project.
The data is simulated from simple assumptions (documented below) that mimic
patterns commonly reported in the no-show literature: longer booking lead
time, first visits and missing reminders are associated with more no-shows.

Because the data is simulated, the *findings* only show that the analysis
pipeline works. They are not claims about any real clinic.
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ---- Clinic setup (assumptions) -------------------------------------------
START_DATE = "2025-01-04"          # a Saturday
N_WEEKS = 52
OPEN_DAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]  # closed Friday
N_DOCTORS = 9
TARGET_PER_WEEK = 150              # roughly 150 booked appointments per week

# Relative demand per weekday (busier at the start of the week)
DAY_WEIGHT = {"Saturday": 1.15, "Sunday": 1.10, "Monday": 1.10,
              "Tuesday": 1.00, "Wednesday": 0.95, "Thursday": 0.70}

# Appointment hours: morning shift 9-12, evening shift 16-20 (start hour)
HOURS = [9, 10, 11, 12, 16, 17, 18, 19, 20]
HOUR_WEIGHT = np.array([0.8, 1.1, 1.0, 0.7, 1.0, 1.3, 1.4, 1.2, 0.8])
HOUR_WEIGHT = HOUR_WEIGHT / HOUR_WEIGHT.sum()

TYPES = ["Check-up", "Cleaning", "Filling", "Root canal",
         "Orthodontic follow-up", "Consultation", "Whitening"]
TYPE_WEIGHT = np.array([0.20, 0.18, 0.18, 0.08, 0.14, 0.14, 0.08])
TYPE_WEIGHT = TYPE_WEIGHT / TYPE_WEIGHT.sum()

# ---- Build the calendar -----------------------------------------------------
dates = pd.date_range(START_DATE, periods=N_WEEKS * 7, freq="D")
dates = dates[dates.day_name().isin(OPEN_DAYS)]

weights = np.array([DAY_WEIGHT[d] for d in dates.day_name()])
per_day_mean = TARGET_PER_WEEK / sum(DAY_WEIGHT.values())

rows = []
for date, w in zip(dates, weights):
    n = rng.poisson(per_day_mean * w)
    for _ in range(n):
        rows.append({"date": date})

df = pd.DataFrame(rows)
n = len(df)

df["day_of_week"] = df["date"].dt.day_name()
df["hour"] = rng.choice(HOURS, size=n, p=HOUR_WEIGHT)
df["doctor_id"] = rng.choice([f"Dr_{i:02d}" for i in range(1, N_DOCTORS + 1)], size=n)
df["appointment_type"] = rng.choice(TYPES, size=n, p=TYPE_WEIGHT)
df["new_patient"] = rng.random(n) < 0.25

# Booking lead time in days: many short, some long (capped at 45)
df["lead_days"] = np.minimum(rng.geometric(p=0.12, size=n) - 1, 45)

# Reminder (SMS/call) was sent for about 65% of appointments
df["reminder_sent"] = rng.random(n) < 0.65

df["age_group"] = rng.choice(["0-17", "18-34", "35-54", "55+"], size=n,
                             p=[0.18, 0.35, 0.30, 0.17])

# ---- Outcome model (assumptions) -----------------------------------------
# Log-odds of a no-show. Signs follow the usual direction found in the
# no-show literature; sizes are chosen by me.
logit = (
    -1.65
    + 0.035 * df["lead_days"]
    + 0.55 * df["new_patient"]
    - 0.85 * df["reminder_sent"]
    + 0.25 * df["hour"].isin([9, 20])
    + 0.30 * (df["day_of_week"] == "Thursday")
    + 0.30 * (df["appointment_type"] == "Consultation")
    - 0.40 * (df["appointment_type"] == "Root canal")
    + 0.25 * (df["age_group"] == "18-34")
)
p_no_show = 1 / (1 + np.exp(-logit))

u = rng.random(n)
cancel_p = 0.07                                   # cancelled in advance
df["status"] = np.where(u < cancel_p, "Cancelled",
                np.where(rng.random(n) < p_no_show, "No-show", "Attended"))

df = df.sort_values(["date", "hour"]).reset_index(drop=True)
df.insert(0, "appointment_id", [f"A{i:05d}" for i in range(1, len(df) + 1)])
df["date"] = df["date"].dt.date

df.to_csv("data/appointments_synthetic.csv", index=False)
print(f"Saved {len(df):,} synthetic appointments -> data/appointments_synthetic.csv")
print(df["status"].value_counts(normalize=True).round(3))
