# 📦 Install required libraries if needed (use pip in terminal)
# pip install scikit-learn xgboost pandas matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 📥 Load the dataset
df = pd.read_csv("noshowappointments.csv")

# 🗓 Convert to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce', format='mixed')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce', format='mixed')

# 🧹 Drop invalid date rows
df.dropna(subset=['ScheduledDay', 'AppointmentDay'], inplace=True)

# 🔧 Feature Engineering
df['DaysBetween'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['AppointmentWeekDay'] = df['AppointmentDay'].dt.dayofweek
df['ScheduledHour'] = df['ScheduledDay'].dt.hour
df['IsWeekend'] = df['AppointmentWeekDay'].apply(lambda x: 1 if x >= 5 else 0)
df['Missed'] = df['No-show'].map({'Yes': 1, 'No': 0})

# 🎯 Select features
features = ['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',
            'SMS_received', 'DaysBetween', 'AppointmentWeekDay', 'ScheduledHour', 'IsWeekend']

df_model = df[features + ['Missed']].dropna()

X = df_model[features]
y = df_model['Missed']

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n🎯 AUC Score:", roc_auc_score(y_test, y_proba))

# 💡 Intervention Logic
def suggest_intervention(prob):
    if prob > 0.9:
        return "📞 Call Patient"
    elif prob > 0.75:
        return "📩 Send SMS Reminder"
    elif prob > 0.5:
        return "📱 App Notification"
    else:
        return "✅ No Action Needed"

# 🔍 Prepare output
probs = model.predict_proba(X_test)[:, 1]
interventions = [suggest_intervention(p) for p in probs]

final_df = X_test.copy()
final_df['NoShowProbability'] = probs
final_df['RecommendedAction'] = interventions

# 🔝 Show top 10 high-risk appointments
print("\n🚨 Top 10 High-Risk Appointments:\n")
print(final_df.sort_values(by='NoShowProbability', ascending=False).head(10))
