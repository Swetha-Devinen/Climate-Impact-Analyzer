import sqlite3
import pandas as pd

# Load CSV
df = pd.read_csv("data/weather_monthly_3cities.csv")

# Ensure month column format
df["month"] = pd.to_datetime(df["month"])

# Create database
conn = sqlite3.connect("climate.db")

# Write table
df.to_sql("weather_monthly", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print("Database created successfully.")
print("Table 'weather_monthly' created successfully.")