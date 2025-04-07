from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import os

app = Flask(__name__)

df = pd.read_csv("https://github.com/ananya240903/signal-chatbot-api/blob/main/signal_db3_augmented.csv")
df.columns = df.columns.str.strip()

# Encoding categorical data
le_sim = LabelEncoder()
le_location = LabelEncoder()
df["Sim"] = le_sim.fit_transform(df["Sim"])
df["Location"] = le_location.fit_transform(df["Location"])

sim_mapping = dict(zip(le_sim.transform(le_sim.classes_), le_sim.classes_))
location_mapping = dict(zip(le_location.transform(le_location.classes_), le_location.classes_))

# Train KNN model
X = df[["Location", "Distance to Tower (km)", "SNR", "Sim"]]
y = df["Strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

def extract_sim_provider(text):
    sim_providers = [s.lower() for s in le_sim.classes_]
    for sim in sim_providers:
        if sim in text.lower():
            return sim
    return None

def build_google_maps_url(location_name):
    return f"https://www.google.com/maps/search/?api=1&query={location_name.replace(' ', '+')}"

@app.route('/chatbot', methods=['POST'])
def chatbot_handler():
    user_input = request.json.get("message", "")
    sim_query = extract_sim_provider(user_input)

    if not sim_query:
        return jsonify({"error": "SIM provider not found in message."}), 400

    sim_code = le_sim.transform([sim_query])[0]
    filtered_df = df[df["Sim"] == sim_code]

    if filtered_df.empty:
        return jsonify({"error": "No data available for this SIM."}), 404

    # Get top 3 locations by strength
    grouped = filtered_df.groupby("Location")["Strength"].mean().reset_index()
    top_locations = grouped.sort_values(by="Strength", ascending=False).head(3)

    results = []
    for _, row in top_locations.iterrows():
        loc_name = location_mapping[row["Location"]]
        strength = round(row["Strength"], 2)
        maps_url = build_google_maps_url(loc_name)
        results.append({
            "location": loc_name,
            "avg_signal_strength": strength,
            "maps_url": maps_url
        })

    return jsonify({
        "sim": sim_query,
        "recommendations": results
    })

@app.route('/predict', methods=['GET'])
def predict_signal():
    location = request.args.get('location', '').strip().lower()
    matching_locations = [loc for loc in location_mapping.values() if loc.lower() == location]

    if not matching_locations:
        return jsonify({"error": "Location not found."})

    loc_encoded = list(location_mapping.keys())[list(location_mapping.values()).index(matching_locations[0])]
    location_data = df[df["Location"] == loc_encoded]

    predicted_signal = knn.predict([[loc_encoded, location_data["Distance to Tower (km)"].mean(), location_data["SNR"].mean(), location_data["Sim"].mode()[0]]])[0]
    return jsonify({"location": matching_locations[0], "predicted_signal_strength": predicted_signal})

@app.route('/compare', methods=['GET'])
def compare_signals():
    loc1 = request.args.get('location1', '').strip().lower()
    loc2 = request.args.get('location2', '').strip().lower()
    locations = [loc1, loc2]

    loc_encoded = []
    for loc in locations:
        matching = [k for k, v in location_mapping.items() if v.lower() == loc]
        if not matching:
            return jsonify({"error": f"Location '{loc}' not found."})
        loc_encoded.append(matching[0])

    results = []
    for loc_code in loc_encoded:
        loc_data = df[df['Location'] == loc_code]
        if loc_data.empty:
            results.append("No data available")
        else:
            avg_strength = loc_data['Strength'].mean()
            best_sim = loc_data.loc[loc_data['Strength'].idxmax()]['Sim']
            results.append({"avg_strength": avg_strength, "best_sim": sim_mapping[best_sim]})

    return jsonify({locations[0]: results[0], locations[1]: results[1]})

@app.route('/')
def home():
    return "Welcome to the Signal Strength Prediction API. Use /predict, /compare, or /chatbot endpoints."


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
