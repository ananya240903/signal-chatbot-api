from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import os
import logging
import requests


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),  # Save logs to debug.log
        logging.StreamHandler()  # Also print logs to the console
    ]
)
logger = logging.getLogger()

app = Flask(__name__)

# Corrected line:
df = pd.read_csv("signal_db3_augmented.csv", on_bad_lines='skip')
df.columns = df.columns.str.strip()

# Encoding categorical data
le_sim = LabelEncoder()
le_location = LabelEncoder()

# Fit and transform the 'Sim' and 'Location' columns
df["Sim"] = le_sim.fit_transform(df["Sim"])
df["Location"] = le_location.fit_transform(df["Location"])

# Ensure that le_sim.classes_ is in numpy array format
le_sim.classes_ = np.array([s.lower() for s in le_sim.classes_])

# Create mappings for sim and location
sim_mapping = dict(zip(le_sim.classes_, range(len(le_sim.classes_))))
#location_mapping = dict(zip(le_location.classes_, range(len(le_location.classes_))))

location_mapping = dict(zip(range(len(le_location.classes_)), le_location.classes_))
# Train KNN model
X = df[["Location", "Distance to Tower (km)", "SNR", "Sim"]]
y = df["Strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

def extract_sim_provider(text):
    # Convert input text to lowercase
    text = text.lower()

    # Loop through all SIM providers (which are already lowercase)
    for sim in le_sim.classes_:
        if sim in text:
            return sim
    return None  # SIM provider not found in message

def get_lat_lng_from_place(place_name):
    try:
        # Try full query first
        full_query = f"{place_name}, Banasthali, Rajasthan, India"
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": full_query, "format": "json", "limit": 1}
        headers = {"User-Agent": "Mozilla/5.0 signal-location-bot/1.0"}
        
        response = requests.get(url, params=params, headers=headers)
        logger.debug(f"Nominatim full query URL: {response.url}")
        
        if response.ok and response.json():
            data = response.json()
            return float(data[0]["lat"]), float(data[0]["lon"])
        
        # Fallback: try simpler location only
        fallback_query = "Banasthali, Rajasthan, India"
        logger.warning(f"No match for '{place_name}'. Retrying with fallback...")
        response = requests.get(url, params={"q": fallback_query, "format": "json", "limit": 1}, headers=headers)
        
        if response.ok and response.json():
            data = response.json()
            return float(data[0]["lat"]), float(data[0]["lon"])

    except Exception as e:
        logger.error(f"Exception in geocoding '{place_name}': {e}")
    
    return 0.0, 0.0



def build_google_maps_url(location_name):
    return f"https://www.google.com/maps/search/?api=1&query={location_name.replace(' ', '+')}"

@app.route('/chatbot', methods=['POST'])
def chatbot_handler():
    user_input = request.json.get("message", "")
    logger.info(f"Received chatbot message: {user_input}")

    sim_query = extract_sim_provider(user_input)
    logger.info(f"Extracted SIM provider: {sim_query}")
    
    if not sim_query:
        logger.warning("SIM provider not found in message.")
        return jsonify({"error": "SIM provider not found in message."}), 400

    try:
        # Convert sim_query to lowercase before encoding it
        sim_query = sim_query.lower()
        sim_code = le_sim.transform([sim_query])[0]

    except KeyError as e:
        logger.error(f"SIM provider {sim_query} not found in label encoder classes.")
        return jsonify({"error": f"SIM provider '{sim_query}' not recognized."}), 400

    filtered_df = df[df["Sim"] == sim_code]
    logger.debug(f"Filtered DataFrame shape: {filtered_df.shape}")

    if filtered_df.empty:
        logger.warning("No data available for this SIM.")
        return jsonify({"error": "No data available for this SIM."}), 404

    # Get top 3 locations by strength
    grouped = filtered_df.groupby("Location")["Strength"].mean().reset_index()
    top_locations = grouped.sort_values(by="Strength", ascending=False).head(3)

    results = []
    for _, row in top_locations.iterrows():
        loc_code = row["Location"]
        logger.debug(f"Row location: {loc_code}, type: {type(loc_code)}")

        # Ensure the location is an integer (cast to int if it's a float)
        if isinstance(loc_code, float):
            loc_code = int(loc_code)  # Cast float to int
            logger.debug(f"Location code casted to int: {loc_code}")

        # Log the location mapping to inspect any mismatch
        logger.info(f"Location Mapping: {location_mapping}")

        # Check if loc_code exists in location_mapping
        if loc_code not in location_mapping:
            logger.error(f"Location code {loc_code} not found in location_mapping.")
            continue

        loc_name = location_mapping[loc_code]
        strength = round(row["Strength"], 2)
        lat, lng = get_lat_lng_from_place(loc_name)
        maps_url = f"https://www.google.com/maps?q={lat},{lng}"  # Better than search endpoint

        results.append({
            "location": loc_name,
            "avg_signal_strength": strength,
            "latitude": lat,
            "longitude": lng,
            "maps_url": maps_url
        })

    if not results:
        logger.warning("No top locations found or recommended.")
        
    return jsonify({
        "sim": sim_query,
        "recommendations": results
    })


@app.route('/predict', methods=['GET'])
def predict_signal():
    location = request.args.get('location', '').strip().lower()
    logger.info(f"Received request to predict signal for location: {location}")

    matching_locations = [loc for loc in location_mapping.values() if loc.lower() == location]

    if not matching_locations:
        logger.warning(f"Location '{location}' not found.")
        return jsonify({"error": "Location not found."})

    loc_encoded = list(location_mapping.keys())[list(location_mapping.values()).index(matching_locations[0])]
    location_data = df[df["Location"] == loc_encoded]
    logger.debug(f"Filtered data for location '{location}': {location_data.shape}")

    if location_data.empty:
        logger.warning(f"No data available for location: {location}")
        return jsonify({"error": "No data available for this location."}), 404

    predicted_signal = knn.predict([[loc_encoded, location_data["Distance to Tower (km)"].mean(), location_data["SNR"].mean(), location_data["Sim"].mode()[0]]])[0]
    return jsonify({"location": matching_locations[0], "predicted_signal_strength": predicted_signal})

@app.route('/compare', methods=['GET'])
def compare_signals():
    loc1 = request.args.get('location1', '').strip().lower()
    loc2 = request.args.get('location2', '').strip().lower()
    locations = [loc1, loc2]
    logger.info(f"Comparing signals between '{loc1}' and '{loc2}'.")

    loc_encoded = []
    for loc in locations:
        matching = [k for k, v in location_mapping.items() if v.lower() == loc]
        if not matching:
            logger.warning(f"Location '{loc}' not found.")
            return jsonify({"error": f"Location '{loc}' not found."})
        loc_encoded.append(matching[0])

    results = []
    for loc_code in loc_encoded:
        loc_data = df[df['Location'] == loc_code]
        if loc_data.empty:
            results.append("No data available")
            logger.warning(f"No data available for location code: {loc_code}")
        else:
            avg_strength = loc_data['Strength'].mean()
            logger.info(f"Average signal strength for location {loc_code}: {avg_strength}")

            # Get SIM with the best signal (max Strength)
            best_sim_code = loc_data.loc[loc_data['Strength'].idxmax()]['Sim']
            logger.info(f"Best SIM code (raw) for location {loc_code}: {best_sim_code}")

            # Use LabelEncoder to map back the best_sim_code to SIM name
            best_sim = le_sim.inverse_transform([best_sim_code])[0]
            logger.info(f"Best SIM for location {loc_code}: {best_sim}")

            results.append({"avg_strength": avg_strength, "best_sim": best_sim})

    return jsonify({locations[0]: results[0], locations[1]: results[1]})


@app.route('/')
def home():
    logger.info("Home endpoint hit.")
    return "Welcome to the Signal Strength Prediction API. Use /predict, /compare, or /chatbot endpoints."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
