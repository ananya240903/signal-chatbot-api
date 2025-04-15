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
import time


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
df = pd.read_csv("https://raw.githubusercontent.com/ananya240903/signal-chatbot-api/main/signal_db3_augmented.csv", on_bad_lines='skip')
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

custom_coords = {
    "Banasthali Airstrip": (26.40783060957509, 75.86939154262777),
    "Banasthali Book Centre": (26.402421726676017, 75.87613606642658),
    "Central Library": (26.40435302911838, 75.87229935997807),
    "Gaushala": (26.405111984013715, 75.87824867914411),
    "Guest House": (26.40130688000608, 75.87826750422454),
    "Horse Riding Ground": (26.405701064357952, 75.87355848330054),
    "Jeev Mandir": (26.404923615655182, 75.87311301815674),
    "Judo Centre": (26.40052983240833, 75.87252094672506),
    "Mukteshwari Canteen": (26.403213389599003, 75.87082940871284),
    "Nav Mandir": (26.402458102635947, 75.87791319788387),
    "New Market": (26.401113654085417, 75.87549388828356),
    "Pragya Mandir": (26.402292220520362, 75.87725651660315),
    "Shakuntalam Cricket Ground": (26.406114522995495, 75.8709052264225),
    "Sharda Mandir": (26.40689104223488, 75.86598816528763),
    "Shiksha Mandir": (26.405491076517077, 75.87560703951348),
    "Shri Shanta Alay": (26.401683490790134, 75.87030816595512),
    "Shri Shanta Ashray": (26.404056623212323, 75.86519922421328),
    "Shri Shanta Ayanam": (26.39930969554158, 75.87637390850108),
    "Shri Shanta Gram": (26.400502933029593, 75.87179251072723),
    "Shri Shanta Kulum": (26.400919068981533, 75.8719879371432),
    "Shri Shanta Lok": (26.398253887574533, 75.87594574581476),
    "Shri Shanta Nikay": (26.40006276403094, 75.8773948542158),
    "Shri Shanta Nikunj": (26.402421548883332, 75.87179235799961),
    "Shri Shanta Sanjavanam": (26.39798383200246, 75.87784984565353),
    "Shri Shanta Sthanam": (26.39684299087043, 75.8762846002069),
    "Shri Shanta Teertham": (26.39726047515216, 75.87562744863783),
    "Shri Shanta Uthjam": (26.397147273246826, 75.87677840098743),
    "Shri Shanta Vishwa Needam": (26.40260794209718, 75.87108246913711),
    "Surya Mandir": (26.404697577583146, 75.87133614126525),
    "Swimming Pool": (26.406121868058506, 75.86981526130005),
    "Utkarsh Mandir": (26.405176422727095, 75.87638569931153),
    "Vigyan Mandir": (26.404605915849633, 75.87386359296549)
}


def extract_sim_provider(text):
    # Convert input text to lowercase
    text = text.lower()

    # Loop through all SIM providers (which are already lowercase)
    for sim in le_sim.classes_:
        if sim in text:
            return sim
    return None  # SIM provider not found in message

geocode_cache = {}

def get_lat_lng_from_place(place_name):
    # ✅ 1. Use hardcoded values if available
    if place_name in custom_coords:
        logger.debug(f"Using custom_coords for {place_name}")
        return custom_coords[place_name]

    # ✅ 2. Check cached values
    if place_name in geocode_cache:
        return geocode_cache[place_name]

    # ✅ 3. Try live geocoding
    try:
        full_query = f"{place_name}, Banasthali, Rajasthan, India"
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": full_query, "format": "json", "limit": 1}
        headers = {"User-Agent": "Mozilla/5.0 signal-location-bot/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=5)
        logger.debug(f"Nominatim query: {response.url}")

        if response.ok and response.json():
            data = response.json()
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            geocode_cache[place_name] = (lat, lon)
            return lat, lon

        # Fallback
        logger.warning(f"No match for '{place_name}', fallback to Banasthali only")
        fallback_query = "Banasthali, Rajasthan, India"
        fallback_resp = requests.get(url, params={"q": fallback_query, "format": "json", "limit": 1}, headers=headers)
        if fallback_resp.ok and fallback_resp.json():
            data = fallback_resp.json()
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            geocode_cache[place_name] = (lat, lon)
            return lat, lon

    except Exception as e:
        logger.error(f"Geocoding failed for {place_name}: {e}")

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
        sim_query = sim_query.lower()
        sim_code = le_sim.transform([sim_query])[0]
    except KeyError as e:
        logger.error(f"SIM provider '{sim_query}' not in label encoder.")
        return jsonify({"error": f"SIM provider '{sim_query}' not recognized."}), 400

    filtered_df = df[df["Sim"] == sim_code]
    logger.debug(f"Filtered DataFrame shape: {filtered_df.shape}")

    if filtered_df.empty:
        logger.warning("No data available for this SIM.")
        return jsonify({"error": "No data available for this SIM."}), 404

    # Top 3 locations with best average signal strength
    grouped = filtered_df.groupby("Location")["Strength"].mean().reset_index()
    top_locations = grouped.sort_values(by="Strength", ascending=False).head(3)

    results = []
    seen_locations = set()

    for _, row in top_locations.iterrows():
        loc_code = int(row["Location"]) if isinstance(row["Location"], float) else row["Location"]
        loc_name = location_mapping.get(loc_code)

        if not loc_name or loc_name in seen_locations:
            continue

        seen_locations.add(loc_name)

        strength = round(row["Strength"], 2)
        lat, lng = get_lat_lng_from_place(loc_name)

        if lat == 0.0 or lng == 0.0:
            logger.warning(f"Skipping {loc_name} due to missing coordinates.")
            continue

        results.append({
            "location": loc_name,
            "avg_signal_strength": strength,
            "latitude": lat,
            "longitude": lng,
            "maps_url": f"https://www.google.com/maps?q={lat},{lng}"
        })

    if not results:
        logger.warning("No valid location recommendations found.")
        return jsonify({"error": "No recommendations could be generated."}), 500

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
