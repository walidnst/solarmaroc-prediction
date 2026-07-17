"""
API Flask pour la prédiction de production solaire au Maroc.

Cette API expose le modèle de prévision entraîné en Phase 2 (celui SANS irradiance,
utilisable en conditions réelles avec seulement des prévisions météo classiques).

Installation requise :
    pip install flask joblib pandas scikit-learn xgboost --break-system-packages

Lancement :
    python app.py

L'API démarre sur http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Coordonnées GPS des villes (nécessaires pour interroger l'API météo)
# ----------------------------------------------------------------------------

VILLES_COORDS = {
    "Beni_Mellal": {"lat": 32.3373, "lon": -6.3498},
    "Ouarzazate": {"lat": 30.9189, "lon": -6.8934},
    "Casablanca": {"lat": 33.5731, "lon": -7.5898},
}

# ----------------------------------------------------------------------------
# Chargement des modèles au démarrage (une seule fois, pas à chaque requête)
# ----------------------------------------------------------------------------

DOSSIER_MODELES = "models"
VILLES_DISPONIBLES = ["Beni_Mellal", "Ouarzazate", "Casablanca"]

modeles = {}

for ville in VILLES_DISPONIBLES:
    chemin = f"{DOSSIER_MODELES}/modele_prevision_{ville}_xgboost.pkl"
    if os.path.exists(chemin):
        modeles[ville] = joblib.load(chemin)
        print(f"Modèle chargé pour {ville}")
    else:
        print(f"[ATTENTION] Modèle introuvable pour {ville} : {chemin}")

# Les features doivent être dans le MÊME ORDRE que lors de l'entraînement (section 9 du notebook)
FEATURES_ATTENDUES = ["mois", "jour_semaine", "heure", "jour_annee",
                       "temperature_c", "nebulosite_pct", "humidite_pct", "vent_kmh"]


# ----------------------------------------------------------------------------
# Route de test simple (vérifie que l'API tourne)
# ----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def accueil():
    return jsonify({
        "message": "API de prédiction de production solaire - Maroc",
        "villes_disponibles": list(modeles.keys()),
        "endpoint_prediction": "/predict-production (POST)",
        "dashboard": "/dashboard",
    })


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html", villes=list(modeles.keys()))


# ----------------------------------------------------------------------------
# Route principale : prédiction de production
# ----------------------------------------------------------------------------

@app.route("/predict-production", methods=["POST"])
def predict_production():
    """
    Attend un JSON de la forme :
    {
        "ville": "Beni_Mellal",
        "mois": 7,
        "jour_semaine": 2,
        "heure": 14,
        "jour_annee": 195,
        "temperature_c": 32.5,
        "nebulosite_pct": 10,
        "humidite_pct": 25,
        "vent_kmh": 8.5
    }

    Renvoie :
    {
        "ville": "Beni_Mellal",
        "production_prevue_kwh": 0.42
    }
    """
    donnees = request.get_json(silent=True)

    if donnees is None:
        return jsonify({"erreur": "Le corps de la requête doit être un JSON valide."}), 400

    # Vérification de la ville
    ville = donnees.get("ville")
    if ville not in modeles:
        return jsonify({
            "erreur": f"Ville inconnue ou modèle non chargé : '{ville}'.",
            "villes_disponibles": list(modeles.keys()),
        }), 400

    # Vérification que toutes les features attendues sont présentes
    champs_manquants = [f for f in FEATURES_ATTENDUES if f not in donnees]
    if champs_manquants:
        return jsonify({
            "erreur": "Champs manquants dans la requête.",
            "champs_manquants": champs_manquants,
            "champs_attendus": FEATURES_ATTENDUES,
        }), 400

    # Construction du DataFrame dans le bon ordre de colonnes (important pour le modèle)
    try:
        entree = pd.DataFrame([{f: donnees[f] for f in FEATURES_ATTENDUES}])
    except Exception as e:
        return jsonify({"erreur": f"Erreur de format dans les données : {e}"}), 400

    # Prédiction
    modele = modeles[ville]
    prediction = modele.predict(entree)[0]
    prediction = max(0.0, float(prediction))  # une production ne peut pas être négative

    return jsonify({
        "ville": ville,
        "production_prevue_kwh": round(prediction, 4),
        "entree_utilisee": donnees,
    })


# ----------------------------------------------------------------------------
# Route : prévision automatique à partir de la vraie météo (Open-Meteo)
# ----------------------------------------------------------------------------

@app.route("/predict-forecast", methods=["GET"])
def predict_forecast():
    """
    Récupère la météo réelle des prochains jours (Open-Meteo, gratuit, sans clé)
    et calcule automatiquement la production prévue pour chaque heure.

    Exemple : /predict-forecast?ville=Beni_Mellal&jours=3

    Renvoie :
    {
        "ville": "Beni_Mellal",
        "previsions_horaires": [{"datetime": "...", "temperature_c": ..., "production_kwh": ...}, ...],
        "totaux_par_jour": {"2026-07-17": 4.82, ...}
    }
    """
    ville = request.args.get("ville")
    jours = request.args.get("jours", default=2, type=int)
    jours = max(1, min(jours, 7))  # Open-Meteo limite raisonnablement à 7 jours ici

    if ville not in modeles:
        return jsonify({
            "erreur": f"Ville inconnue ou modèle non chargé : '{ville}'.",
            "villes_disponibles": list(modeles.keys()),
        }), 400

    if ville not in VILLES_COORDS:
        return jsonify({"erreur": f"Coordonnées GPS inconnues pour '{ville}'."}), 400

    coords = VILLES_COORDS[ville]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,cloudcover,wind_speed_10m",
        "timezone": "Africa/Casablanca",
        "forecast_days": jours,
    }

    try:
        reponse = requests.get(url, params=params, timeout=15)
        reponse.raise_for_status()
        donnees_meteo = reponse.json()
    except requests.exceptions.RequestException as e:
        return jsonify({"erreur": f"Impossible de récupérer la météo en direct : {e}"}), 502

    hourly = donnees_meteo.get("hourly")
    if not hourly:
        return jsonify({"erreur": "Réponse météo invalide (pas de données horaires)."}), 502

    # Construction du DataFrame à partir de la météo réelle reçue
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "humidite_pct": hourly["relative_humidity_2m"],
        "nebulosite_pct": hourly["cloudcover"],
        "vent_kmh": hourly["wind_speed_10m"],
    })

    # Mêmes features temporelles que lors de l'entraînement (section 9 du notebook)
    df["mois"] = df["datetime"].dt.month
    df["jour_semaine"] = df["datetime"].dt.dayofweek
    df["heure"] = df["datetime"].dt.hour
    df["jour_annee"] = df["datetime"].dt.dayofyear

    # Prédiction pour toutes les heures d'un coup (plus rapide qu'une par une)
    entree = df[FEATURES_ATTENDUES]
    modele = modeles[ville]
    predictions = modele.predict(entree)
    df["production_kwh"] = [max(0.0, float(p)) for p in predictions]

    previsions_horaires = [
        {
            "datetime": row.datetime.strftime("%Y-%m-%d %H:%M"),
            "temperature_c": round(row.temperature_c, 1),
            "nebulosite_pct": round(row.nebulosite_pct, 0),
            "production_kwh": round(row.production_kwh, 4),
        }
        for row in df.itertuples()
    ]

    df["jour"] = df["datetime"].dt.strftime("%Y-%m-%d")
    totaux_par_jour = df.groupby("jour")["production_kwh"].sum().round(3).to_dict()

    return jsonify({
        "ville": ville,
        "previsions_horaires": previsions_horaires,
        "totaux_par_jour": totaux_par_jour,
    })


# ----------------------------------------------------------------------------
# Lancement du serveur
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)