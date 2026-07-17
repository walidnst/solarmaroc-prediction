"""
Script de collecte de données pour le projet de prédiction de production solaire au Maroc.

Ce script récupère :
1. Les données de production solaire théorique via l'API PVGIS (Union Européenne)
2. Les données météo historiques via l'API Open-Meteo

Puis fusionne les deux en un seul dataset propre, prêt pour l'entraînement d'un modèle ML.

Installation requise :
    pip install requests pandas --break-system-packages

Utilisation :
    python collecte_donnees_solaire.py
"""

import requests
import pandas as pd
from datetime import datetime
import time

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Villes marocaines avec leurs coordonnées GPS (latitude, longitude)
# Tu peux ajouter/modifier des villes selon ton besoin
VILLES = {
    "Beni_Mellal": {"lat": 32.3373, "lon": -6.3498},
    "Ouarzazate": {"lat": 30.9189, "lon": -6.8934},
    "Casablanca": {"lat": 33.5731, "lon": -7.5898},
}

# Période de collecte (Open-Meteo archive couvre plusieurs années en arrière)
# NB : 2020 est utilisée par défaut car les bases de données satellite de PVGIS
# ont souvent 1 à 2 ans de retard - une année trop récente (ex: 2023, 2024) peut
# être rejetée si elle n'est pas encore disponible pour la région choisie.
DATE_DEBUT = "2020-01-01"
DATE_FIN = "2020-12-31"

# Puissance de l'installation PV simulée (en kWc) pour PVGIS
PUISSANCE_PV_KWC = 1.0  # 1 kWc = facile à mettre à l'échelle ensuite

# Dossier de sortie
DOSSIER_SORTIE = "data"


# ----------------------------------------------------------------------------
# 1. COLLECTE PVGIS (production solaire théorique horaire)
# ----------------------------------------------------------------------------

def fetch_pvgis_data(lat: float, lon: float, annee_debut: int, annee_fin: int) -> pd.DataFrame:
    """
    Récupère les données horaires de production PV théorique depuis l'API PVGIS.
    Documentation : https://re.jrc.ec.europa.eu/api/v5_2/seriescalc

    Retourne un DataFrame avec : datetime, irradiance (G(i)), temperature, production_kwh
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": annee_debut,
        "endyear": annee_fin,
        "pvcalculation": 1,       # active le calcul de production PV
        "peakpower": PUISSANCE_PV_KWC,
        "loss": 14,               # pertes système standard (%)
        "angle": 30,              # inclinaison des panneaux (degrés) - proche de la latitude du Maroc
        "aspect": 0,              # orientation : 0 = plein sud (optimal dans l'hémisphère nord)
        "pvtechchoice": "crystSi",  # technologie silicium cristallin (la plus courante)
        "mountingplace": "free",    # montage au sol / libre (pas intégré au bâtiment)
        "trackingtype": 0,          # 0 = panneaux fixes (pas de suivi solaire)
        "raddatabase": "PVGIS-SARAH2",  # base satellite couvrant Europe/Afrique/Asie
        "outputformat": "json",
    }

    print(f"  -> Requête PVGIS pour ({lat}, {lon})...")
    response = requests.get(url, params=params, timeout=60)

    if not response.ok:
        # PVGIS renvoie un message d'erreur détaillé en JSON même sur un 400 -
        # on l'affiche pour comprendre le vrai problème (au lieu du code générique)
        try:
            err_detail = response.json()
            raise RuntimeError(f"PVGIS a refusé la requête : {err_detail}")
        except ValueError:
            raise RuntimeError(f"PVGIS a refusé la requête (HTTP {response.status_code}) : {response.text[:500]}")

    data = response.json()

    hourly = data["outputs"]["hourly"]
    df = pd.DataFrame(hourly)

    # Colonnes utiles : time (format YYYYMMDD:HHMM), P (puissance produite en W),
    # G(i) (irradiance globale inclinée en W/m2), T2m (température à 2m)
    df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")

    # IMPORTANT : PVGIS (base SARAH2) décale ses timestamps de quelques minutes
    # (ex: 00:10 au lieu de 00:00) car c'est l'heure exacte où le satellite
    # "voit" la zone. On arrondit à l'heure pile pour pouvoir fusionner
    # proprement avec les données météo (qui sont pile à l'heure).
    df["datetime"] = df["datetime"].dt.floor("h")

    df = df.rename(columns={
        "P": "production_w",
        "G(i)": "irradiance_wm2",
        "T2m": "temperature_c",
    })
    df["production_kwh"] = df["production_w"] / 1000.0  # conversion W -> kWh (pas horaire)

    return df[["datetime", "irradiance_wm2", "temperature_c", "production_kwh"]]


# ----------------------------------------------------------------------------
# 2. COLLECTE OPEN-METEO (météo historique horaire)
# ----------------------------------------------------------------------------

def fetch_openmeteo_data(lat: float, lon: float, date_debut: str, date_fin: str) -> pd.DataFrame:
    """
    Récupère les données météo horaires historiques depuis l'API Open-Meteo Archive.
    Documentation : https://open-meteo.com/en/docs/historical-weather-api

    Retourne un DataFrame avec : datetime, nebulosite, humidite, vitesse_vent
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_debut,
        "end_date": date_fin,
        "hourly": "cloudcover,relative_humidity_2m,wind_speed_10m",
        "timezone": "UTC",  # doit correspondre au fuseau UTC utilisé par PVGIS
    }

    print(f"  -> Requête Open-Meteo pour ({lat}, {lon})...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "nebulosite_pct": hourly["cloudcover"],
        "humidite_pct": hourly["relative_humidity_2m"],
        "vent_kmh": hourly["wind_speed_10m"],
    })

    return df


# ----------------------------------------------------------------------------
# 3. FUSION DES DEUX SOURCES
# ----------------------------------------------------------------------------

def fusionner_donnees(df_pvgis: pd.DataFrame, df_meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les données PVGIS et Open-Meteo sur la colonne datetime,
    puis ajoute des features temporelles utiles pour le modèle ML.
    """
    df = pd.merge(df_pvgis, df_meteo, on="datetime", how="inner")

    # Le datetime de fusion est en UTC (nécessaire pour aligner PVGIS et Open-Meteo).
    # On ajoute une colonne en heure locale marocaine, plus lisible pour l'analyse.
    df["datetime_local_maroc"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("Africa/Casablanca").dt.tz_localize(None)

    # Features temporelles (importantes pour capter la saisonnalité) - basées sur l'heure locale
    df["heure"] = df["datetime_local_maroc"].dt.hour
    df["jour_semaine"] = df["datetime_local_maroc"].dt.dayofweek  # 0 = lundi
    df["mois"] = df["datetime_local_maroc"].dt.month
    df["jour_annee"] = df["datetime_local_maroc"].dt.dayofyear

    # Réordonner les colonnes pour la lisibilité
    colonnes = [
        "datetime_local_maroc", "mois", "jour_semaine", "heure", "jour_annee",
        "irradiance_wm2", "temperature_c", "nebulosite_pct", "humidite_pct", "vent_kmh",
        "production_kwh",
    ]
    colonnes = [c for c in colonnes if c in df.columns]

    return df[colonnes].sort_values("datetime_local_maroc").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. PIPELINE PRINCIPAL
# ----------------------------------------------------------------------------

def main():
    import os
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    annee_debut = int(DATE_DEBUT[:4])
    annee_fin = int(DATE_FIN[:4])

    for nom_ville, coords in VILLES.items():
        print(f"\n=== Collecte pour {nom_ville} ===")
        try:
            df_pvgis = fetch_pvgis_data(coords["lat"], coords["lon"], annee_debut, annee_fin)
            time.sleep(1)  # pause pour ne pas surcharger l'API

            df_meteo = fetch_openmeteo_data(coords["lat"], coords["lon"], DATE_DEBUT, DATE_FIN)
            time.sleep(1)

            df_final = fusionner_donnees(df_pvgis, df_meteo)

            chemin_sortie = f"{DOSSIER_SORTIE}/{nom_ville}_dataset_solaire.csv"
            df_final.to_csv(chemin_sortie, index=False)

            print(f"  -> {len(df_final)} lignes sauvegardées dans {chemin_sortie}")
            print(f"  -> Aperçu :\n{df_final.head(3)}")

        except (requests.exceptions.RequestException, RuntimeError) as e:
            print(f"  [ERREUR] Échec de la collecte pour {nom_ville} : {e}")
        except Exception as e:
            print(f"  [ERREUR] Problème inattendu pour {nom_ville} : {e}")

    print("\nCollecte terminée. Vérifie le dossier 'data/' pour les fichiers CSV générés.")


if __name__ == "__main__":
    main()