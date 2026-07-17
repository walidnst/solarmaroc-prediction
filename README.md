# ☀️ SolarMaroc — Prédiction de production solaire

Système de prédiction de la production d'énergie solaire au Maroc, basé sur des données météo historiques et une prévision météo en temps réel — sans dépendre de mesures d'irradiance, indisponibles en pratique avant que la production n'ait lieu.

**Villes couvertes :** Béni Mellal · Ouarzazate · Casablanca

---

## Le problème

Le Maroc investit massivement dans les énergies renouvelables (Noor Ouarzazate, stratégie 52% de renouvelable d'ici 2030), mais l'énergie solaire est **intermittente** : la production dépend du soleil, des nuages, de la saison. Sans prévision fiable, il est difficile pour un opérateur d'anticiper combien d'électricité solaire sera disponible demain, et donc de bien équilibrer le réseau électrique.

**Ce projet répond à une question précise :** peut-on prédire la production solaire de demain **sans connaître l'irradiance exacte à l'avance** — seulement à partir des prévisions météo classiques (température, nébulosité, humidité, vent), les seules informations réellement disponibles avant que la journée n'arrive ?

## Ce que fait le projet

1. **Collecte** des données historiques météo + production solaire (PVGIS, Open-Meteo) pour 3 villes marocaines
2. **Entraîne** un modèle de Machine Learning (XGBoost) à prédire la production solaire à partir de la météo seule
3. **Valide** le modèle avec une méthodologie rigoureuse (validation croisée temporelle + optimisation d'hyperparamètres)
4. **Expose** le modèle via une API Flask, connectée en direct à une vraie prévision météo (Open-Meteo Forecast)
5. **Affiche** les résultats dans un dashboard interactif

## Résultats

| Ville | R² (prévision réaliste, sans irradiance) | RMSE |
|---|---|---|
| Béni Mellal | ~0.92 | ~0.04 kWh |
| Ouarzazate | ~0.98 (après optimisation) | ~0.04 kWh |
| Casablanca | ~0.83 | ~0.10 kWh |

*Le score plus bas à Casablanca reflète une réalité physique : les zones côtières marocaines connaissent une météo plus variable (brouillard marin, humidité) que les zones continentales comme Ouarzazate ou Béni Mellal — le modèle capture bien cette différence.*

Un modèle "trivial" utilisant l'irradiance mesurée directement atteint un R² > 0.99, mais cette information n'est jamais disponible à l'avance en conditions réelles — ce projet se concentre donc volontairement sur le scénario honnête de prévision J-1.

## Architecture

```
Météo (Open-Meteo Forecast) → Modèle XGBoost → API Flask → Dashboard
```

- **Collecte** : `collecte_donnees_solaire.py` — récupère données historiques PVGIS + Open-Meteo
- **Modélisation** : `exploration_modelisation_solaire.ipynb` — exploration, entraînement, validation croisée
- **API** : `app.py` — sert les prédictions (`/predict-production`, `/predict-forecast`)
- **Interface** : `templates/dashboard.html` — dashboard interactif (météo réelle + simulation manuelle)

## Stack technique

- **Data & ML** : Python, pandas, scikit-learn, XGBoost
- **API** : Flask
- **Frontend** : HTML/CSS/JS (vanilla, SVG pour les visualisations)
- **Sources de données** : [PVGIS](https://re.jrc.ec.europa.eu/) (JRC, Commission Européenne), [Open-Meteo](https://open-meteo.com/) (historique + prévisions)

## Installation

```bash
git clone https://github.com/walidnst/solarmaroc-prediction.git
cd solarmaroc-prediction
pip install flask joblib pandas scikit-learn xgboost requests matplotlib
python app.py
```

Puis ouvrir : `http://127.0.0.1:5000/dashboard`

## Ré-entraîner les modèles

Ouvrir `exploration_modelisation_solaire.ipynb`, modifier la variable `VILLE` (`Beni_Mellal`, `Ouarzazate`, ou `Casablanca`), puis exécuter toutes les cellules. Les modèles sont automatiquement sauvegardés dans `models/`.

## Pistes d'amélioration

- Déploiement en ligne (Render / Railway)
- Intervalle de confiance sur les prédictions
- Prise en compte de la taille réelle d'une installation (actuellement calibré sur 1 kWc)
- Extension à d'autres villes marocaines

---

*Projet réalisé dans le cadre d'un parcours Data Science / Intelligence Artificielle.*
