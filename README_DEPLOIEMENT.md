# ACPOF — Gestion ingrédients & recettes (Streamlit)

## Déploiement via GitHub + Streamlit Community Cloud

### 1) Préparez ce dépôt
Placez ces fichiers à la racine du repo :
- `app.py`
- `schema.sql`
- `requirements.txt`
- (optionnel) `recette_import_template.csv`
- (optionnel) données CSV à importer (`ingredients_import.csv`, etc.)

> **Note sur SQLite :** la base `acpof.db` est créée automatiquement si absente.
> Le stockage de Streamlit Cloud est **éphémère** : à chaque rebuild, les données locales peuvent repartir à zéro.
> Pour persister, utilisez un backend externe (ex. PostgreSQL) ou réimportez vos CSV à chaque déploiement.

### 2) Poussez sur GitHub
```
git init
git add .
git commit -m "Initial commit ACPOF"
git branch -M main
git remote add origin https://github.com/<votre-user>/<votre-repo>.git
git push -u origin main
```

### 3) Déployez sur Streamlit Cloud
1. Allez sur https://share.streamlit.io
2. **New app** → choisissez votre repo, branche `main`, fichier **`app.py`**
3. **Deploy**

### 4) Importez vos données
Dans l’app :
- onglet **Importer ingrédients** pour charger `ingredients_import.csv`
- onglet **Importer recettes** pour charger les recettes (via gabarit CSV)
- onglet **Coût recette** pour le calcul des coûts (avec conversions g↔kg, ml↔L)

### 5) Persistance des données (optionnel)
Pour garder les données entre déploiements, envisagez :
- une base **PostgreSQL** managée (Neon, Supabase, etc.) et adaptez le code de connexion,
- ou un **Google Sheet** comme source/“sauvegarde” (à connecter via API).