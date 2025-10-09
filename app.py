# ===============================================================
# Atelier Culinaire Pierre-Olivier Ferry – Gestion des recettes
# Version raffinée claire avec logo, thème et bandeau dynamique
# ===============================================================

import streamlit as st
import sqlite3
import pandas as pd
import re
from pathlib import Path
from pandas.errors import ParserError
from typing import Optional, Tuple
from streamlit.runtime.media_file_storage import MediaFileStorageError

# ---------- Configuration générale ----------
st.set_page_config(
    page_title="Atelier Culinaire Pierre-Olivier Ferry – Gestion des recettes",
    layout="wide",
    page_icon="🍞"
)

# ---------- Constantes ----------
DB_FILE = Path("data/acpof.db")
LOGO_PATH = Path("assets/Logo_atelierPOF.png")
DEBUG_MODE = True

# ---------- Thème clair raffiné (CSS) ----------
def inject_custom_theme():
    st.markdown(
        """
        <style>
            /* --- Fond général --- */
            body, .stApp {
                background-color: #faf9f7;
                color: #2b2b2b;
                font-family: "Helvetica Neue", "Segoe UI", sans-serif;
            }

            /* --- En-tête bandeau --- */
            .header-container {
                background: linear-gradient(90deg, #f4f1ed 0%, #fff 100%);
                border-bottom: 1px solid #ddd;
                padding: 0.8rem 1.5rem;
                border-radius: 0 0 12px 12px;
                display: flex;
                align-items: center;
            }
            .header-logo {
                width: 90px;
                margin-right: 1rem;
                border-radius: 8px;
            }
            .header-title {
                font-size: 1.6rem;
                font-weight: 600;
                color: #4b3f35;
            }
            .header-subtitle {
                font-size: 1.1rem;
                font-weight: 400;
                color: #7d746a;
            }

            /* --- Titres des sections --- */
            h1, h2, h3 {
                color: #3a322b;
            }

            /* --- Tableaux --- */
            .stDataFrame {
                border-radius: 8px;
                background-color: #ffffff;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }

            /* --- Boutons --- */
            button[kind="primary"] {
                background-color: #c49e62 !important;
                color: white !important;
                border: none;
                border-radius: 8px;
            }
            button[kind="primary"]:hover {
                background-color: #b58c4f !important;
            }

            /* --- Champs de saisie --- */
            .stTextInput, .stNumberInput {
                background-color: white !important;
            }

            /* --- Barre latérale --- */
            section[data-testid="stSidebar"] {
                background-color: #f8f7f5;
                border-right: 1px solid #e0dcd8;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Injecte le thème dès le démarrage
inject_custom_theme()

# ---------- Bandeau supérieur avec logo et titre dynamique ----------
def app_header(title: str, subtitle: str = ""):
    """Affiche le bandeau supérieur avec logo, titre principal et sous-titre."""
    c1, c2 = st.columns([1, 5])
    with c1:
        try:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width='stretch')
            else:
                st.caption("Logo manquant : assets/Logo_atelierPOF.png")
        except MediaFileStorageError:
            st.caption("Problème de lecture du logo.")
    with c2:
        st.markdown(
            f"""
            <div class="header-container">
                <div>
                    <div class="header-title">Atelier Culinaire Pierre-Olivier Ferry – Gestion des recettes</div>
                    <div class="header-subtitle">{subtitle}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- Fonctions utilitaires de base ----------
def clean_text(x):
    """Normalise les textes : enlève espaces insécables, trim."""
    if x is None:
        return ""
    return str(x).replace("\u00A0", " ").strip()

def to_float_safe(x) -> Optional[float]:
    """Convertit une valeur texte en float de manière robuste."""
    s = clean_text(x)
    if s == "" or s.lower() == "#value!":
        return None
    s = re.sub(r"[^\d,.\-]+", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def connect():
    """Connexion SQLite avec activation des clés étrangères."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# ---------- Initialisation de la base ----------
def ensure_db():
    """Crée les tables si elles n’existent pas déjà."""
    with connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS units(
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            abbreviation TEXT UNIQUE
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS ingredients(
            ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            unit_default INTEGER,
            cost_per_unit REAL,
            supplier TEXT,
            category TEXT,
            FOREIGN KEY(unit_default) REFERENCES units(unit_id)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS recipes(
            recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            type TEXT,
            yield_qty REAL,
            yield_unit INTEGER,
            sell_price REAL,
            FOREIGN KEY(yield_unit) REFERENCES units(unit_id)
        )""")
        conn.commit()
# ===============================================================
# Partie 2 / 3 — Importation ingrédients + recettes et conversions
# ===============================================================

# ---------- Fonctions unités ----------
def map_unit_text_to_abbr(u: str) -> Optional[str]:
    """Mappe les variantes d'unités vers une abréviation standard."""
    s = clean_text(u).lower()
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e")
    aliases = {
        "g": "g", "gramme": "g", "grammes": "g", "gr": "g",
        "kg": "kg", "kilogramme": "kg", "kilogrammes": "kg",
        "ml": "ml", "millilitre": "ml", "millilitres": "ml",
        "l": "l", "litre": "l", "litres": "l",
        "pc": "pc", "piece": "pc", "pieces": "pc",
        "unite": "pc", "unites": "pc", "portion": "pc"
    }
    return aliases.get(s, s)

UNIT_GROUP = {"g": "mass", "kg": "mass", "ml": "vol", "l": "vol", "pc": "pc"}

def same_group(u1: str, u2: str) -> bool:
    """Vérifie si deux unités appartiennent au même groupe (masse, volume, pièce)."""
    if not u1 or not u2:
        return False
    return UNIT_GROUP.get(u1.lower()) == UNIT_GROUP.get(u2.lower())

def convert_qty(qty: float, from_u: str, to_u: str) -> Optional[float]:
    """Convertit une quantité entre unités compatibles (ex: kg↔g, l↔ml)."""
    if qty is None or not from_u or not to_u:
        return None
    f, t = from_u.lower(), to_u.lower()
    if f == t: return qty
    if f == "kg" and t == "g": return qty * 1000.0
    if f == "g" and t == "kg": return qty / 1000.0
    if f == "l" and t == "ml": return qty * 1000.0
    if f == "ml" and t == "l": return qty / 1000.0
    return None

def convert_unit_price(cpu: Optional[float], from_u: Optional[str], to_u: Optional[str]) -> Optional[float]:
    """Convertit un coût unitaire exprimé par 'from_u' en coût par 'to_u'."""
    if cpu is None or pd.isna(cpu) or from_u is None or to_u is None:
        return None
    try:
        cpu = float(cpu)
    except Exception:
        return None
    qty_from_for_one_to = convert_qty(1.0, to_u, from_u)
    if qty_from_for_one_to is None:
        return None
    return cpu * qty_from_for_one_to

# ---------- Lecture CSV robuste ----------
def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Lit un CSV uploadé, en détectant le séparateur automatiquement."""
    if uploaded_file is None:
        raise ValueError("Aucun fichier à lire")
    def _read(**kwargs) -> pd.DataFrame:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, dtype=str, **kwargs).fillna("")
    try:
        return _read(sep=None, engine="python")
    except (ParserError, UnicodeDecodeError, ValueError):
        return _read()

def build_column_lookup(df: pd.DataFrame) -> dict:
    """Crée une table de correspondance nom_normalisé → nom original."""
    def normalise(name: str) -> str:
        return " ".join(str(name).strip().lower().split())
    return {normalise(col): col for col in df.columns}

# ---------- Gestion de l'import combiné ----------
def show_import_data():
    """Page combinée pour importer ingrédients et recettes."""
    app_header("Importation des données", "Importer ingrédients et recettes")
    st.info(
        "💡 Cette page permet d’importer à la fois les **ingrédients** et les **recettes** à partir de fichiers CSV."
    )

    tabs = st.tabs(["📦 Ingrédients", "🧑‍🍳 Recettes"])

    # --- Onglet ingrédients ---
    with tabs[0]:
        st.subheader("Importer les ingrédients")
        st.caption("Colonnes reconnues : **Description de produit**, **UDM d’inventaire**, **Prix unitaire produit**, **Nom fournisseur**, **Catégorie**.")
        up = st.file_uploader("Téléverser le CSV d’ingrédients", type=["csv"], key="import_ing")
        if up:
            try:
                df = read_uploaded_csv(up)
            except Exception as exc:
                st.error(f"Lecture du CSV impossible : {exc}")
                return
            st.dataframe(df.head(), width="stretch")

            colmap = build_column_lookup(df)
            col_name = colmap.get("description de produit") or list(df.columns)[0]
            col_unit = colmap.get("udm d'inventaire") or colmap.get("unité") or colmap.get("unite")
            col_cost = colmap.get("prix unitaire produit") or colmap.get("coût") or colmap.get("cout") or colmap.get("prix")
            col_sup = colmap.get("nom fournisseur")
            col_cat = colmap.get("catégorie") or colmap.get("categorie")

            inserted, updated = 0, 0
            with connect() as conn:
                for _, row in df.iterrows():
                    name = clean_text(row.get(col_name, ""))
                    if not name:
                        continue
                    unit = map_unit_text_to_abbr(row.get(col_unit, ""))
                    uid = conn.execute("SELECT unit_id FROM units WHERE abbreviation=?", (unit,)).fetchone()
                    if uid: uid = uid[0]
                    cost = to_float_safe(row.get(col_cost))
                    sup = clean_text(row.get(col_sup, ""))
                    cat = clean_text(row.get(col_cat, ""))

                    r = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (name,)).fetchone()
                    if r:
                        conn.execute("""
                            UPDATE ingredients SET
                            unit_default=COALESCE(?, unit_default),
                            cost_per_unit=COALESCE(?, cost_per_unit),
                            supplier=COALESCE(?, supplier),
                            category=COALESCE(?, category)
                            WHERE name=?""",
                            (uid, cost, sup, cat, name))
                        updated += 1
                    else:
                        conn.execute("""
                            INSERT INTO ingredients(name, unit_default, cost_per_unit, supplier, category)
                            VALUES (?,?,?,?,?)""",
                            (name, uid, cost, sup, cat))
                        inserted += 1
                conn.commit()
            st.success(f"✅ {inserted} ingrédients ajoutés, {updated} mis à jour.")

    # --- Onglet recettes ---
    with tabs[1]:
        st.subheader("Importer les recettes")
        st.caption("Colonnes reconnues : **Titre de la recette**, **Type**, **Rendement**, **Format rendement**, **Ingrédient 1/Format/Quantité**, etc.")
        up = st.file_uploader("Téléverser le CSV des recettes", type=["csv"], key="import_rec")
        if up:
            try:
                df = read_uploaded_csv(up)
            except Exception as exc:
                st.error(f"Lecture du CSV impossible : {exc}")
                return
            st.dataframe(df.head(), width="stretch")
            colmap = build_column_lookup(df)

            TITLE = colmap.get("titre de la recette") or colmap.get("titre")
            TYPE = colmap.get("type de recette") or colmap.get("type")
            YQTY = colmap.get("rendement de la recette") or colmap.get("rendement")
            YUNIT = colmap.get("format rendement") or colmap.get("format de rendement")

            if not TITLE:
                st.error("Colonne 'Titre de la recette' introuvable.")
                return

            inserted, updated = 0, 0
            with connect() as conn:
                for _, row in df.iterrows():
                    name = clean_text(row.get(TITLE, ""))
                    if not name:
                        continue
                    rtype = clean_text(row.get(TYPE, ""))
                    yqty = to_float_safe(row.get(YQTY))
                    yabbr = map_unit_text_to_abbr(row.get(YUNIT, ""))
                    yuid = None
                    if yabbr:
                        r = conn.execute("SELECT unit_id FROM units WHERE abbreviation=?", (yabbr,)).fetchone()
                        if r: yuid = r[0]

                    existing = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name,)).fetchone()
                    if existing:
                        conn.execute("UPDATE recipes SET type=?, yield_qty=?, yield_unit=? WHERE recipe_id=?",
                                     (rtype, yqty, yuid, existing[0]))
                        updated += 1
                    else:
                        conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                                     (name, rtype, yqty, yuid))
                        inserted += 1
                conn.commit()
            st.success(f"✅ {inserted} recettes ajoutées, {updated} mises à jour.")
# ===============================================================
# Partie 3 / 3 — Recettes, coût, planification et navigation
# ===============================================================

# ---------- Calcul du coût d’une recette ----------
def compute_recipe_cost(recipe_id: int) -> Tuple[Optional[float], list]:
    """Calcule le coût total d’une recette selon les ingrédients."""
    total = 0.0
    issues = []
    with connect() as conn:
        r = conn.execute("SELECT name, yield_qty, yield_unit FROM recipes WHERE recipe_id=?", (recipe_id,)).fetchone()
        if not r:
            return None, ["Recette introuvable"]
        recipe_name, yqty, yunit = r

        lines = conn.execute("""
            SELECT i.name, i.cost_per_unit, u.abbreviation
            FROM ingredients i
            LEFT JOIN units u ON i.unit_default = u.unit_id
        """).fetchall()

        for (iname, cpu, unit) in lines:
            if cpu is None:
                issues.append(f"Coût manquant pour {iname}")
            else:
                total += cpu  # ici simplifié : un ingrédient = coût direct
    return (total if total > 0 else None), issues


# ---------- Planification des achats ----------
def page_purchase_planner():
    """Page pour planifier les achats selon le nombre de recettes."""
    app_header("Planification des achats", "Estimer les besoins d’ingrédients")
    with connect() as conn:
        recipes = pd.read_sql_query("SELECT recipe_id, name, yield_qty, yield_unit FROM recipes", conn)

    if recipes.empty:
        st.warning("Aucune recette trouvée. Importez d’abord vos recettes.")
        return

    recipe_choice = st.selectbox("Choisir une recette :", recipes["name"])
    selected = recipes[recipes["name"] == recipe_choice].iloc[0]
    base_yield = selected["yield_qty"]
    st.write(f"**Rendement de base :** {base_yield} portions")

    multiplier = st.number_input("Nombre de fois à préparer :", min_value=1, value=1)
    st.divider()
    st.markdown("### Liste prévisionnelle des ingrédients")

    with connect() as conn:
        df = pd.read_sql_query("SELECT name, cost_per_unit, supplier, category FROM ingredients", conn)
        df["Quantité totale estimée"] = base_yield * multiplier
        df["Coût total estimé"] = df["cost_per_unit"].fillna(0) * df["Quantité totale estimée"]

    st.dataframe(df, width="stretch")
    st.success("Liste de besoins calculée.")

# ---------- Pages diverses ----------
def page_home():
    app_header("Accueil", "Bienvenue dans votre espace de gestion culinaire")
    st.markdown("""
        ### 🍽️ Bienvenue à l’Atelier Culinaire Pierre-Olivier Ferry
        Cette application vous permet de :
        - Gérer vos **ingrédients** et leurs coûts unitaires  
        - Créer et importer vos **recettes**  
        - Calculer automatiquement les **coûts de revient**  
        - Planifier vos **achats** selon vos menus  
        
        💾 Toutes les données sont conservées localement dans `data/acpof.db`.
    """)

def page_manage_ingredients():
    app_header("Ingrédients", "Liste et gestion")
    with connect() as conn:
        df = pd.read_sql_query("SELECT * FROM ingredients", conn)
    st.dataframe(df, width="stretch")
    st.caption(f"{len(df)} ingrédients trouvés.")

def page_recipe_costs():
    app_header("Coût des recettes", "Estimation automatique du coût de revient")
    with connect() as conn:
        df = pd.read_sql_query("SELECT recipe_id, name FROM recipes", conn)
    if df.empty:
        st.warning("Aucune recette enregistrée.")
        return
    recipe_choice = st.selectbox("Choisir une recette :", df["name"])
    rid = int(df[df["name"] == recipe_choice]["recipe_id"].iloc[0])
    cost, issues = compute_recipe_cost(rid)
    if cost:
        st.success(f"💰 Coût estimé : **{cost:.2f} $**")
    else:
        st.warning("Coût non calculable (ingrédients manquants).")
    if issues:
        st.info("⚠️ Points à vérifier :")
        st.write(issues)

def page_create_recipe():
    app_header("Créer une recette", "Saisie manuelle d’une nouvelle recette")
    with connect() as conn:
        units = pd.read_sql_query("SELECT * FROM units", conn)
    name = st.text_input("Nom de la recette")
    rtype = st.text_input("Type (ex: dessert, plat principal, etc.)")
    yqty = st.number_input("Rendement (quantité totale)", min_value=0.0, value=1.0)
    yunit = st.selectbox("Unité de rendement", units["abbreviation"])
    if st.button("Enregistrer la recette"):
        with connect() as conn:
            uid = conn.execute("SELECT unit_id FROM units WHERE abbreviation=?", (yunit,)).fetchone()
            uid = uid[0] if uid else None
            conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                         (name, rtype, yqty, uid))
            conn.commit()
        st.success(f"Recette '{name}' enregistrée.")

def page_view_recipes():
    app_header("Consulter les recettes", "Liste complète des recettes")
    with connect() as conn:
        df = pd.read_sql_query("SELECT * FROM recipes", conn)
    st.dataframe(df, width="stretch")

def page_edit_recipe():
    app_header("Corriger une recette", "Modifier les informations d’une recette existante")
    with connect() as conn:
        df = pd.read_sql_query("SELECT recipe_id, name, type, yield_qty FROM recipes", conn)
    if df.empty:
        st.warning("Aucune recette à corriger.")
        return
    recipe_choice = st.selectbox("Sélectionnez une recette :", df["name"])
    row = df[df["name"] == recipe_choice].iloc[0]
    new_type = st.text_input("Type", row["type"])
    new_yield = st.number_input("Rendement", min_value=0.0, value=row["yield_qty"])
    if st.button("Mettre à jour"):
        with connect() as conn:
            conn.execute("UPDATE recipes SET type=?, yield_qty=? WHERE recipe_id=?",
                         (new_type, new_yield, row["recipe_id"]))
            conn.commit()
        st.success("Recette mise à jour.")

# ---------- Navigation principale ----------
def main():
    ensure_db()
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Aller à",
        [
            "🏠 Accueil",
            "📦 Importation données",
            "🥕 Ingrédients",
            "🧾 Créer recette",
            "📖 Consulter recettes",
            "✏️ Corriger recette",
            "💰 Coût des recettes",
            "🛒 Planifier achats",
        ],
        index=0
    )

    if page.startswith("🏠"):
        page_home()
    elif page.startswith("📦"):
        show_import_data()
    elif page.startswith("🥕"):
        page_manage_ingredients()
    elif page.startswith("🧾"):
        page_create_recipe()
    elif page.startswith("📖"):
        page_view_recipes()
    elif page.startswith("✏️"):
        page_edit_recipe()
    elif page.startswith("💰"):
        page_recipe_costs()
    elif page.startswith("🛒"):
        page_purchase_planner()

# ---------- Lancement principal ----------
if __name__ == "__main__":
    main()
