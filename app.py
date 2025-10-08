
import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = "acpof.db"

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def ensure_schema(conn):
    with open("schema.sql", "r", encoding="utf-8") as f:
        conn.executescript(f.read())

def load_units(conn):
    return dict(conn.execute("SELECT abbreviation, unit_id FROM units").fetchall())

def page_dashboard(conn):
    st.title("ACPOF — Tableau de bord")
    recipes = pd.read_sql_query("SELECT COUNT(*) as n FROM recipes", conn)["n"][0]
    ings = pd.read_sql_query("SELECT COUNT(*) as n FROM ingredients", conn)["n"][0]
    st.metric("Recettes", recipes)
    st.metric("Ingrédients", ings)

def page_import_ingredients(conn):
    st.header("Importer des ingrédients (CSV)")
    st.caption("Colonnes attendues : name,cost_per_unit,unit_default")
    uploaded = st.file_uploader("Choisir un CSV d'ingrédients", type=["csv"], key="ingcsv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Importer ingrédients"):
            units = load_units(conn)
            rows = []
            for _, r in df.iterrows():
                unit_id = units.get(str(r.get("unit_default","")).lower())
                cost = r.get("cost_per_unit")
                try:
                    cost = float(cost) if pd.notna(cost) else None
                except:
                    cost = None
                rows.append((str(r.get("name","")).strip(), unit_id, cost))
            cur = conn.executemany(
                "INSERT OR IGNORE INTO ingredients(name, unit_default, cost_per_unit) VALUES(?,?,?)",
                rows,
            )
            conn.commit()
            st.success(f"Ingrédients importés: {cur.rowcount} (ligne(s) traitée(s))")

def page_import_recipes(conn):
    st.header("Importer une recette (CSV)")
    st.caption("Téléchargez le gabarit, remplissez-le, puis importez-le.")
    try:
        with open("recette_import_template.csv", "rb") as f:
            st.download_button("Télécharger le gabarit CSV", f, file_name="recette_import_template.csv")
    except FileNotFoundError:
        st.warning("Le gabarit n'est pas présent à côté de l'app. Téléchargez-le depuis la conversation.")
    up = st.file_uploader("Choisir un CSV de recette au format du gabarit", type=["csv"], key="reccsv")
    if up:
        df = pd.read_csv(up)
        st.write("Aperçu :")
        st.dataframe(df)
        if st.button("Importer la recette"):
            # Première ligne = meta
            meta = df.iloc[0].to_dict()
            name = str(meta.get("recipe_name","")).strip()
            rtype = str(meta.get("type","")).strip() or None
            yield_qty = float(meta.get("yield_qty", 0) or 0)
            yield_unit_abbr = str(meta.get("yield_unit","")).lower().strip()
            unit_id = conn.execute("SELECT unit_id FROM units WHERE abbreviation = ?", (yield_unit_abbr,)).fetchone()
            unit_id = unit_id[0] if unit_id else None

            conn.execute(
                "INSERT OR IGNORE INTO recipes(name, type, yield_qty, yield_unit) VALUES(?,?,?,?)",
                (name, rtype, yield_qty, unit_id)
            )
            rid = conn.execute("SELECT recipe_id FROM recipes WHERE name = ?", (name,)).fetchone()[0]

            # Lignes suivantes = ingrédients
            ing = df.iloc[1:].dropna(subset=["ingredient_name"])
            units = load_units(conn)
            for _, r in ing.iterrows():
                iname = str(r["ingredient_name"]).strip()
                qty = float(r["qty"])
                uabbr = str(r.get("unit","")).lower().strip()
                u_id = units.get(uabbr)
                # S'assurer que l'ingrédient existe
                conn.execute("INSERT OR IGNORE INTO ingredients(name) VALUES(?)", (iname,))
                iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name = ?", (iname,)).fetchone()[0]
                conn.execute(
                    "INSERT INTO recipe_ingredients(recipe_id, ingredient_id, qty, unit_id) VALUES(?,?,?,?)",
                    (rid, iid, qty, u_id)
                )
            conn.commit()
            st.success(f"Recette '{name}' importée avec succès.")

def page_recipes(conn):
    st.header("Recettes")
    df = pd.read_sql_query(\"\\"\n        SELECT r.recipe_id, r.name, r.type, r.yield_qty, u.abbreviation AS yield_unit\n        FROM recipes r LEFT JOIN units u ON u.unit_id = r.yield_unit\n        ORDER BY r.name\n    \"\\"\n, conn)
    st.dataframe(df)

def page_ingredients(conn):
    st.header("Ingrédients")
    df = pd.read_sql_query(\"\\"\n        SELECT i.ingredient_id, i.name, i.cost_per_unit, u.abbreviation AS unit\n        FROM ingredients i LEFT JOIN units u ON u.unit_id = i.unit_default\n        ORDER BY i.name\n    \"\\"\n, conn)
    st.dataframe(df)


def page_recipe_cost(conn):
    import math
    st.header("Coût d'une recette")
    # Liste des recettes
    recipes = pd.read_sql_query("SELECT recipe_id, name, yield_qty, yield_unit FROM recipes ORDER BY name", conn)
    if recipes.empty:
        st.info("Aucune recette. Importez-en une via 'Importer recettes'.")
        return
    # Choix recette
    names = {row["name"]: row["recipe_id"] for _, row in recipes.iterrows()}
    choice = st.selectbox("Choisir une recette", list(names.keys()))
    rid = names[choice]

    # Détail ingrédients + coûts
    df = pd.read_sql_query("""
        SELECT ri.qty, u.abbreviation AS unit,
               i.name AS ingredient, i.cost_per_unit, iu.abbreviation AS ing_unit
        FROM recipe_ingredients ri
        LEFT JOIN units u ON u.unit_id = ri.unit_id
        LEFT JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
        LEFT JOIN units iu ON iu.unit_id = i.unit_default
        WHERE ri.recipe_id = ?
    """, conn, params=(rid,))

    if df.empty:
        st.warning("Cette recette n'a pas encore d'ingrédients.")
        return

    # --- Conversions d'unités ---
    # On considère que cost_per_unit est exprimé par défaut dans l'unité par défaut de l'ingrédient (ing_unit).
    # On convertit la quantité (ri.qty + unit) vers l'unité de l'ingrédient si possible.
    def convert_qty_to_ing_unit(qty, unit, ing_unit):
        if qty is None or (isinstance(qty, float) and math.isnan(qty)):
            return None, False
        if unit == ing_unit or unit is None or ing_unit is None:
            # Même unité ou unité manquante => pas de conversion
            return qty, True if unit == ing_unit else False

        # g <-> kg
        if unit == "kg" and ing_unit == "g":
            return qty * 1000.0, True
        if unit == "g" and ing_unit == "kg":
            return qty / 1000.0, True  # peu probable si cost_per_unit est /kg mais on gère

        # ml <-> l
        if unit == "l" and ing_unit == "ml":
            return qty * 1000.0, True
        if unit == "ml" and ing_unit == "l":
            return qty / 1000.0, True

        # pc (pièce) : pas de conversion auto avec g/ml/l/kg
        return None, False

    conv_ok = []
    qty_conv = []
    for _, row in df.iterrows():
        q, ok = convert_qty_to_ing_unit(row["qty"], row["unit"], row["ing_unit"])
        qty_conv.append(q)
        conv_ok.append(ok and pd.notna(row["cost_per_unit"]))

    df["qty_in_ing_unit"] = qty_conv

    # Coût de ligne = qty_in_ing_unit * cost_per_unit quand conversion possible & cost connu
    def line_cost(row):
        q = row["qty_in_ing_unit"]
        c = row["cost_per_unit"]
        if pd.isna(q) or pd.isna(c):
            return None
        try:
            return float(q) * float(c)
        except:
            return None

    df["line_cost"] = df.apply(line_cost, axis=1)

    # Affichage avec informations de conversion
    st.subheader("Détail")
    st.dataframe(df[["ingredient","qty","unit","qty_in_ing_unit","ing_unit","cost_per_unit","line_cost"]])

    total = df["line_cost"].sum(skipna=True)
    # Récupérer le rendement
    y = conn.execute("SELECT yield_qty, yield_unit FROM recipes WHERE recipe_id = ?", (rid,)).fetchone()
    y_qty, y_unit_id = y
    y_unit = conn.execute("SELECT abbreviation FROM units WHERE unit_id = ?", (y_unit_id,)).fetchone()
    y_unit = y_unit[0] if y_unit else None

    st.metric("Coût total (lot)", f"{total:.2f}$" if total else "—")
    if total and y_qty:
        st.metric("Coût par unité de rendement", f"{total / y_qty:.4f}$ / {y_unit or 'unité'}")

    with st.expander("Règles de conversion utilisées"):
        st.write("• kg → g : × 1000  • g → kg : ÷ 1000  • L → mL : × 1000  • mL → L : ÷ 1000  • Aucune conversion auto avec 'pc'.")


PAGES = {
    "Tableau de bord": page_dashboard,
    "Ingrédients": page_ingredients,
    "Importer ingrédients": page_import_ingredients,
    "Importer recettes": page_import_recipes,
    "Recettes": page_recipes,
    "Coût recette": page_recipe_cost,
}

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Aller à", list(PAGES.keys()))
    conn = get_conn()
    ensure_schema(conn)
    # Seed default units if empty
    existing = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    if existing == 0:
        conn.executemany(
            "INSERT INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()
    PAGES[choice](conn)

if __name__ == "__main__":
    main()
