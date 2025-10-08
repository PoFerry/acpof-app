import streamlit as st
import pandas as pd
import sqlite3
import io
import os

DB_FILE = "data.db"

# --- INITIALISATION DE LA BD ---
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS units (
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            abbreviation TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ingredients (
            ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT,
            supplier TEXT,
            package_size REAL,
            package_unit TEXT,
            price_per_package REAL,
            cost_per_unit REAL,
            unit_default INTEGER,
            FOREIGN KEY (unit_default) REFERENCES units(unit_id)
        );

        CREATE TABLE IF NOT EXISTS recipes (
            recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            yield_qty REAL,
            yield_unit INTEGER,
            FOREIGN KEY (yield_unit) REFERENCES units(unit_id)
        );

        CREATE TABLE IF NOT EXISTS recipe_ingredients (
            recipe_ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            ingredient_id INTEGER,
            quantity REAL,
            unit INTEGER,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
            FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id),
            FOREIGN KEY (unit) REFERENCES units(unit_id)
        );
        """)
        conn.commit()

# --- PAGE D’ACCUEIL ---
def show_home():
    st.title("🧾 Gestion ACPOF – Recettes & Ingrédients")
    st.markdown("""
    Bienvenue dans ton application de gestion culinaire ACPOF 🍽️  
    - 📦 Importer tes **ingrédients** depuis un CSV  
    - 🧑‍🍳 Ajouter ou importer tes **recettes**  
    - 💰 Calculer automatiquement le **coût des recettes**
    """)

# --- IMPORTATION D’INGRÉDIENTS ---
def show_import_ingredients():
    st.header("📦 Importer les ingrédients")

    uploaded = st.file_uploader("Téléverse ton fichier CSV d'ingrédients", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        if st.button("Importer dans la base de données"):
            with sqlite3.connect(DB_FILE) as conn:
                df.to_sql("ingredients", conn, if_exists="append", index=False)
            st.success("Ingrédients importés avec succès !")

# --- IMPORTATION DE RECETTES ---
def show_import_recipes():
    st.header("🧑‍🍳 Importer les recettes")

    uploaded = st.file_uploader("Téléverse ton fichier CSV de recettes", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        if st.button("Importer dans la base de données"):
            with sqlite3.connect(DB_FILE) as conn:
                df.to_sql("recipes", conn, if_exists="append", index=False)
            st.success("Recettes importées avec succès !")

# --- LISTE DES INGRÉDIENTS ---
def show_ingredients():
    st.header("📋 Liste des ingrédients")

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT i.ingredient_id, i.name, i.category, i.supplier, 
                   i.cost_per_unit, u.abbreviation AS unit
            FROM ingredients i 
            LEFT JOIN units u ON u.unit_id = i.unit_default
            ORDER BY i.name
        """, conn)

    st.dataframe(df)

# --- LISTE DES RECETTES ---
def show_recipes():
    st.header("📘 Liste des recettes")

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT r.recipe_id, r.name, r.type, r.yield_qty, 
                   u.abbreviation AS yield_unit
            FROM recipes r 
            LEFT JOIN units u ON u.unit_id = r.yield_unit
            ORDER BY r.name
        """, conn)

    st.dataframe(df)

# --- CALCUL DU COÛT DES RECETTES ---
def show_recipe_costs():
    st.header("💰 Calcul des coûts de recettes")

    with sqlite3.connect(DB_FILE) as conn:
        recipes = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)

    if recipes.empty:
        st.warning("Aucune recette trouvée. Importez d’abord vos recettes.")
        return

    recipe_choice = st.selectbox("Choisis une recette :", recipes["name"])
    selected_id = recipes.loc[recipes["name"] == recipe_choice, "recipe_id"].iloc[0]

    with sqlite3.connect(DB_FILE) as conn:
        query = """
        SELECT i.name AS ingredient, ri.quantity, u.abbreviation AS unit,
               i.cost_per_unit * ri.quantity AS total_cost
        FROM recipe_ingredients ri
        JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
        LEFT JOIN units u ON u.unit_id = ri.unit
        WHERE ri.recipe_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(selected_id,))

    if df.empty:
        st.info("Aucun ingrédient lié à cette recette.")
        return

    df["total_cost"] = df["total_cost"].round(2)
    st.dataframe(df)

    total = df["total_cost"].sum().round(2)
    st.subheader(f"Coût total de la recette : **{total:.2f} $**")

# --- INTERFACE PRINCIPALE ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à :", [
        "Accueil", "Importer ingrédients", "Importer recettes",
        "Liste des ingrédients", "Liste des recettes", "Coût recette"
    ])

    init_db()

    if page == "Accueil":
        show_home()
    elif page == "Importer ingrédients":
        show_import_ingredients()
    elif page == "Importer recettes":
        show_import_recipes()
    elif page == "Liste des ingrédients":
        show_ingredients()
    elif page == "Liste des recettes":
        show_recipes()
    elif page == "Coût recette":
        show_recipe_costs()

if __name__ == "__main__":
    main()
