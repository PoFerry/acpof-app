import streamlit as st
import pandas as pd
import sqlite3
import re

DB_FILE = "data.db"

# ---------------------------
# INIT & SCHEMA
# ---------------------------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS units (
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            abbreviation TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS ingredients (
            ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
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
            name TEXT NOT NULL UNIQUE,
            type TEXT,
            yield_qty REAL,
            yield_unit INTEGER,
            FOREIGN KEY (yield_unit) REFERENCES units(unit_id)
        );

        CREATE TABLE IF NOT EXISTS recipe_ingredients (
            recipe_ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER NOT NULL,
            ingredient_id INTEGER NOT NULL,
            quantity REAL,
            unit INTEGER,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE,
            FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id),
            FOREIGN KEY (unit) REFERENCES units(unit_id)
        );

        CREATE INDEX IF NOT EXISTS idx_ri_recipe ON recipe_ingredients(recipe_id);
        CREATE INDEX IF NOT EXISTS idx_ing_name ON ingredients(name);
        """)
        # seed units
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

def unit_id_by_abbr(conn, abbr: str):
    if not abbr:
        return None
    a = str(abbr).strip().lower()
    aliases = {
        "/g": "g", "gramme": "g", "grammes": "g",
        "/kg": "kg",
        "/ml": "ml",
        "/l": "l", "litre": "l", "litres": "l",
        "/unite": "pc", "unite": "pc", "pièce": "pc", "piece": "pc",
        "portion": "pc", "/portion": "pc",
    }
    a = aliases.get(a, a)
    row = conn.execute("SELECT unit_id FROM units WHERE LOWER(abbreviation)=?", (a,)).fetchone()
    return row[0] if row else None

def parse_money(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    s = s.replace('\u00A0', '').replace(' ', '').replace('$', '').replace('CAD', '')
    s = s.replace(',', '.')
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    try:
        return float(s)
    except:
        return None

def to_float(x, default=None):
    if x is None:
        return default
    s = str(x).strip().replace('\u00A0', '')
    s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return default

# ---------------------------
# PAGES
# ---------------------------
def show_home():
    st.title("🧾 Gestion ACPOF – Recettes & Ingrédients")
    st.markdown("""
- 📦 **Importer ingrédients** : ajoute/actualise tes ingrédients (coût unitaire + unité par défaut)  
- 🧑‍🍳 **Importer recettes** : crée tes recettes (nom, type, rendement, unité)  
- 💰 **Coût recette** : calcule automatiquement le coût (avec conversions g↔kg, ml↔L)
    """)

def show_import_ingredients():
    st.header("📦 Importer les ingrédients")
    st.caption("Colonnes reconnues (au minimum **name** + **cost_per_unit** + **unit_default** ou équivalent) : "
               "`name`, `category`, `supplier`, `cost_per_unit`, `unit_default` "
               "— ou leurs variantes francisées comme dans ton fichier Google Sheet.")

    uploaded = st.file_uploader("Téléverse ton fichier CSV d'ingrédients", type=["csv"])
    if not uploaded:
        return

    df_raw = pd.read_csv(uploaded)
    st.subheader("Aperçu du CSV importé")
    st.dataframe(df_raw.head())

    # détection souple des colonnes
    cols = {c.lower().strip(): c for c in df_raw.columns}

    def pick(*keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None

    name_col = pick(
        "name", "description de produit", "description de produit *", "description de produit\t",
        "description de produit  ", "description de produit"
    )
    if not name_col:
        # dernier recours: heuristique
        for c in df_raw.columns:
            cl = c.lower()
            if "description" in cl or "produit" in cl or cl.startswith("nom"):
                name_col = c; break

    category_col = pick("category", "catégorie *", "catégorie", "categorie *", "categorie")
    supplier_col = pick("supplier", "nom fournisseur", "fournisseur")

    cost_col = pick("cost_per_unit", "prix pour recette", "prix unitaire produit")
    price_pkg_col = pick("prix du format d'achat", "prix du format dachat", "prix format achat", "prix format d'achat")
    qty_unit_col = None
    for c in cols:
        if "qté unité" in c or "qte unité" in c or "qte unite" in c or "qté_unité" in c or c == "qté unité *":
            qty_unit_col = cols[c]; break
    unit_default_col = pick("unit_default", "udm d'inventaire", "format d'inventaire", "udm", "unité", "unite")

    if not name_col:
        st.error("Impossible de trouver la colonne du nom d’ingrédient (ex: 'name' ou 'Description de produit').")
        return

    def norm_str(x):
        if pd.isna(x): return None
        s = str(x).strip()
        return s if s else None

    df = pd.DataFrame({
        "name": df_raw[name_col].apply(norm_str),
        "category": df_raw[category_col].apply(norm_str) if category_col else None,
        "supplier": df_raw[supplier_col].apply(norm_str) if supplier_col else None,
    })

    # coût unitaire: direct ou estimé
    if cost_col:
        df["cost_per_unit"] = df_raw[cost_col].apply(parse_money)
    elif price_pkg_col:
        price = df_raw[price_pkg_col].apply(parse_money)
        if qty_unit_col:
            try:
                qty_unit = pd.to_numeric(
                    df_raw[qty_unit_col].astype(str).str.replace(',', '.').str.replace('\u00A0','').str.replace(' ', ''),
                    errors="coerce"
                )
            except Exception:
                qty_unit = pd.to_numeric(df_raw[qty_unit_col], errors="coerce")
            df["cost_per_unit"] = (price / qty_unit).replace([pd.NA, pd.NaT], None)
        else:
            df["cost_per_unit"] = None
    else:
        df["cost_per_unit"] = None

    with sqlite3.connect(DB_FILE) as conn:
        # s'assurer des unités
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        # unité par défaut -> id
        if unit_default_col:
            unit_ids = df_raw[unit_default_col].apply(lambda x: unit_id_by_abbr(conn, x))
        else:
            def guess_from_name(n):
                if not isinstance(n, str): return None
                s = n.lower()
                if "/kg" in s: return unit_id_by_abbr(conn, "kg")
                if "/g" in s: return unit_id_by_abbr(conn, "g")
                if "/ml" in s: return unit_id_by_abbr(conn, "ml")
                if "/l" in s or "/litre" in s: return unit_id_by_abbr(conn, "l")
                if "/unité" in s or "/unite" in s or "/pc" in s: return unit_id_by_abbr(conn, "pc")
                return None
            unit_ids = df["name"].apply(guess_from_name)

        df["unit_default"] = unit_ids
        df = df[df["name"].notna()]

        inserted = 0
        updated = 0
        for _, r in df.iterrows():
            name = r["name"]
            category = r.get("category")
            supplier = r.get("supplier")
            cost = r.get("cost_per_unit")
            unit_id = int(r["unit_default"]) if pd.notna(r.get("unit_default")) else None

            row = conn.execute("SELECT ingredient_id FROM ingredients WHERE name = ?", (name,)).fetchone()
            if row:
                conn.execute("""
                    UPDATE ingredients
                    SET category = COALESCE(?, category),
                        supplier = COALESCE(?, supplier),
                        cost_per_unit = COALESCE(?, cost_per_unit),
                        unit_default = COALESCE(?, unit_default)
                    WHERE name = ?
                """, (category, supplier, cost, unit_id, name))
                updated += 1
            else:
                conn.execute("""
                    INSERT INTO ingredients(name, category, supplier, cost_per_unit, unit_default)
                    VALUES (?,?,?,?,?)
                """, (name, category, supplier, cost, unit_id))
                inserted += 1

        conn.commit()

    st.success(f"Ingrédients traités : {inserted} insérés, {updated} mis à jour.")

def show_import_recipes():
    st.header("🧑‍🍳 Importer les recettes (tolérant aux champs manquants)")

    # --- A) Recettes (métadonnées) ---
    st.subheader("A) Métadonnées des recettes")
    st.caption("CSV recommandé : recipe_name, type, yield_qty, yield_unit (unit parmi g|kg|ml|l|pc). Seul recipe_name est strictement requis.")

    up_meta = st.file_uploader("Téléverse ton CSV de recettes (métadonnées)", type=["csv"], key="recipes_meta")
    if up_meta:
        df = pd.read_csv(up_meta)
        st.dataframe(df.head())

        lc = {c.lower().strip(): c for c in df.columns}
        if "recipe_name" not in lc:
            st.error("Colonne 'recipe_name' obligatoire.")
        else:
            with sqlite3.connect(DB_FILE) as conn:
                inserted = updated = skipped = 0
                reasons = []
                for _, row in df.iterrows():
                    name = str(row[lc["recipe_name"]]).strip() if pd.notna(row[lc["recipe_name"]]) else ""
                    if not name:
                        skipped += 1; reasons.append("Ligne sans recipe_name")
                        continue
                    rtype = str(row[lc.get("type","type")]).strip() if "type" in lc and pd.notna(row[lc["type"]]) else None
                    yqty  = to_float(row[lc["yield_qty"]], default=None) if "yield_qty" in lc else None
                    yabbr = str(row[lc["yield_unit"]]).strip().lower() if "yield_unit" in lc and pd.notna(row[lc["yield_unit"]]) else None
                    yuid  = unit_id_by_abbr(conn, yabbr) if yabbr else None

                    exists = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name,)).fetchone()
                    if exists:
                        conn.execute("""UPDATE recipes
                                        SET type=COALESCE(?,type),
                                            yield_qty=COALESCE(?,yield_qty),
                                            yield_unit=COALESCE(?,yield_unit)
                                        WHERE name=?""", (rtype, yqty, yuid, name))
                        updated += 1
                    else:
                        conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                                     (name, rtype, yqty, yuid))
                        inserted += 1
                conn.commit()
            st.success(f"Recettes: {inserted} insérées, {updated} mises à jour, {skipped} ignorées.")
            if reasons:
                with st.expander("Pourquoi certaines lignes ont été ignorées ?"):
                    st.write("\n".join(reasons))

    st.markdown("---")

    # --- B) Lignes recette ↔ ingrédients ---
    st.subheader("B) Lignes recette ↔ ingrédients")
    st.caption("CSV recommandé : recipe_name, ingredient_name, quantity, unit (unit parmi g|kg|ml|l|pc). Seuls recipe_name et ingredient_name sont strictement requis.")

    up_lines = st.file_uploader("Téléverse ton CSV de lignes (ingrédients par recette)", type=["csv"], key="recipes_lines")
    if up_lines:
        df = pd.read_csv(up_lines)
        st.dataframe(df.head())

        lc = {c.lower().strip(): c for c in df.columns}
        need = ["recipe_name", "ingredient_name"]
        if not all(k in lc for k in need):
            st.error("Colonnes requises manquantes : il faut au minimum 'recipe_name' et 'ingredient_name'.")
            return

        with sqlite3.connect(DB_FILE) as conn:
            inserted_links = created_recipes = created_ingredients = skipped = 0
            reasons = []
            for _, r in df.iterrows():
                rec_name = str(r[lc["recipe_name"]]).strip() if pd.notna(r[lc["recipe_name"]]) else ""
                ing_name = str(r[lc["ingredient_name"]]).strip() if pd.notna(r[lc["ingredient_name"]]) else ""
                qty = to_float(r[lc["quantity"]], default=None) if "quantity" in lc else None
                uabbr = str(r[lc["unit"]]).strip().lower() if "unit" in lc and pd.notna(r[lc["unit"]]) else None
                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                if not rec_name or not ing_name:
                    skipped += 1; reasons.append("Ligne sans recipe_name ou ingredient_name")
                    continue

                # recette
                row_rec = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
                if row_rec:
                    rid = row_rec[0]
                else:
                    conn.execute("INSERT INTO recipes(name) VALUES(?)", (rec_name,))
                    rid = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()[0]
                    created_recipes += 1

                # ingrédient
                row_ing = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                if row_ing:
                    iid = row_ing[0]
                else:
                    conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                    iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]
                    created_ingredients += 1

                conn.execute("""INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                                VALUES (?,?,?,?)""", (rid, iid, qty, uid))
                inserted_links += 1

            conn.commit()

        st.success(f"Lignes importées : {inserted_links}. Nouvelles recettes : {created_recipes}. Nouveaux ingrédients : {created_ingredients}. Ignorées : {skipped}.")
        if reasons:
            with st.expander("Détails des lignes ignorées"):
                st.write("\n".join(reasons))
        st.info("Va maintenant dans **Coût recette** pour voir le calcul (conversions kg↔g, L↔mL).")


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

def show_recipe_costs():
    st.header("💰 Coût des recettes (avec conversions)")

    with sqlite3.connect(DB_FILE) as conn:
        recipes = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)

    if recipes.empty:
        st.warning("Aucune recette trouvée. Importe d’abord tes recettes.")
        return

    choice = st.selectbox("Choisis une recette :", recipes["name"])
    rid = recipes.loc[recipes["name"] == choice, "recipe_id"].iloc[0]

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT ri.quantity AS qty, u.abbreviation AS unit,
                   i.name AS ingredient, i.cost_per_unit, iu.abbreviation AS ing_unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u  ON u.unit_id  = ri.unit
            LEFT JOIN units iu ON iu.unit_id = i.unit_default
            WHERE ri.recipe_id = ?
        """, conn, params=(rid,))

    if df.empty:
        st.info("Aucun ingrédient n’est encore lié à cette recette.")
        st.caption("Prochain module possible : formulaire pour lier ingrédients ↔ recettes.")
        return

    # conversions de base
    def convert(qty, unit, ing_unit):
        if pd.isna(qty) or qty is None:
            return None
        unit = (unit or "").lower()
        ing_unit = (ing_unit or "").lower()
        if unit == ing_unit:
            return qty
        if unit == "kg" and ing_unit == "g":
            return qty * 1000.0
        if unit == "g" and ing_unit == "kg":
            return qty / 1000.0
        if unit == "l" and ing_unit == "ml":
            return qty * 1000.0
        if unit == "ml" and ing_unit == "l":
            return qty / 1000.0
        # pas de conversion automatique avec 'pc'
        return None

    df["qty_in_ing_unit"] = df.apply(lambda r: convert(r["qty"], r["unit"], r["ing_unit"]), axis=1)

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

    st.subheader("Détail")
    st.dataframe(df[["ingredient","qty","unit","qty_in_ing_unit","ing_unit","cost_per_unit","line_cost"]])

    total = df["line_cost"].sum(skipna=True)
    st.metric("Coût total (lot)", f"{total:.2f} $" if pd.notna(total) else "—")

    st.caption("Conversions utilisées : kg↔g (×/÷1000), L↔mL (×/÷1000). Aucune conversion automatique avec 'pc'.")
def show_fix_recipes():
    st.header("🛠️ Corriger / compléter les recettes")

    with sqlite3.connect(DB_FILE) as conn:
        recs = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)

    if recs.empty:
        st.info("Aucune recette dans la base. Importe d'abord des recettes.")
        return

    choice = st.selectbox("Choisir une recette :", recs["name"])
    rid = recs.loc[recs["name"] == choice, "recipe_id"].iloc[0]

    with sqlite3.connect(DB_FILE) as conn:
        # métadonnées
        meta = conn.execute("SELECT name, type, yield_qty, yield_unit FROM recipes WHERE recipe_id=?", (rid,)).fetchone()
        units = pd.read_sql_query("SELECT unit_id, abbreviation FROM units ORDER BY abbreviation", conn)

    st.subheader("Métadonnées")
    name = st.text_input("Nom de la recette", value=meta[0] or "")
    rtype = st.text_input("Type", value=meta[1] or "")
    yqty = st.number_input("Rendement (quantité)", value=float(meta[2]) if meta[2] is not None else 0.0, min_value=0.0, step=0.1, format="%.3f")
    unit_map = dict(units.values)  # id -> abbr
    rev_unit_map = {v:k for k,v in unit_map.items()}
    yunit_abbr = unit_map.get(meta[3], None)
    yunit = st.selectbox("Unité de rendement", [""] + list(rev_unit_map.keys()), index=([""] + list(rev_unit_map.keys())).index(yunit_abbr or "") if yunit_abbr else 0)

    # ingrédients de la recette
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT ri.recipe_ingredient_id, i.name AS ingredient, ri.quantity, u.abbreviation AS unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u ON u.unit_id = ri.unit
            WHERE ri.recipe_id = ?
            ORDER BY ri.recipe_ingredient_id
        """, conn, params=(rid,))

    st.subheader("Ingrédients de la recette")
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="edit_lines",
                            column_config={
                                "recipe_ingredient_id": st.column_config.Column(disabled=True),
                                "ingredient": st.column_config.TextColumn(help="Nom exact de l’ingrédient existant (ou nouveau)"),
                                "quantity": st.column_config.NumberColumn(format="%.3f"),
                                "unit": st.column_config.TextColumn(help="g, kg, ml, l, pc")
                            })

    if st.button("💾 Enregistrer les modifications"):
        with sqlite3.connect(DB_FILE) as conn:
            # maj meta
            yunit_id = rev_unit_map.get(yunit) if yunit else None
            conn.execute("UPDATE recipes SET name=?, type=?, yield_qty=?, yield_unit=? WHERE recipe_id=?",
                         (name.strip() or None, rtype.strip() or None, yqty if yqty>0 else None, yunit_id, rid))

            # synchroniser les lignes
            # 1) supprimer celles qui n'existent plus
            current_ids = set([int(x) for x in edited["recipe_ingredient_id"].dropna().tolist()])
            existing_ids = set([r[0] for r in conn.execute("SELECT recipe_ingredient_id FROM recipe_ingredients WHERE recipe_id=?", (rid,)).fetchall()])
            to_delete = existing_ids - current_ids
            for _id in to_delete:
                conn.execute("DELETE FROM recipe_ingredients WHERE recipe_ingredient_id=?", (int(_id),))

            # 2) upsert chaque ligne
            for _, r in edited.iterrows():
                ing_name = str(r.get("ingredient") or "").strip()
                qty = to_float(r.get("quantity"), default=None)
                uabbr = str(r.get("unit") or "").strip().lower()
                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                if not ing_name:
                    continue

                row_ing = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                if row_ing:
                    iid = row_ing[0]
                else:
                    conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                    iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]

                if pd.notna(r.get("recipe_ingredient_id")):
                    # update
                    conn.execute("""UPDATE recipe_ingredients
                                    SET ingredient_id=?, quantity=?, unit=?
                                    WHERE recipe_ingredient_id=?""", (iid, qty, uid, int(r["recipe_ingredient_id"])))
                else:
                    # insert
                    conn.execute("""INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                                    VALUES (?,?,?,?)""", (rid, iid, qty, uid))
            conn.commit()

        st.success("Modifications enregistrées ✅")

# ---------------------------
# MAIN
# ---------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à :",
        [
            "Accueil",
            "Importer ingrédients",
            "Importer recettes",
            "Corriger recettes",          # <-- ajoute ceci
            "Liste des ingrédients",
            "Liste des recettes",
            "Coût recette",
        ],
    )

    init_db()

    if page == "Accueil":
        show_home()
    elif page == "Importer ingrédients":
        show_import_ingredients()
    elif page == "Importer recettes":
        show_import_recipes()
    elif page == "Corriger recettes":       # <-- et ceci
        show_fix_recipes()
    elif page == "Liste des ingrédients":
        show_ingredients()
    elif page == "Liste des recettes":
        show_recipes()
    elif page == "Coût recette":
        show_recipe_costs()
