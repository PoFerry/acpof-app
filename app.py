import streamlit as st
import pandas as pd
import sqlite3

# Nom du fichier SQLite (créé automatiquement)
DB_FILE = "data.db"

# ================================
# Helpers
# ================================
def parse_money(x):
    """Convertit '17,32 $' -> 17.32 (float)."""
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

def unit_id_by_abbr(conn, abbr):
    """Retourne l'unit_id depuis une abréviation flexible (g, kg, ml, l, pc)."""
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

# ================================
# DB init
# ================================
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
        # unités de base
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

# ================================
# Pages
# ================================
def show_home():
    st.title("🧾 Gestion ACPOF — Recettes & Ingrédients")
    st.markdown(
        "- 📦 **Importer ingrédients** : ajoute/actualise tes ingrédients (coût unitaire + unité par défaut)\n"
        "- 🧑‍🍳 **Importer recettes** : importe noms/rendements, puis lignes ingrédient↔recette (tolérant aux manques)\n"
        "- 🛠️ **Corriger recettes** : édite recettes et lignes directement dans l’interface\n"
        "- 💰 **Coût recette** : calcule le coût (avec conversions g↔kg, ml↔L)"
    )

def show_import_ingredients():
    st.header("📦 Importer les ingrédients")
    st.caption(
        "CSV recommandé : `name`, `category`, `supplier`, `cost_per_unit`, `unit_default` "
        "(ou leurs variantes francisées : 'Description de produit', 'Prix pour recette', 'UDM d'inventaire', etc.)."
    )

    up = st.file_uploader("Téléverse ton fichier CSV d'ingrédients", type=["csv"])
    if not up:
        return

    df_raw = pd.read_csv(up)
    st.subheader("Aperçu")
    st.dataframe(df_raw.head())

    # mapping souple des colonnes
    cols = {c.lower().strip(): c for c in df_raw.columns}

    def pick(*keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None

    name_col = pick("name", "description de produit", "description de produit *", "produit", "description")
    if not name_col:
        # heuristique de secours
        for c in df_raw.columns:
            cl = c.lower()
            if "description" in cl or "produit" in cl or cl.startswith("nom"):
                name_col = c; break

    category_col = pick("category", "catégorie *", "catégorie", "categorie *", "categorie")
    supplier_col = pick("supplier", "nom fournisseur", "fournisseur")
    cost_col = pick("cost_per_unit", "prix pour recette", "prix unitaire produit")
    price_pkg_col = pick("prix du format d'achat", "prix format d'achat", "prix format achat")
    qty_unit_col = None
    for key in cols:
        if "qté unité" in key or "qte unité" in key or "qte unite" in key or "qté_unité" in key or key == "qté unité *":
            qty_unit_col = cols[key]; break
    unit_default_col = pick("unit_default", "udm d'inventaire", "format d'inventaire", "udm", "unité", "unite")

    if not name_col:
        st.error("Impossible de trouver la colonne du nom d’ingrédient (ex: 'name' ou 'Description de produit').")
        return

    def norm_str(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        return s if s else None

    df = pd.DataFrame({
        "name": df_raw[name_col].apply(norm_str),
        "category": df_raw[category_col].apply(norm_str) if category_col else None,
        "supplier": df_raw[supplier_col].apply(norm_str) if supplier_col else None,
    })

    # coût unitaire
    if cost_col:
        df["cost_per_unit"] = df_raw[cost_col].apply(parse_money)
    elif price_pkg_col and qty_unit_col:
        price = df_raw[price_pkg_col].apply(parse_money)
        qty = pd.to_numeric(
            df_raw[qty_unit_col].astype(str).str.replace(',', '.').str.replace('\u00A0','').str.replace(' ', ''),
            errors="coerce"
        )
        df["cost_per_unit"] = (price / qty).replace([pd.NA, pd.NaT], None)
    else:
        df["cost_per_unit"] = None

    with sqlite3.connect(DB_FILE) as conn:
        # s'assurer que les unités existent
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        # unité par défaut -> id
        if unit_default_col:
            unit_ids = df_raw[unit_default_col].apply(lambda x: unit_id_by_abbr(conn, x))
        else:
            # essai via le texte du nom
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

        inserted = updated = 0
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
    st.header("🧑‍🍳 Importer les recettes (format large Google Sheet)")

    st.caption("""
    Téléverse le CSV exporté de ton onglet recettes.
    En-têtes attendus (français) repérés automatiquement :
    - Titre de la recette, Type de recette, Rendement de la recette, Format rendement
    - Ingrédient 1..25, Format ingrédient 1..25, Quantité ingrédient 1..25
    Import tolérant : on crée/maj les recettes même si des champs manquent.
    """)

    up = st.file_uploader("Téléverser le CSV", type=["csv"])
    if not up:
        return

    # Auto-détection du séparateur (tabulations / virgules), tolère #VALUE!, vides, etc.
    try:
        df = pd.read_csv(up, sep=None, engine="python", dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(up, dtype=str).fillna("")

    st.subheader("Aperçu")
    st.dataframe(df.head())

    # Normalisation des noms de colonnes (minuscules, sans espaces multiples)
    def norm_col(c):
        return " ".join(str(c).strip().lower().split())

    colmap = {norm_col(c): c for c in df.columns}

    # Colonnes FR usuelles
    TITLE = colmap.get("titre de la recette")
    TYPE = colmap.get("type de recette")
    YQTY = colmap.get("rendement de la recette")
    YUNIT = colmap.get("format rendement")

    if not TITLE:
        st.error("Colonne 'Titre de la recette' introuvable. Vérifie l’export CSV.")
        return

    # utilitaires
    def to_float_safe(x):
        if x is None: return None
        s = str(x).strip().replace("\u00A0","")
        if s == "" or s.lower() == "#value!": return None
        s = s.replace(",", ".")
        try:
            return float(s)
        except:
            return None

    def map_unit_text_to_abbr(u):
        """Mappe les formats français/variantes vers g, kg, ml, l, pc."""
        if not u: return None
        s = str(u).strip().lower()
        aliases = {
            "/g":"g","g":"g","gramme":"g","grammes":"g",
            "/kg":"kg","kg":"kg","kilogramme":"kg",
            "/ml":"ml","ml":"ml",
            "/l":"l","l":"l","litre":"l","litres":"l",
            "/unité":"pc","unité":"pc","unite":"pc","/unite":"pc",
            "portion":"pc","/portion":"pc","pc":"pc",
        }
        return aliases.get(s, s)

    # Préparer compteurs / logs
    meta_ins = meta_upd = 0
    line_ins = new_rec = new_ing = 0
    skipped_meta = 0
    skipped_lines = 0
    reasons_meta = []
    reasons_lines = []

    with sqlite3.connect(DB_FILE) as conn:
        # s’assurer unités
        conn.executemany(
            "INSERT OR IGNORE INTO units(name, abbreviation) VALUES(?,?)",
            [("gramme","g"),("kilogramme","kg"),("millilitre","ml"),("litre","l"),("pièce","pc")]
        )
        conn.commit()

        # --- 1) Métadonnées recettes ---
        for _, row in df.iterrows():
            name = str(row[TITLE]).strip() if TITLE else ""
            if not name:
                skipped_meta += 1; reasons_meta.append("Ligne sans 'Titre de la recette'"); continue

            rtype = (str(row[TYPE]).strip() if TYPE else "") or None
            yqty  = to_float_safe(row[YQTY]) if YQTY else None
            yabbr = map_unit_text_to_abbr(row[YUNIT]) if YUNIT else None
            yuid  = unit_id_by_abbr(conn, yabbr) if yabbr else None

            exists = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (name,)).fetchone()
            if exists:
                conn.execute("""
                    UPDATE recipes
                    SET type=COALESCE(?, type),
                        yield_qty=COALESCE(?, yield_qty),
                        yield_unit=COALESCE(?, yield_unit)
                    WHERE name=?
                """, (rtype, yqty, yuid, name))
                meta_upd += 1
            else:
                conn.execute("INSERT INTO recipes(name, type, yield_qty, yield_unit) VALUES (?,?,?,?)",
                             (name, rtype, yqty, yuid))
                meta_ins += 1
                new_rec += 1
        conn.commit()

        # --- 2) Lignes ingrédient↔recette ---
        # colonnes Ingrédient n / Format ingrédient n / Quantité ingrédient n
        # On scanne n=1..35 pour être large.
        for _, row in df.iterrows():
            rec_name = str(row[TITLE]).strip() if TITLE else ""
            if not rec_name:
                continue
            rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
            if not rid_row:
                # sécurité : ne devrait pas arriver
                conn.execute("INSERT INTO recipes(name) VALUES(?)", (rec_name,))
                rid_row = conn.execute("SELECT recipe_id FROM recipes WHERE name=?", (rec_name,)).fetchone()
                new_rec += 1
            rid = rid_row[0]

            for n in range(1, 36):
                ing_key = f"ingrédient {n}"
                fmt_key = f"format ingrédient {n}"
                qty_key = f"quantité ingrédient {n}"

                ing_col = colmap.get(ing_key)
                fmt_col = colmap.get(fmt_key)
                qty_col = colmap.get(qty_key)

                if not ing_col:
                    continue

                ing_name = str(row[ing_col]).strip()
                if ing_name == "" or ing_name.lower() == "#value!":
                    continue  # pas d’ingrédient -> on saute

                qty = to_float_safe(row[qty_col]) if qty_col else None
                uabbr = map_unit_text_to_abbr(row[fmt_col]) if fmt_col else None
                uid = unit_id_by_abbr(conn, uabbr) if uabbr else None

                # récupérer/créer l’ingrédient
                iid_row = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()
                if iid_row:
                    iid = iid_row[0]
                else:
                    conn.execute("INSERT INTO ingredients(name) VALUES(?)", (ing_name,))
                    iid = conn.execute("SELECT ingredient_id FROM ingredients WHERE name=?", (ing_name,)).fetchone()[0]
                    new_ing += 1

                # insérer la ligne (même si qty/unit manquent -> valeurs NULL tolérées)
                conn.execute("""
                    INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                    VALUES (?,?,?,?)
                """, (rid, iid, qty, uid))
                line_ins += 1

        conn.commit()

    st.success(f"Recettes: {meta_ins} insérées, {meta_upd} mises à jour. Lignes ingréd.: {line_ins}. Nouvelles recettes: {new_rec}. Nouveaux ingrédients: {new_ing}.")
    if skipped_meta:
        with st.expander("Lignes de métadonnées ignorées"):
            st.write("\n".join(reasons_meta))

def show_fix_recipes():
    st.header("🛠️ Corriger / compléter les recettes")
    with sqlite3.connect(DB_FILE) as conn:
        recs = pd.read_sql_query("SELECT recipe_id, name FROM recipes ORDER BY name", conn)
    if recs.empty:
        st.info("Aucune recette. Importe d'abord des recettes.")
        return

    choice = st.selectbox("Choisir une recette :", recs["name"])
    rid = recs.loc[recs["name"] == choice, "recipe_id"].iloc[0]

    with sqlite3.connect(DB_FILE) as conn:
        meta = conn.execute("SELECT name, type, yield_qty, yield_unit FROM recipes WHERE recipe_id=?", (rid,)).fetchone()
        units = pd.read_sql_query("SELECT unit_id, abbreviation FROM units ORDER BY abbreviation", conn)

    st.subheader("Métadonnées")
    name = st.text_input("Nom de la recette", value=meta[0] or "")
    rtype = st.text_input("Type", value=meta[1] or "")
    yqty = st.number_input("Rendement (quantité)", value=float(meta[2]) if meta[2] is not None else 0.0, min_value=0.0, step=0.1, format="%.3f")
    unit_map = dict(units.values)  # id -> abbr
    rev_unit_map = {v: k for k, v in unit_map.items()}
    yunit_abbr = unit_map.get(meta[3], None)
    yunit = st.selectbox("Unité de rendement", [""] + list(rev_unit_map.keys()),
                         index=([""] + list(rev_unit_map.keys())).index(yunit_abbr or "") if yunit_abbr else 0)

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
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="edit_lines",
        column_config={
            "recipe_ingredient_id": st.column_config.Column(disabled=True),
            "ingredient": st.column_config.TextColumn(help="Nom exact de l’ingrédient (existant ou nouveau)"),
            "quantity": st.column_config.NumberColumn(format="%.3f"),
            "unit": st.column_config.TextColumn(help="g, kg, ml, l, pc"),
        },
    )

    if st.button("💾 Enregistrer les modifications"):
        with sqlite3.connect(DB_FILE) as conn:
            yunit_id = rev_unit_map.get(yunit) if yunit else None
            conn.execute("UPDATE recipes SET name=?, type=?, yield_qty=?, yield_unit=? WHERE recipe_id=?",
                         (name.strip() or None, rtype.strip() or None, yqty if yqty > 0 else None, yunit_id, rid))

            current_ids = set([int(x) for x in edited["recipe_ingredient_id"].dropna().tolist()])
            existing_ids = set([r[0] for r in conn.execute("SELECT recipe_ingredient_id FROM recipe_ingredients WHERE recipe_id=?", (rid,)).fetchall()])
            to_delete = existing_ids - current_ids
            for _id in to_delete:
                conn.execute("DELETE FROM recipe_ingredients WHERE recipe_ingredient_id=?", (int(_id),))

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
                    conn.execute("""UPDATE recipe_ingredients
                                    SET ingredient_id=?, quantity=?, unit=?
                                    WHERE recipe_ingredient_id=?""", (iid, qty, uid, int(r["recipe_ingredient_id"])))
                else:
                    conn.execute("""INSERT INTO recipe_ingredients(recipe_id, ingredient_id, quantity, unit)
                                    VALUES (?,?,?,?)""", (rid, iid, qty, uid))
            conn.commit()
        st.success("Modifications enregistrées ✅")

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
        st.info("Aucun ingrédient lié à cette recette.")
        return

    # conversions
    def convert(qty, unit, ing_unit):
        if pd.isna(qty) or qty is None:
            return None
        unit = (unit or "").lower()
        ing_unit = (ing_unit or "").lower()
        if unit == ing_unit:
            return qty
        if unit == "kg" and ing_unit == "g":  return qty * 1000.0
        if unit == "g"  and ing_unit == "kg": return qty / 1000.0
        if unit == "l"  and ing_unit == "ml": return qty * 1000.0
        if unit == "ml" and ing_unit == "l":  return qty / 1000.0
        return None  # pas de conversion automatique avec 'pc'

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
    if pd.notna(total):
        st.metric("Coût total (lot)", f"{total:.2f} $")
    else:
        st.metric("Coût total (lot)", "—")
    st.caption("Conversions : kg↔g et L↔mL. Aucune conversion automatique avec 'pc'.")
def show_view_recipes():
    st.header("📖 Consulter les recettes")

    # — Recherche & filtres —
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        q = st.text_input("Recherche (nom contient…)", "")
    with sqlite3.connect(DB_FILE) as conn:
        types_df = pd.read_sql_query("SELECT DISTINCT COALESCE(type, '') AS type FROM recipes ORDER BY type", conn)
    with col2:
        type_filter = st.multiselect("Type", [t for t in types_df["type"].tolist() if t])
    with col3:
        sort_by = st.selectbox("Trier par", ["Nom", "Type", "Rendement"])

    # — Liste des recettes (selon filtres) —
    with sqlite3.connect(DB_FILE) as conn:
        base = """
            SELECT r.recipe_id, r.name, COALESCE(r.type,'') AS type,
                   r.yield_qty, u.abbreviation AS yield_unit
            FROM recipes r
            LEFT JOIN units u ON u.unit_id = r.yield_unit
        """
        conds, params = [], []
        if q:
            conds.append("LOWER(r.name) LIKE ?")
            params.append(f"%{q.lower()}%")
        if type_filter:
            conds.append("COALESCE(r.type,'') IN (%s)" % ",".join("?"*len(type_filter)))
            params.extend(type_filter)
        if conds:
            base += " WHERE " + " AND ".join(conds)

        if sort_by == "Nom":
            base += " ORDER BY r.name"
        elif sort_by == "Type":
            base += " ORDER BY r.type, r.name"
        else:  # Rendement
            base += " ORDER BY r.yield_qty DESC NULLS LAST, r.name"

        recipes = pd.read_sql_query(base, conn, params=params)

    if recipes.empty:
        st.info("Aucune recette trouvée avec ces critères.")
        return

    # — Sélection d’une recette —
    choice = st.selectbox("Sélectionne une recette :", recipes["name"])
    rid = recipes.loc[recipes["name"] == choice, "recipe_id"].iloc[0]

    # — Métadonnées recette —
    rrow = recipes[recipes["recipe_id"] == rid].iloc[0]
    st.subheader("Informations")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nom", rrow["name"])
    c2.metric("Type", rrow["type"] or "—")
    c3.metric("Rendement", f"{rrow['yield_qty']:.3f} {rrow['yield_unit']}" if pd.notna(rrow["yield_qty"]) and rrow["yield_unit"] else "—")

    # — Détail ingrédients + coûts —
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("""
            SELECT i.name AS ingredient,
                   ri.quantity AS qty,
                   u.abbreviation AS unit,
                   i.cost_per_unit,
                   iu.abbreviation AS ing_unit
            FROM recipe_ingredients ri
            JOIN ingredients i ON i.ingredient_id = ri.ingredient_id
            LEFT JOIN units u  ON u.unit_id  = ri.unit
            LEFT JOIN units iu ON iu.unit_id = i.unit_default
            WHERE ri.recipe_id = ?
            ORDER BY i.name
        """, conn, params=(rid,))

    def convert(qty, unit, ing_unit):
        if pd.isna(qty) or qty is None:
            return None
        unit = (unit or "").lower()
        ing_unit = (ing_unit or "").lower()
        if unit == ing_unit: return qty
        if unit == "kg" and ing_unit == "g":  return qty * 1000.0
        if unit == "g"  and ing_unit == "kg": return qty / 1000.0
        if unit == "l"  and ing_unit == "ml": return qty * 1000.0
        if unit == "ml" and ing_unit == "l":  return qty / 1000.0
        return None

    if df.empty:
        st.info("Cette recette n’a pas encore d’ingrédients liés.")
    else:
        df["qty_in_ing_unit"] = df.apply(lambda r: convert(r["qty"], r["unit"], r["ing_unit"]), axis=1)

        def line_cost(row):
            q, c = row["qty_in_ing_unit"], row["cost_per_unit"]
            if pd.isna(q) or pd.isna(c): return None
            try: return float(q) * float(c)
            except: return None

        df["line_cost"] = df.apply(line_cost, axis=1)

        st.subheader("Ingrédients")
        st.dataframe(
            df[["ingredient","qty","unit","qty_in_ing_unit","ing_unit","cost_per_unit","line_cost"]],
            use_container_width=True
        )

        total = df["line_cost"].sum(skipna=True)
        st.metric("💰 Coût total (lot)", f"{total:.2f} $" if pd.notna(total) else "—")

        # — Export CSV —
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter les ingrédients de la recette (CSV)", data=csv, file_name=f"{rrow['name']}_ingredients.csv", mime="text/csv")

    st.caption("Conversions : kg↔g et L↔mL. Pas de conversion automatique avec 'pc'.")

# ================================
# Main
# ================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à :",
        [
            "Accueil",
            "Importer ingrédients",
            "Importer recettes",
            "Corriger recettes",
            "Consulter recettes",       # <-- ajouté
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
    elif page == "Corriger recettes":
        show_fix_recipes()
    elif page == "Consulter recettes":       # <-- ajouté
        show_view_recipes()
    elif page == "Liste des ingrédients":
        show_ingredients()
    elif page == "Liste des recettes":
        show_recipes()
    elif page == "Coût recette":
        show_recipe_costs()
