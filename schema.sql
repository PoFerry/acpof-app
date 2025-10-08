
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS units (
  unit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT UNIQUE,
  abbreviation TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS ingredients (
  ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT UNIQUE NOT NULL,
  unit_default  INTEGER NULL REFERENCES units(unit_id),
  cost_per_unit REAL NULL
);

CREATE TABLE IF NOT EXISTS recipes (
  recipe_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT UNIQUE NOT NULL,
  type       TEXT,
  yield_qty  REAL,
  yield_unit INTEGER NULL REFERENCES units(unit_id)
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
  recipe_ing_id INTEGER PRIMARY KEY AUTOINCREMENT,
  recipe_id     INTEGER NOT NULL REFERENCES recipes(recipe_id) ON DELETE CASCADE,
  ingredient_id INTEGER NOT NULL REFERENCES ingredients(ingredient_id),
  qty           REAL,
  unit_id       INTEGER NULL REFERENCES units(unit_id)
);

CREATE INDEX IF NOT EXISTS idx_ri_recipe ON recipe_ingredients(recipe_id);
CREATE INDEX IF NOT EXISTS idx_ing_name ON ingredients(name);
