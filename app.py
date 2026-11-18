from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hotel_bookings.csv"
TARGET_COLUMN = "is_canceled"
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "reservation_status",
    "reservation_status_date",
}

app = Flask(__name__)

bookings = pd.read_csv(DATA_PATH)
feature_columns = [
    column for column in bookings.columns if column not in EXCLUDED_COLUMNS
]
categorical_columns = [
    column
    for column in feature_columns
    if pd.api.types.is_object_dtype(bookings[column].dtype)
    or pd.api.types.is_string_dtype(bookings[column].dtype)
]
numeric_columns = [
    column for column in feature_columns if column not in categorical_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_columns,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear")),
    ]
)
model.fit(bookings[feature_columns], bookings[TARGET_COLUMN])


def build_form_fields():
    fields = []
    for column in feature_columns:
        if column in categorical_columns:
            options = sorted(
                bookings[column].dropna().unique().tolist(), key=str
            )
            fields.append(
                {
                    "name": column,
                    "label": column.replace("_", " ").title(),
                    "type": "select",
                    "options": options,
                    "value": options[0],
                }
            )
        else:
            fields.append(
                {
                    "name": column,
                    "label": column.replace("_", " ").title(),
                    "type": "number",
                    "step": "any" if bookings[column].dtype.kind == "f" else "1",
                    "value": bookings[column].median(),
                }
            )
    return fields


FORM_FIELDS = build_form_fields()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = {}
            for field in FORM_FIELDS:
                raw_value = request.form.get(field["name"], "").strip()
                if not raw_value:
                    raise ValueError(f'{field["label"]} is required.')
                if field["type"] == "number":
                    values[field["name"]] = float(raw_value)
                else:
                    values[field["name"]] = raw_value

            input_row = pd.DataFrame([values], columns=feature_columns)
            result = int(model.predict(input_row)[0])
            prediction = {
                "value": result,
                "label": "Canceled" if result == 1 else "Not canceled",
            }
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)