from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CATEGORICAL_COLUMNS = [
    "gender",
    "branch",
    "city",
    "internship",
    "has_backlog",
]

NUMERIC_COLUMNS = [
    "attendance_percent",
    "communication_skill",
    "tech_events_attended",
    "salary_expectation",
    "cgpa",
    "coding_problems_solved",
]


def get_canonical_branch(branch_name):
    branch_name = str(branch_name).strip().lower()

    branch_mapping = {
        "cse": "cse",
        "computer science": "cse",
        "c.s.e": "cse",
        "it": "it",
        "i.t.": "it",
        "information technology": "it",
        "ece": "ece",
        "electronics": "ece",
        "eee": "eee",
        "electrical": "eee",
        "mech": "mech",
        "mechanical": "mech",
        "civil": "civil",
    }

    return branch_mapping.get(branch_name, branch_name)


def prepare_dataframe(df):
    df = df.copy()
    df = df.drop_duplicates()

    df["cgpa"] = (
        df["cgpa"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df["cgpa"] = pd.to_numeric(
        df["cgpa"],
        errors="coerce"
    )

    df["branch"] = df["branch"].map(
        get_canonical_branch
    )

    df.loc[
        df["attendance_percent"] > 100,
        "attendance_percent"
    ] = np.nan

    df.loc[
        df["cgpa"] > 10,
        "cgpa"
    ] = np.nan

    df.loc[
        df["salary_expectation"] > 5000000,
        "salary_expectation"
    ] = np.nan

    df.loc[
        df["coding_problems_solved"] > 1000,
        "coding_problems_solved"
    ] = np.nan

    for column_name in NUMERIC_COLUMNS:
        median_value = df[column_name].median()
        df[column_name] = df[column_name].fillna(
            median_value
        )

    return df


def encode_features(df):
    encoded_df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )

    return encoded_df


def train_model(csv_path):
    df = pd.read_csv(csv_path)
    df = prepare_dataframe(df)

    columns_to_drop = [
        "student_id",
        "name",
        "registration_date",
        "placed",
    ]

    features_df = df.drop(
        columns=columns_to_drop
    )

    encoded_df = encode_features(features_df)

    target_map = {
        "No": 0,
        "Yes": 1,
    }

    target = df["placed"].map(target_map)

    x_train, x_test, y_train, y_test = train_test_split(
        encoded_df,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        random_state=42,
    )

    model.fit(
        x_train_scaled,
        y_train
    )

    y_pred = model.predict(
        x_test_scaled
    )
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"Model Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClass Distribution:")
    print(target.value_counts())

    joblib.dump(
        model,
        "gradient_boosting_model.pkl"
    )

    joblib.dump(
        scaler,
        "scaler.pkl"
    )

    joblib.dump(
        list(encoded_df.columns),
        "model_columns.pkl"
    )

    print(
        "\nTraining complete."
    )

    print(
        "Model saved successfully."
    )

    return (
        model,
        scaler,
        list(encoded_df.columns),
        accuracy,
    )


def predict_student(
    student_data,
    model,
    scaler,
    feature_columns
):
    student_df = pd.DataFrame(
        [student_data]
    )

    student_df = prepare_dataframe(
        student_df
    )

    student_encoded = encode_features(
        student_df
    )

    student_encoded = student_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    student_scaled = scaler.transform(
        student_encoded
    )

    prediction = int(
        model.predict(student_scaled)[0]
    )

    probability = float(
        model.predict_proba(student_scaled)[0][1]
    )

    return prediction, probability


if __name__ == "__main__":
    project_folder = Path(
        __file__
    ).resolve().parent

    csv_path = (
        project_folder / "students_record.csv"
    )

    model, scaler, feature_columns, accuracy = (
        train_model(csv_path)
    )

    sample_student = {
        "gender": "Male",
        "branch": "cse",
        "city": "Mumbai",
        "cgpa": 8.5,
        "attendance_percent": 85.0,
        "technical_skills": 8,
        "projects_completed": 3,
        "coding_problems_solved": 250,
        "communication_skill": 7.5,
        "internship": "Yes",
        "has_backlog": "No",
        "tech_events_attended": 2.0,
        "salary_expectation": 600000.0,
    }

    prediction, probability = predict_student(
        sample_student,
        model,
        scaler,
        feature_columns,
    )

    print(
        f"\nPrediction: {prediction}"
    )

    print(
        f"Placement Probability: "
        f"{probability * 100:.2f}%"
    )