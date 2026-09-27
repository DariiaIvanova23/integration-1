"""Train the model and persist the resulting artifact."""

from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .model import (
    CAT,
    NUM,
    TARGET,
    load_training_data,
    train_model,
)


ARTIFACTS = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
)

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    """Train the model, report metrics, and persist the artifact."""

    df = load_training_data()

    # Held-out evaluation 
    # Split off a test set the model never sees during fitting, so
    # R2/MAE reflect generalization rather than memorization of the
    # training data (unlike the original in-sample evaluation).
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    eval_model = train_model(train_df)

    train_predictions = eval_model.predict(train_df[NUM + CAT])
    test_predictions = eval_model.predict(test_df[NUM + CAT])

    train_r2 = r2_score(train_df[TARGET], train_predictions)
    train_mae = mean_absolute_error(train_df[TARGET], train_predictions)

    test_r2 = r2_score(test_df[TARGET], test_predictions)
    test_mae = mean_absolute_error(test_df[TARGET], test_predictions)

    print(
        f"[train] Ames train (in-sample, n={len(train_df)}): "
        f"R2={train_r2:.4f}, MAE={train_mae:.0f} USD"
    )
    print(
        f"[train] Ames test (held-out, n={len(test_df)}): "
        f"R2={test_r2:.4f}, MAE={test_mae:.0f} USD"
    )
    print(
        f"[train] Generalization gap: "
        f"dR2={train_r2 - test_r2:.4f}, "
        f"dMAE={test_mae - train_mae:.0f} USD"
    )

    # --- Final production model ---
    # After confirming the approach generalizes reasonably on the
    # held-out split, refit on the full dataset so the served model
    # benefits from all available data.
    model = train_model(df)

    full_predictions = model.predict(df[NUM + CAT])

    full_r2 = r2_score(df[TARGET], full_predictions)
    full_mae = mean_absolute_error(df[TARGET], full_predictions)

    print(
        f"[train] Ames full-data in-sample (original metric, "
        f"production model): R2={full_r2:.4f}, MAE={full_mae:.0f} USD"
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = ARTIFACTS / "model.pkl"

    joblib.dump(
        model,
        artifact_path,
    )

    print(
        f"[train] artifact persisted: {artifact_path}"
    )


if __name__ == "__main__":
    main()