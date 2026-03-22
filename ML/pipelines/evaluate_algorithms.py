from sklearn.model_selection import cross_val_score
import pandas as pd


def get_best_model_r2_score(df: pd.DataFrame):
    results = df.to_dict(orient="records")
    r2_score = next(
        (
            item["CV Mean R2"]
            for item in results
            if item["Algorithm"] == "random_forest_model"
        )
    )
    return r2_score


def algorithm_results(
    models,
    Pipeline,
    preprocessor,
    X_train,
    y_train,
):

    cv_results = []

    print("Starting Cross-Validation (this may take a few minutes)...")

    for name, model in models.items():
        # Recreate the pipeline for each model
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Perform 5-Fold Cross-Validation on the training data
        # This trains and evaluates the model 5 different times on different slices
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

        # Calculate the average performance and the 'stability' (Std Dev)
        mean_r2 = scores.mean()
        std_r2 = scores.std()

        print(f"Done: {name:20} | Mean R2: {mean_r2:.4f} (+/- {std_r2:.4f})")

        cv_results.append(
            {"Algorithm": name, "CV Mean R2": mean_r2, "Consistency (Std Dev)": std_r2}
        )

    # 4. View the new, more reliable ranking
    cv_comparison_df = pd.DataFrame(cv_results).sort_values(
        by="CV Mean R2", ascending=False
    )
    print("\n=== Robust Leaderboard (Cross-Validated) ===")
    save_algorithm_results(cv_comparison_df)
    return cv_comparison_df


def save_algorithm_results(cv_comparison_df: pd.DataFrame):
    cv_comparison_df.to_csv(
        "../../data/processed/algoritm_comparison_data.csv", index=False
    )
