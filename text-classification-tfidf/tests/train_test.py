import pandas as pd
from steps.transform import transformer_fn
from steps.train import estimator_fn


def test_tranform_fn_returns_object_with_correct_spec():
    # pylint: disable=assignment-from-none
    transformer = transformer_fn()
    estimator = estimator_fn()
    # pylint: enable=assignment-from-none
    if estimator:
        df = pd.read_parquet(
            "./text-classification-tfidf/data/complaints_cicd.parquet"
        )[:100]
        transformed_df = transformer.fit_transform(df)
        estimator.fit(transformed_df, df["category_id"])
        pred_df = estimator.predict(transformed_df)
        assert len(df) == len(pred_df)
