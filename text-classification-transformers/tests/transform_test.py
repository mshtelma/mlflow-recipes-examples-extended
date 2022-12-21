import pandas as pd
from steps.transform import transformer_fn


def test_tranform_fn_returns_object_with_correct_spec():
    # pylint: disable=assignment-from-none
    transformer = transformer_fn()
    # pylint: enable=assignment-from-none
    if transformer:
        df = pd.read_parquet("./text-classification-transformers/data/complaints_cicd.parquet")
        res_df = transformer.fit_transform(df)
        assert len(df) == len(res_df)
