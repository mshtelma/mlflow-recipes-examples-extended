"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer

# distilbert-base-uncased-finetuned-sst-2-english
model = SentenceTransformer("distilbert-base-uncased-finetuned-sst-2-english")

def encode_sentences(X):
    pool = model.start_multi_process_pool()
    emb = model.encode_multi_process(X.values, pool)
    model.stop_multi_process_pool(pool)
    return emb


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """

    return Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        (
                            "Consumer_complaint_narrative_transformer",
                            Pipeline([("fn", FunctionTransformer(encode_sentences))]),
                            "Consumer_complaint_narrative",
                        )
                    ]
                ),
            )
        ],
        verbose=True,
    )
