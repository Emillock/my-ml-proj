import pandas as pd
from category_encoders import CatBoostEncoder
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

categories_col = ["Brand", "Style", "Country"]
embedding_col = "Variety"


class AddEmbed(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_col: str):
        self.embedding_col = embedding_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=self._input_cols)

        X = X.copy()

        # generate embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = X[self.embedding_col].astype(str).tolist()
        embs = model.encode(texts)  # shape (n_samples, 384)

        # turn into DataFrame
        emb_df = pd.DataFrame(
            embs,
            index=X.index,
            columns=[f"{self.embedding_col}_emb_{i}" for i in range(embs.shape[1])],
        )

        # drop original text column, concat embeddings
        X2 = X.drop(columns=[self.embedding_col])
        return pd.concat([X2, emb_df], axis=1)


def main():
    df = pd.read_csv("./data/external/ramen-ratings.csv")

    target = "Stars"

    X = df.drop(columns=[target, "Top Ten"])
    y = df[target]

    preprocessor = ColumnTransformer(
        [
            # ('text', Pipeline(),embedding_col),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encode", CatBoostEncoder()),
                    ]
                ),
                categories_col,
            ),
            ("text", "passthrough", [embedding_col]),
        ]
    )

    to_df = FunctionTransformer(
        func=lambda X: pd.DataFrame(X, columns=categories_col + [embedding_col]),
        validate=False,  # allow mixed dtypes / object arrays
    )

    preproc_pipeline = Pipeline(
        [
            ("preproc", preprocessor),
            ("to_df", to_df),
            ("embedding", AddEmbed(embedding_col=embedding_col)),
        ]
    )

    X_transformed = preproc_pipeline.fit_transform(X, y)

    df_merged = X_transformed.join(y)
    df_merged.to_parquet("./data/processed/ramen-ratings.parquet")


if __name__ == "__main__":
    main()
