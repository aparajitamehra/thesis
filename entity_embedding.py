from keras.layers import (
    Input,
    Dense,
    Activation,
    Reshape,
    Concatenate,
    Dropout,
)
from keras.layers.embeddings import Embedding
from keras.models import Model
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EntityEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def fit(self, X, y):
        if self.embedding_model:
            return self
        else:
            self._fit(X, y)
            return self

    def _fit(self, X, y):
        input_models = []
        output_embeddings = []

        for category in X.T:
            input_model, output_embedding = self._create_category_embedding(category)
            input_models.append(input_model)
            output_embeddings.append(output_embedding)

        trainer_output = Concatenate()(output_embeddings)
        trainer_output = Dense(1000, kernel_initializer="uniform")(trainer_output)
        trainer_output = Activation("relu")(trainer_output)
        trainer_output = Dropout(0.4)(trainer_output)
        trainer_output = Dense(512, kernel_initializer="uniform")(trainer_output)
        trainer_output = Activation("relu")(trainer_output)
        trainer_output = Dropout(0.3)(trainer_output)
        trainer_output = Dense(1, activation="sigmoid")(trainer_output)

        embedding_trainer = Model(inputs=input_models, outputs=trainer_output)
        embedding_trainer.compile(
            loss="mean_squared_error", optimizer="Adam", metrics=["mse", "mape"]
        )

        embedding_trainer.fit(X.T.tolist(), y, epochs=2, batch_size=512)

        self.embedding_model = Model(inputs=input_models, outputs=output_embeddings)
        self.embedding_model.compile(
            loss="mean_squared_error", optimizer="Adam", metrics=["mse", "mape"]
        )

        return self

    def transform(self, X, y=None):
        predictions = self.embedding_model.predict(X.T.tolist())
        return np.concatenate(predictions, axis=1)

    @staticmethod
    def _create_category_embedding(category_values):
        num_unique_cat = len(np.unique(category_values))
        embedding_size = int(min(np.ceil(num_unique_cat / 2), 50))

        input_model = Input(shape=(1,))
        output_embedding = Embedding(num_unique_cat, embedding_size,)(input_model)
        output_embedding = Reshape(target_shape=(embedding_size,))(output_embedding)

        return input_model, output_embedding


if __name__ == "__main__":
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from utils.data import load_credit_scoring_data

    for ds_name in ["german", "UK", "bene1", "bene2"]:
        data = load_credit_scoring_data(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
        )

        y = data.pop("censor")
        X = data

        categorical_features = X.select_dtypes("category").columns
        categorical_pipeline = Pipeline(
            steps=[("encoder", OrdinalEncoder()), ("embedder", EntityEmbedder())]
        )

        ct = ColumnTransformer(
            transformers=[("categorical", categorical_pipeline, categorical_features)]
        )

        ct.fit(X, y)

        ct.named_transformers_["categorical"].named_steps[
            "embedder"
        ].embedding_model.save(f"datasets/{ds_name}/embedding_model_{ds_name}.h5")
