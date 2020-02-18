import numpy as np
from numpy import savetxt
import matplotlib.pylab as plt
from IPython.display import Image

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import plot_model

from utils.data import load_credit_scoring_data
from utils.preprocessing import HighVIFDropper


# categorical features to list
def preproc(X_train, X_val, X_test, data):
    input_list_train = []
    input_list_val = []
    input_list_test = []
    input_list_full = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i, cat in enumerate(raw_vals):
            val_map[cat] = i
        input_list_train.append(X_train[c].map(val_map).fillna(0).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
        input_list_full.append(data[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in num_cols]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    input_list_full.append(data[other_cols].values)

    return input_list_train, input_list_val, input_list_test, input_list_full


# load data
data = load_credit_scoring_data(
    "datasets/UK/input_UK.csv", "datasets/UK/descriptor_UK.csv"
)

# identify target and feature columns
target = ["censor"]
features = data.columns.difference(["censor"])

# train/val/test split
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2
)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# transform numerical variables
numeric_features = X_train.select_dtypes("number").columns
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("vif_dropper", HighVIFDropper(threshold=10)),
        ("scaler", StandardScaler()),
    ]
)
numeric_columns = ColumnTransformer(
    transformers=[("numerical", numeric_pipeline, numeric_features)]
)
num_cols = [i for i in X_train.select_dtypes(include=["number"])]
for i in num_cols:
    X_train[i].values.reshape(-1, 1)
    X_train[i] = X_train[i].values.reshape(-1, 1)
    X_val[i] = X_val[i].values.reshape(-1, 1)

# check missing

input_models = []
output_embeddings = []

# build inputs and outputs for categorical
embed_cols = [i for i in X_train.select_dtypes(include=["category", "bool"])]

for categorical_var in embed_cols:
    cat_emb_name = categorical_var.replace(" ", "") + "_Embedding"

    no_of_unique_cat = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil(no_of_unique_cat / 2), 50))

    input_model = Input(shape=(1,))
    output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(
        input_model
    )
    output_model = Reshape(target_shape=(embedding_size,))(output_model)

    input_models.append(input_model)
    output_embeddings.append(output_model)

# build input and output for numeric
input_numeric = Input(
    shape=(len(X_train.select_dtypes(include=["number"]).columns.tolist()),)
)
embedding_numeric = Dense(128)(input_numeric)
input_models.append(input_numeric)
output_embeddings.append(embedding_numeric)

# build model
output = Concatenate()(output_embeddings)
output = Dense(1000, kernel_initializer="uniform")(output)
output = Activation("relu")(output)
output = Dropout(0.4)(output)
output = Dense(512, kernel_initializer="uniform")(output)
output = Activation("relu")(output)
output = Dropout(0.3)(output)
output = Dense(1, activation="sigmoid")(output)

# compile model
model = Model(inputs=input_models, outputs=output)
model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["mse", "mape"])

# plot model structure
plot_model(model, show_shapes=True, show_layer_names=True, to_file="model.png")
Image(retina=True, filename="model.png")
model.summary()

# apply model to data and fit
X_train_list, X_val_list, X_test_list, full_data_list = preproc(
    X_train, X_val, X_test, data
)
history = model.fit(
    X_train_list,
    y_train,
    validation_data=(X_val_list, y_val),
    epochs=2,
    batch_size=512,
    verbose=2,
)

# plot loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper right")
plt.show()

embedding_model = Model(inputs=input_models, outputs=output_embeddings)

# extract predicted weights to data file
predictions = embedding_model.predict(full_data_list)
embedded_data = np.concatenate(predictions, axis=1)
savetxt("embedded_UK.csv", embedded_data, delimiter=",")
