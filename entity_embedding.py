import matplotlib.pylab as plt
import numpy as np
from numpy import savetxt

from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
from IPython.display import Image

from utils.data import load_credit_scoring_data

# load data
data_df = load_credit_scoring_data(
    "datasets/UK/input_UK.csv", "datasets/UK/descriptor_UK.csv"
)

# identify target and feature columns
target = ["censor"]
features = data_df.columns.difference(["censor"])

# train/val/test split
X_train, y_train = data_df.iloc[:21000][features], data_df.iloc[:21000][target]
X_val, y_val = data_df.iloc[21000:27000][features], data_df.iloc[21000:27000][target]
X_test = data_df.iloc[27000:][features]

# transform numerical variables
scalar = StandardScaler()
num_cols = [i for i in X_train.select_dtypes(include=["number"])]
for i in num_cols:
    scalar.fit(X_train[i].values.reshape(-1, 1))
    X_train[i] = scalar.transform(X_train[i].values.reshape(-1, 1))
    X_val[i] = scalar.transform(X_val[i].values.reshape(-1, 1))

# check missing


# categorical embedding cardinality
embed_cols = [i for i in X_train.select_dtypes(include=["category", "bool"])]

for i in embed_cols:
    print(i, data_df[i].nunique())


# categorical features to list
def preproc(X_train, X_val, X_test, data_df):
    input_list_train = []
    input_list_val = []
    input_list_test = []
    full_data_list = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i, cat in enumerate(raw_vals):
            val_map[cat] = i
        input_list_train.append(X_train[c].map(val_map).fillna(0).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
        full_data_list.append(data_df[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in num_cols]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    full_data_list.append(data_df[other_cols].values)

    return input_list_train, input_list_val, input_list_test, full_data_list


# rename categorical embeddings and print info
for categorical_var in embed_cols:
    cat_emb_name = categorical_var.replace(" ", "") + "_Embedding"

    no_of_unique_cat = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat) / 2), 50))

    print(
        "Categorical Variable:",
        categorical_var,
        "Unique Categories:",
        no_of_unique_cat,
        "Embedding Size:",
        embedding_size,
    )

# rename categorical inputs and print
for categorical_var in embed_cols:
    input_name = "Input_" + categorical_var.replace(" ", "")
    print(input_name)

input_models = []
output_embeddings = []

# build inputs and outputs for categorical
for categorical_var in embed_cols:
    cat_emb_name = categorical_var.replace(" ", "") + "_Embedding"

    no_of_unique_cat = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat) / 2), 50))

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
    X_train, X_val, X_test, data_df
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

# extract weights to data file
embedding_model = Model(inputs=input_models, outputs=output_embeddings)
predictions = embedding_model.predict(full_data_list)
embedded_data = np.concatenate(predictions, axis=1)
savetxt("embedded_UK.csv", embedded_data, delimiter=",")
