from os import path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = None


# Constructs and returns a LabelEncoder for speaker IDs
def get_encoder(file, base_dir):
    global encoder

    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(
            list(pd.read_csv(path.join(base_dir, file), sep="|", header=None)[0])
        )

    return encoder
