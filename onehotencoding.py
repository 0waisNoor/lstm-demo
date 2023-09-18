import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def oneHotEncodeColumn(df,col_name):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(df[[col_name]]).toarray())
    values=[]
    #merge hot encoded column
    for index in range(0,encoder_df.shape[0]):
        row = list(encoder_df.iloc[index])
        new_arr = [int(i) for i in row]
        value = ''.join(str(x) for x in new_arr)
        values.append(value)

    return values
