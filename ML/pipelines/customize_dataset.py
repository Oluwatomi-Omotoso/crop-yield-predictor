def dataframe_to_arrays(dataframe, input_cols, output_cols, categorical_cols):
    df = dataframe.copy(deep=True)

    # categorical data handling
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    inputs_array = df[input_cols].to_numpy()
    targets_array = df[output_cols].to_numpy()

    return inputs_array, targets_array


def customize_dataset(dataframe_raw, random_string):
    df = dataframe_raw.copy(deep=True)
    random_string = random_string

    # drop some columns
    df = df.sample(int(0.95 * len(df)), random_state=int(ord(random_string[0])))

    # scale inputs
    df.year = df.year * ord(random_string[1]) / 100

    # scale target
    df.crop_yields = df.crop_yields * ord(random_string[2]) / 100

    # drop columns
    if ord(random_string[3]) % 2 == 1:
        df = df.drop(["area"], axis=1)

    return df
