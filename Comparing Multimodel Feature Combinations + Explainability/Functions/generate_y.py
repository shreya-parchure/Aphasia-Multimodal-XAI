def generate_y(dataset):
    y_extracted = dataset[['Noun_Acc']]
    y_extracted = y_extracted.values.flatten()
    len(y_extracted)
    return y_extracted