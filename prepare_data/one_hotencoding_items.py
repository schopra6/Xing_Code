import pandas as pd

readfile = '../items.csv'
writefile ='data/items_indicators.csv'
delimeter = '\t'
items_df = pd.read_csv(readfile, delimiter=delimeter)
# omitting serial number and item id
features = list(items_df)[2:]
indicators = [items_df[['id']]]
to_exclude = {'tags', 'created_at', 'latitude', 'longitude'}
lat_long = {'latitude', 'longitude'}

print("Generating 1-hot coded variables ...")
for feature in features:
    if feature not in to_exclude:
        temp = pd.get_dummies(items_df[feature])
        temp.columns = [feature + '_' + str((column)) for column in temp.columns]
    elif feature in lat_long:
        temp = items_df[feature]
    else:
        continue
    print("1-hot encoding " + feature)
    indicators.append(temp)

items_indicators = pd.concat(indicators, axis=1)

print("Saving ...")
items_indicators.to_csv(writefile, sep=delimeter, mode="w")