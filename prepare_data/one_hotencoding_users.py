import pandas as pd

readfile = '../assigned_users.csv'
writefile ='data/users_indicators.csv'
delimeter = '\t'

users_df = pd.read_csv(readfile)
features = list(users_df)[3:]
indicators = [users_df[['id']]]

print("Generating 1-hot coded variables ...")
for feature in features:
    print("1-hot encoding " + feature)
    if feature != 'edu_fieldofstudies':
        temp = pd.get_dummies(users_df[feature])
    else:
        temp = users_df[[feature]]
        temp = temp[feature].str.get_dummies(',')
    temp.columns = [feature + '_' + str((column)) for column in temp.columns]
    indicators.append(temp)

users_indicators = pd.concat(indicators, axis=1)

print("Saving ...")
users_indicators.to_csv(writefile, sep=delimeter)