# Data Preparation

In this directory, code is provided to clean the XING data and introduce bias in the data by assigning gender attribute to users.

## Transform 
[YAML file](data.yml) It encodes the configuration for different features
````config1:
  fields:
    industry_id : [0,21]
    career_level : [4,5,6]
    experience_years_experience : [0,1,2,3]
  set_attr : 'gender'
  proportion : [0.2,0.8]
  choices : [0,1]
````
- Here fields industry_id,career_level,experience_years_experience are picked from the user metadata 
- Users having those config value in yaml file would be assigned attribute 'gender' in a proportion of 0.2:0.8 
- We assume that 0 and 1 corresponds to female and male respectively and choices are set accordingly.
- 20% of users will be assigned value 0 and 80% of the users will be assigned gender 1

*disclaimer* : read the section of processing of the data in the [Processing Doc](https://github.wdf.sap.corp/I587293/Fairness_Recsys/blob/master/Processing%20Doc.docx) to know about how the values in the yaml files are selected.
 


### To generate attribute assigned user file, train and test interactions split files

` python split_data.py --interactions_file '../interactions.csv' --users_file '../users.csv' --items_file '../items.csv' --files_save_path 'test_dataset' --data_config 'data.yml' ` 
