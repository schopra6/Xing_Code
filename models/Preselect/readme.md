# Preselect Workflow

This folder contains python files to preselect job items for test users and  create features by implementing techniques from RecSys Challenge 2016: job recommendations based - on preselection of offers and gradient boosting, A. Pacuk, P. Sankowski, K. Wegrzycki, A. Witkowski and P. Wygocki. https://dl.acm.org/doi/10.1145/2987538.2987544


### Train and Test data creation
`python process_xgb_data.py  --items_file '../dummy_items.csv' --files_read_path '../prepare_data/dummy_dataset' `

Output files :
test_data.csv each row contains feature of the test data

train_data.csv each row contains feature of the train data

results.csv it contains interactions

### XGB train and results

`python xgb_train.py`
