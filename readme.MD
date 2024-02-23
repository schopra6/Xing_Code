# About
 Given a XING user, the recommender should predict those job postings (items) that the user will interact with in the next week. 
 With the existing problem of recommending job items, we try to find if there exists social bias in the recommendation system.
 
 # Dataset 
 The dataset has been taken from the recsys2016 challenge http://2016.recsyschallenge.com/
 
 # Step to setup the project 
 
Prerequisite : clone this repository 
 
### 1. Create and activate a new conda environment
`conda create -n <environment_name> python=3.9`

`conda activate <environment_name>`

`conda install --yes --file requirements.txt`

**CPU only**
conda install -c conda-forge py-xgboost-cpu

**Use NVIDIA GPU**
conda install -c conda-forge py-xgboost-gpu

### 2. Extract the files
Place interactions.csv,items.csv,users.csv files from http://2016.recsyschallenge.com/ in Fairness_Recsys directory.  Since the data is unavailable , kindly ask the authors for it otherwise load it from /workspace/home/Recsysdata in DGX server.
```bash
|──Fairness_Recsys
    ├── train_interactions.csv
    ├── users.csv
    ├── items.csv
```

### 3. Prepare customised biased users data
`cd prepare_data`

### 4. Train and run Recommendation system
`cd PreselectRecsys`

### 5 Generate Biased analysis on the output of the recommendations
`cd results`

 
 
