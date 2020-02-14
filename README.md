# 1) Here's how to set up the project.

### 1.1) Pull the code
```
git clone git@github.com:mzgubic/fashion.git
```

### 1.2) Set up the environment
For the first time:
```
python3 -m venv fashion
cd fashion/
source bin/activate
pip install -r requirements.txt
```

And then everytime you start:
```
source bin\activate
```

### 1.3) Set up some local variables

Edit fashion/utils.py to add your local path to this directory, see example.

If you want to use Google Places API you need an account with them. You will
also get an api key, which you need to copy to data/api\_key.txt file.

# 2) Running the code

The instructions for preparing a basic dataset and training the baseline
model.

### 2.1 Prepare the dataset

Download unzip the files from the Egnyte folder, and place them into data/.

Split the 2018-2019 into separate datasets by running
```
cd scripts/
python split_sales.py
```

Prepare the baseline dataset: Each line represents the aggregated sales of an
item in a particular store during a particular week.
```
cd scripts/
python prepare_dataset.py --year 17
python prepare_dataset.py --year 18
python prepare_dataset.py --year 19
```



