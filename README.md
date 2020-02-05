Here's how to set up the project.

### 1) Pull the code
```
git clone git@github.com:mzgubic/fashion.git
```

### 2) Set up the environment
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

### 3) Set up some local variables

3.1) Edit fashion/utils.py to add your local path to this directory, see example.

3.2) If you want to use Google Places API you need an account with them. You will
also get an api key, which you need to copy to data/api\_key.txt file.

