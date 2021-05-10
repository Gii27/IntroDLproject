# IntroDLproject
This is for Final Project of Intro to Deep Learning System

## Installation
The code is based on python3 and pytorch 1.8, you can install dependencies with
```
pip install -r requirements.txt
```
## Data preparation
Please put training and testing data under directory as `data/train.txt` and `data/test.txt`.
Also put word and char embedding file under directory as `data/word_vec.txt` and `data/char_vec.txt`.
You can find our data on 
https://drive.google.com/drive/folders/191zGqp7Ad5vYhDhYNa1CwAtbg_pZkutI?usp=sharing

## Running
You can directly use command
```
python main.py 'word' 0 0
```
to run basic model directly. 'word' implies using word model, the first 0 corresponds to embedding dropout rate and the second 0 corresponds to LSTM dropout rate.
