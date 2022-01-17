# ML4NLP WiSe 2021/22 Challenge

## Usage

```bash
# download the repository to your local machine
git clone https://github.com/smnmnkr/sentiment-challenge.git

# move into repository
cd sentiment-challenge

# run prediction
execute __main__.py file in folder lib
(select variables accordingly)
```

## Config

```json
{
  "data": {
    "train_path": "data/train.csv",
    "eval_path": "data/eval.csv",
    "test_path": null
  },
  "preprocess": [
    "hyperlinks",
    "mentions",
    "hashtags",
    "retweet",
    "repetitions",
    "emojis",
    "smileys",
    "spaces"
  ],
  "model": {
    "name": "microsoft/deberta-base"
  },
  "trainer": {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "output_dir": "results/",
    "logging_dir": "logs/"
  }
}
```

## Credits

- Lisandro A. Cesaratto: <https://github.com/lcesaratto>
- Simon Münker: <https://github.com/smnmnkr>
