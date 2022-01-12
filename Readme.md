# ML4NLP WiSe 2021/22 Challenge

## Usage
```bash
# download the repository to your local machine
git clone https://github.com/smnmnkr/sentiment-challenge.git

# move into repository
cd sentiment-challenge

# install requirements
make install

# if you want to use fasttext based model, download and extract (unix):
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz

# move fasttext file to data (unix)
mv cc.en.300.bin.gz ./data/

# run given baseline
make baseline

# run example configuration (./config_examples)
make example_base # untrained vector word embedding
make example_fasttext # fasttext embedding (see hint above)
make example_bert # bert as embedding (distilbert-base-cased provided by huggingface)

# reproduce experiments (./experiments)
# add path to test file in corresponding config to train and eval given on test
make experiment_fasttext
make experiment_bert

```

## Config
```json5
{
  "data": {
    "train_path": "data/train.csv",
    "eval_path": "data/eval.csv",
    "test_path": null
  },
  // see: ./challenge/data/preprocessor.py
  "preprocess": [
    "hyperlinks",
    "mentions",
    "hashtags",
    "retweet",
    "repetitions",
    "emojis",
    "smileys",
    "spaces",
    "punctuation",
    "tokenize" // incompatible with bert
  ],
  "embedding": {
    "type": "STRING", // choose from ["base", "fasttext", "bert"]
    "config": {
      "model_name": "distilbert-base-cased", // required for bert
      "data_path": "data/cc.en.300.bin", // required for fasttext
      "dimension": 300, // only base, fasttext
      "dropout": 0.2
    }
  },
  // see: ./challenge/model.py
  "model": {
    "rnn": {
      "hid_size": 64,
      "depth": 2,
      "dropout": 0.2
    },
    "score": {
      "hid_size": 32,
      "dropout": 0.2
    }
  },
  // see: ./challenge/trainer.py
  "trainer": {
    "epochs": 20,
    "shuffle": true,
    "batch_size": 32,
    "report_rate": 1,
    "max_grad_norm": 1.0,
    "optimizer": {
      "lr": 2e-3,
      "weight_decay": 1e-5,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8
    },
    "scheduler": "linear",
    "stopper": {
      "delta": 1e-3,
      "patience": 5
    }
  },
  "seed": 13,
  "debug": false,
  "log_dir": "experiments/_tmp/"
}
```

## Results (on eval)

### FastText
```
Load best model based on evaluation loss.
@009: 	loss(train)=0.5511 	loss(eval)=0.5552 	f1(train)=0.7221 	f1(eval)=0.7242 	duration(epoch)=0:00:27.883236

[--- METRIC (data: test) ---]
AVG           	 tp:     4588	 fp:     1743 	 tn:     4588	 fn:     1743	 prec=0.7247	 rec=0.7247	 f1=0.7247	 acc=0.7247
Neutral       	 tp:     2003	 fp:      680 	 tn:     2585	 fn:     1063	 prec=0.7466	 rec=0.6533	 f1=0.6968	 acc=0.7247
Non-Neutral   	 tp:     2585	 fp:     1063 	 tn:     2003	 fn:      680	 prec=0.7086	 rec=0.7917	 f1=0.7479	 acc=0.7247
```

### Bert
```
Load best model based on evaluation loss.
@005: 	loss(train)=0.5866 	loss(eval)=0.5700 	f1(train)=0.7278 	f1(eval)=0.7204 	duration(epoch)=0:05:58.375898

[--- METRIC (data: test) ---]
AVG           	 tp:     4564	 fp:     1767 	 tn:     4564	 fn:     1767	 prec=0.7209	 rec=0.7209	 f1=0.7209	 acc=0.7209
Neutral       	 tp:     1870	 fp:      571 	 tn:     2694	 fn:     1196	 prec=0.7661	 rec=0.6099	 f1=0.6791	 acc=0.7209
Non-Neutral   	 tp:     2694	 fp:     1196 	 tn:     1870	 fn:      571	 prec=0.6925	 rec=0.8251	 f1=0.7530	 acc=0.7209
```

## Credits

* Lisandro A. Cesaratto: <https://github.com/lcesaratto>
* Simon Münker: <https://github.com/smnmnkr>