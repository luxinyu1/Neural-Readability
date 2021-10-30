# Neural-Readability

Implementation of representative neural (supervised and unsupervised) approaches to measuring readability. Currently, this repo only focuses on sentence-level readability.

Unfortunately, Newsela is not a publicly available dataset. Only place the raw Newsela file in ```./datasets/newsela``` will make this repo works.

## Usage

- Finetune a BERT model to do readability classification task:

	```shell
	python -m neural-readability.bert.finetune
	```

- Train a BiLSTM model to do readability classification task:

	```shell
	python -m neural-readablity.lstm.train
	```

Unless otherwise specified, the training / validation / testing logs should be found in ```./logs/```. More usage scripts can be found in ```./scripts/```.

## Acknowledgements

Some code in this repo is based on [GRANT](https://github.com/kinimod23/GRANT/). Thank for its wonderful works.
