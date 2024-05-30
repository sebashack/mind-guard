# Mind Guard


## Data Sources

All of the original datasets where uplaoded to our "Raw Zone" in S3:

- https://mindguard.s3.amazonaws.com/raw/cyberbullying_tweets.tsv
- https://mindguard.s3.amazonaws.com/raw/mental-health.tsv
- https://mindguard.s3.amazonaws.com/raw/sentiment_tweets3.tsv
- https://mindguard.s3.amazonaws.com/raw/student_depression_text.tsv
- https://mindguard.s3.amazonaws.com/raw/students_anxiety_and_depression_classify.tsv
- https://mindguard.s3.amazonaws.com/raw/Suicide_Ideation_Datase_Twitter_based.tsv

The original Kaggle links can be found at:

- https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets
- https://www.kaggle.com/code/sasakitetsuya/students-anxiety-and-depression-classify-model
- https://www.kaggle.com/datasets/nidhiy07/student-depression-text
- https://www.kaggle.com/datasets/subhranilpaul/suicidal-mental-health-review
- https://www.kaggle.com/code/thomasgamet/potential-suicide-post-detection-90-from-chatgpt
- https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
- https://huggingface.co/datasets/Samsung/samsum


The SAMSUM dataset can be found here:

- https://huggingface.co/datasets/Samsung/samsum


## Project Setup

This project has been developed on Ubuntu 22.04.4 LTS, and some subprojects such as `tiny-llama`,
`distil-bert` and `social-thread-analyzer` require an NVIDIA GPU with Ampere architecture and at least
10GB of RAM. In particular, this projects were developed with a NVIDIA GeForce RTX 3060 with 12GB of RAM.

All of the subprojects in this repo were developed in Python virtual environments so it is necessary
to run the `first-time-install.sh` script at the root of this repo in order to have all the necessary
dependencies on the system:

```
./first-time-install.sh
```

The following are the instructions to setup each on of the subprojects:

### lda

To setup this subproject:

```
cd lda
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the lda pipeline:

```
python src
```

To deactivate environment

```
deactivate
```

### svd-kmeans

To setup this subproject:

```
cd svd-kmeans
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the svd-kmeans pipeline:

```
python src
```

To deactivate environment

```
deactivate
```


### multinomial-nb

To setup this subproject:

```
cd multinomial-nb
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the multinomial-nb pipeline:

```
python src
```

To deactivate environment

```
deactivate
```

### logistic-regression

To setup this subproject:

```
cd logistic-regression
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the logistic-regression pipeline:

```
python src
```

To deactivate environment

```
deactivate
```

### distil-bert


To setup this subproject:

```
cd distil-bert
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the training pipeline:

```
python training -d /path/to/dataset
```

The dataset for training can be downloaded from:

```
https://mindguard.s3.amazonaws.com/filtered/filtered_distilbert/distilbert-fine-tuning-output-2024-05-26_19-35-43.tsv
```

To deactivate environment

```
deactivate
```

## tiny-llama

To setup this subproject:

```
cd tiny-llama
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the training pipeline:

```
python training
```

The dataset to train this model is already downloaded withing the pipleine with HugginFace api.


To deactivate environment

```
deactivate
```

## social-thread-analyzer

To setup this subproject:

```
cd social-thread-analyzer
make venv
```

Once the virtual environment has been created activate it with:

```
source venv/bin/activate
```

To run the inference pipeline

```
python src -u https://mindguard.s3.amazonaws.com/demo/tiktok/tiktok-1.tsv
```

The example social media thread is downloaded from S3. You can use the url above as an example.
You should see that a file `classification.json` is generated. You can inspect it to visualize the
results.


To deactivate environment

```
deactivate
```

Note that our fine-tunned tiny-llama model is downloaded from:
https://mindguard.s3.amazonaws.com/trusted/tiny-llama/002__tiny_llama_fine_tuned_peft_model_5_epochs_15-05-2024__18-41-12.tar.lz

Similarly, our fine-tunned distil-bert model is downloaded from:
https://mindguard.s3.amazonaws.com/trusted/distil-bert/fine_tuned_distil_bert_model__metrics_5_epochs__28-05-2024__18-38-10.tar.lz
