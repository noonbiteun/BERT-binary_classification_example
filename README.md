# BERT_classification_example

it is a tutorial that you can follow to run binary-classification or multi-classification with BERT

## 1 Preparation

Befor we start to build up the classification example, it is advisable for you to visit the [website](https://github.com/google-research/bert) that you can get more information about BERT directly. It enables you to get a general understanding of BERT

#### 1.1 download bert-master

First of all, you have to download bert-master. 

[download](https://github.com/google-research/bert/archive/master.zip)

#### 1.2 download pre-trained model

- BERT-Large
- BERT-Base

In this tutorial, we choose the BERT-Base.

[download](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)

## 2 binary-classification

#### 2.1 Dataset

In binary-classification example, we use the Yelp Reviews-Polarity dataset. 

There are 1,569,264 samples from the Yelp Dataset Challenge 2015. This subset has 280,000 training samples and 19,000 test samples in each polarity.

[download](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)

Most datasets you find will typically come in the csv format and the Yelp Reviews dataset is no exception. Let’s load it in with pandas and take a look

**code segment**

```python
import pandas as pd

train_df = pd.read_csv('./train.csv', header=None)
print(train_df.head())

test_df = pd.read_csv('./test.csv', header=None)
print(test_df.head())
```

**output**

|   | 0 | 1 |
| - | - | - |
| 0 |	1 |	Unfortunately, the frustration of being Dr. Go... |
| 1	| 2	| Been going to Dr. Goldberg for over 10 years. ... |
| 2	| 1	| I don't know what Dr. Goldberg was like before... |
| 3	| 1	| I'm writing this review to give you a heads up... |
| 4	| 2	| All the food is great here. But the best thing... |

|   | 0 | 1 |
| - | - | - |
| 0 |	2	| Contrary to other reviews, I have zero complai... |
| 1	| 1	| Last summer I had an appointment to get new ti... |
| 2	| 2	| Friendly staff, same starbucks fair you get an... |
| 3	| 1	| The food is good. Unfortunately the service is... |
| 4 | 2	| Even when we didn't have a car Filene's Baseme... |

As you can see, the dataset contains two `.csv` files, and the labels of them is using 1 and 2 instead of the typical 0 and 1. So it is better for us to change the labels as 0 and 1.

BERT, however, wants data to be in a tsv file with a specific format.

Let’s make things a little BERT-friendly.

**code segment**

```python
import pandas as pd

# in
train_df = pd.read_csv('./train.csv', header=None)
test_df = pd.read_csv('./test.csv', header=None)

# change labels
train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)

line_num = len(train_df)
train_num = int(line_num / 10 * 8)
dev_num = line_num - train_num

# to make data format more friendly
# 80% for train, 20% for dev
train_df_bert = pd.DataFrame({
    'id':range(train_num),
    'label':train_df[0][:train_num],
    'text': train_df[1][:train_num].replace(r'\n', ' ', regex=True)
})

dev_df_bert = pd.DataFrame({
    'id':range(dev_num),
    'label':train_df[0][train_num:],
    'text': train_df[1][train_num:].replace(r'\n', ' ', regex=True)
})

test_df_bert = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    # 'alpha':['a']*test_df.shape[0],
    'text': test_df[2].replace(r'\n', ' ', regex=True)
})


train_df_bert.to_csv('./train.tsv', sep='\t', index=False, header=False)
dev_df_bert.to_csv('./dev.tsv', sep='\t', index=False, header=False)
test_df_bert.to_csv('./test.tsv', sep='\t', index=False, header=False)
```

#### 2.2 Processor

