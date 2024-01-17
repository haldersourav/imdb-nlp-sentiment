# Transformer-based NLP classifier for IMDB reviews

A pretrained transformers model (bert-base-uncased) was fine-tuned to act as a classifier for sentiment analysis on the IMDB reviews dataset.

### Data
The IMDB dataset has 50K movie reviews for binary sentiment classification through natural language processing (NLP). This dataset can be downloaded at https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

For more dataset information, please go through the following link,
http://ai.stanford.edu/~amaas/data/sentiment/

### Neural network
BERT base model (uncased)
BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. It was pretrained on the raw texts only, with no humans labeling.

It was pretrained with two objectives:

1) Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. 
2) Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.

Paper link: https://arxiv.org/abs/1810.04805

### Important libraries
1) PyTorch (https://pytorch.org)
2) Transformers (https://huggingface.co/)
3) scikit-learn (https://scikit-learn.org/stable/)