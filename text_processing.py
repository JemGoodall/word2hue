import nltk
from nltk.tokenize import word_tokenize
import re
import os
import sys
# uncomment for first run to download
# nltk.download('gutenberg')
# nltk.download('punkt')
# nltk.download('punkt_tab')

NLTK_TEXT_CHOICES = nltk.corpus.gutenberg.fileids()

def exclude_punctuation(words:list) -> list:
    # tokens = nltk.word_tokenize(text)
    # # Regular expression to match punctuation
    cleaned_tokens = [re.sub(r'[^\w\s]', '', word) for word in words]
    return cleaned_tokens


def get_tokens(text_input, keep_punctuation=False, n_tokens=-1):
    ''' Takes text_input and returns list of tokens.
    Text_input could be either:
    - gutenberg text name
    - file path to .txt file
    - raw string text '''
    if text_input in NLTK_TEXT_CHOICES:
         raw_text = nltk.corpus.gutenberg.raw(text_input)
    elif os.path.exists(text_input):
        with open(text_input, 'r') as f:
            raw_text = f.read()
    else: raw_text = text_input
    
    tokens = re.findall(r"\w+[-']?\w*|[^\w\s]", raw_text)  # don't separate hyphenated or contracted words
        
    if not keep_punctuation:
        tokens = exclude_punctuation(tokens)

    return tokens[:n_tokens]