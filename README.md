# word2hue
Interactive project rethinking how we visualise word representations, combining art and tech and merging the boundaries between them. 

We have text, crafted by and meaningful to humans. In machine learning and natural language processing, these texts must be turned into something processable by the machine -> features, vectors, embeddings. (See Word2Vec for more info, and the inspiration for the title of this project.) These are no longer meaningful to humans. This project aims to take these vectors and turn them back into something interpretable, processable, by humans, but now in a new modality - visual art. This is Word2Hue. 

For instance, this is what Lewis Caroll's Alice in Wonderland looks like in word vectors: 

![image](https://github.com/user-attachments/assets/e027fbfd-669b-4cd5-a90b-7c029aa06768)

---
## How it works:
1. Text processing - optionally clean punctuation, get tokens
2. Generate word embeddings with a given model - the current default is spaCy's 'en_core_web_lg'
3. Dimensionality reduction - reduce each word embedding to 3 values which are used as RGB values, default method is PCA.
4. Generate image - plot the RGB values in a grid, add optional blur for effect. 

