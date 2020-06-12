
### Customized loading and processing BUCC data ###

# Recommended Usage: import bucc_proc as bp
# runs on import

try:
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import nltk
    import jieba
    import re

    nltk.download('punkt')
    nltk.download('stopwords')

    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    from string import punctuation
    from nltk.corpus import stopwords
    
except Exception as e:
    print(e)

def get_merge():
    merged_file = Path("zh-en.training.merge")
    if merged_file.is_file():
        
        print('Merged file exists, reading...')
        new_df = pd.read_csv(merged_file, header=0, sep='\t')
    
    else:
    
        print('Merged file does not exist, creating...')
    
        zh_file = "zh-en.training.zh"
        en_file = "zh-en.training.en"
        pair_file = "zh-en.training.gold"
    
        zh_df = pd.read_csv(zh_file, names=['ID_zh','Sentence_zh'], sep='\t')
        en_df = pd.read_csv(en_file, names=['ID_en','Sentence_en'], sep='\t')
        pair_df = pd.read_csv(pair_file, names=['ID_zh','ID_en'], sep='\t')
        new_df = pair_df.merge(zh_df, 'inner', 'ID_zh')
        new_df = new_df.merge(en_df, 'inner', 'ID_en')
        new_df.to_csv(merged_file, index=False, sep='\t')
        
    return new_df

en_stopwords=stopwords.words("english")+["'s"]  #chinese de is stopword
stemmer=PorterStemmer()
punctuation = punctuation +'–’“”'

def en_proc(sentence):
    ''' 1. Tokenize Sentence -> Words
        2. Remove punctuation and stopwords
        3. Stemming Words'''
    word_list = word_tokenize(sentence)
    bow_list = [stemmer.stem(w.lower()) for w in word_list if w.lower() not in en_stopwords and w not in punctuation]
    
    return bow_list

with open('../zh_stopwords.txt','r', encoding='utf-8') as file:
    zh_stopwords = file.read()
zh_stopwords = re.sub('[ A-Za-z]+\n', ',', zh_stopwords)
zh_stopwords = zh_stopwords.translate(str.maketrans('', '', '\n')).split(',') 
zh_stopwords = list(filter(None, zh_stopwords))
punctuation = punctuation + '，「」。！？《》【】、'


def zh_proc(sentence):
    ''' 1. Segmentation
        2. Remove punctuation and stopwords'''
    bow_list = [w for w in jieba.cut(sentence) if w not in zh_stopwords and w not in punctuation]
    return bow_list


def cosine_similarity(v1,v2):
    '''cosine_similarity(transformed_docs[2], transformed_docs[2])'''
    ## Idk why need to np.squeeze (1,148) into (148,) shape to dot product [error: shapes not aligned]
    ## toarray() [error: dimension mismatch] v1.toarray()v2.toarray()
    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)
    return np.dot(v1,v2) / ( np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)) )