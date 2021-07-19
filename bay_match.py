#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:29:29 2021

@author: chengren
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import plotly.express as px
#import spacy
import nltk
nltk.download('punkt')
import gensim
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#import flair
import matplotlib.dates as mdates
from datetime import datetime
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2021)
import nltk
nltk.download('wordnet')
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models import Word2Vec
#import en_core_web_lg
from rapidfuzz import process, fuzz




#def lemmatize_stemming(text):
#    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(token)
    return result
stemmer = SnowballStemmer('english')


df_val= pd.read_csv('https://raw.githubusercontent.com/chengren/NPOs/master/hold_out.csv',dtype={'ein': object})
df_val['clean_org'] = df_val['Org Name'].map(preprocess)
df_val['clean_org_2'] = df_val.apply(lambda row: TreebankWordDetokenizer().detokenize(row['clean_org']), axis=1)

bmf_2020 = pd.read_csv('bmf.bm2004.csv',dtype={'EIN': object,'FIPS':object,'ZIP5':object})
bmf_2019 = pd.read_csv('bmf.bm1908.csv',dtype={'EIN': object,'FIPS':object,'ZIP5':object})
bmf_2018 = pd.read_csv('bmf.bm1812.csv',dtype={'EIN': object,'FIPS':object,'ZIP5':object})
bmf_2003 = pd.read_csv('bmf.bm0311.csv',dtype={'EIN': object,'FIPS':object,'ZIP5':object})
bmf_2007 = pd.read_csv('bmf.bm0709.csv',dtype={'EIN': object,'FIPS':object,'ZIP5':object})

col = ['EIN','NAME', 'CITY', 'STATE','ZIP5','FIPS','NTEECC', 'NTEE1']
bmf_2020 = bmf_2020[col]
bmf_2019 = bmf_2019[col]
bmf_2018 = bmf_2018[col]
bmf_2003 = bmf_2003[col]
bmf_2007 = bmf_2007[col]

df = pd.concat([bmf_2020, bmf_2019,bmf_2018,bmf_2007, bmf_2003], ignore_index=True)
df = df.dropna(subset=['NTEE1'])
df = df.drop_duplicates(subset=['EIN'])
df_val = df_val.merge(df, left_on='ein',right_on='EIN',how='left')

df_val.isna().sum()


## call model 
country = pd.read_csv('https://raw.githubusercontent.com/Dinuks/country-nationality-list/master/countries.csv')
country['low_cot'] = country['en_short_name'].str.lower()
country['low_nat'] = country['nationality'].str.lower()
country_list = country['low_cot'].to_list()+country['low_nat'].to_list()+['asian','asia','europe','european','africa','african','hispanic',
                                                                          'latin','immigrant','immigration','migrant','refugee']

df_val['clean_org_2'].str.contains('|'.join(country_list)).sum()

ntee_code = ['P84','R21','Q71','A23','R22']
df_val['NTEECC'].str.contains('|'.join(ntee_code)).sum()



##############
##############
######NY######
df_ny = df[df['STATE'].isin(['NY'])]
df_ny['name'] = df_ny['NAME'].str.lower()
df_ny = df_ny.drop_duplicates(subset=['name'])
df_ny.reset_index(inplace=True, drop=True)

nyc = pd.read_csv('nyc.csv',dtype={'Name': object})
nyc = nyc[['Name','Place']]
nyc['Name'] = nyc['Name'].str.replace(r'[^\x00-\x7F]+','')
nyc['Name']=nyc['Name'].str.strip()
nyc = nyc.drop_duplicates(subset=['Name'])
nyc.reset_index(inplace=True, drop=True)


names_array_f=[]
ratio_array_f=[]
names_array_s=[]
ratio_array_s=[]
for word in nyc['Name']:
    result = process.extract(word, df_ny['name'],scorer=fuzz.token_sort_ratio, limit=2)#fuzz.token_set_ratio
    first_res = result[0]
    second_res = result[1]
    names_array_f.append(first_res[0])
    ratio_array_f.append(first_res[1])
    names_array_s.append(second_res[0])
    ratio_array_s.append(second_res[1])

combine = pd.DataFrame({'name':nyc['Name'],'first_name':names_array_f,'first_ratio':ratio_array_f,
                       'second_name':names_array_s,'second_ratio':ratio_array_s})

combine= combine.merge(df_ny,left_on='first_name',right_on='name',how='left')
combine.to_csv('nyc_ein.csv',index=False)

#####check after manual
ny_ma = pd.read_csv('nyc_ein_manual.csv')
ny_ma['val']=ny_ma['name_mau'].str.lower()

## call model 


ny_ma['val'].str.contains('|'.join(country_list)).sum()

ntee_code = ['P84','R21','Q71','A23','R22']
ny_ma['ntee_mau'].str.contains('|'.join(ntee_code)).sum()