# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:37:12 2021

@author: musta
"""
import random
import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_text as tf_text



def bosluksayisi(txt):
    sayac=0
    for i in txt:
        if i==' ':
            sayac+=1
    return sayac        
text = open(r'C:/Users/musta/.keras/datasets/shakespeare.txt', 'rb').read().decode(encoding='utf-8')
spl= text.split('\n')
sira=-1


inp=[]
targ=[]
for text in spl:
    
    y=bosluksayisi(text)
    if y!=0:
     x= random.randint(1,y)
     for i in range(len(text)):
         if(text[i]==' '):
             x-=1
         if x==0:
             inp.append(text[:i])
             targ.append(text[i:])
             break

BUFFER_SIZE = len(inp)
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE)

for example_input_batch, example_target_batch in dataset.take(1):
  print(example_input_batch)
  print()
  print(example_target_batch)
  break

def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)
  
  
def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

max_vocab_size = 5000

input_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

output_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

input_text_processor.adapt(inp)
output_text_processor.adapt(targ)

example_tokens = input_text_processor(example_input_batch)
print(example_tokens[:3, :10])




#print("\n\n\n"+tf_lower_and_split_punct(inp).numpy().decode())