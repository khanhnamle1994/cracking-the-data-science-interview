## Problem
This question was asked by: Indeed

Write a function that can take a string and return a list of bigrams.

Example:
```
sentence = "Have free hours and love children? Drive kids to school, soccer practice and other activities."

output = [('have', 'free'),
 ('free', 'hours'),
 ('hours', 'and'),
 ('and', 'love'),
 ('love', 'children?'),
 ('children?', 'drive'),
 ('drive', 'kids'),
 ('kids', 'to'),
 ('to', 'school,'),
 ('school,', 'soccer'),
 ('soccer', 'practice'),
 ('practice', 'and'),
 ('and', 'other'),
 ('other', 'activities.')]
```

<!-- ## Solution:
There are a number of ways to solve this. At it's core, bi-grams are two words that are placed next to each other. Bi-grams give additional complexity in feature extraction versus just bag-of-words where each word frequency is set as it's own feature. Two words versus one in feature engineering for a NLP model gives an interaction effect.

To actually parse them out of a string, we need to first split the input string. Then, once we've identified each individual word, we need to loop through each and append the subsequent word to make a tuple. This tuple gets added to a list that we eventually return.

```
def find_bigrams(sentence):
  input_list = sentence.split()
  bigram_list = []

  # Now we have to loop through each word
  for i in range(len(input_list)-1):
    #strip the whitespace and lower the word to ensure consistency
    bigram_list.append((input_list[i].strip().lower(), input_list[i+1].strip().lower()))
  return bigram_list
``` -->
