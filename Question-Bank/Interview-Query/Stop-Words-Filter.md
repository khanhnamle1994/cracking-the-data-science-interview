## Problem
This question was asked by: Instacart

Given a list of stop words, write a function that takes a string and returns a string stripped of the stop words.

Example:

```
stopwords = [
    'I',
    'as',
    'to',
    'you',
    'your',
    'but',
    'be',
    'a',
]

paragraph = 'I want to figure out how I can be a better data scientist'

Output:

stripped_paragraph = 'want figure out how can better data scientist'
```

<!-- ## Solution
Stripping stop words is pretty key in data science. Many times when creating word vectors or TFIDF vectors out of text, we have to make sure to isolate important words to understand the placement of non-stop words next to each other as bi-grams or trigrams.

The solution is relatively straightforward and we could even use a lambda function to filter out the input string if we wanted to. However let's try to do it from scratch in order to understand all of the moving parts.

First off we need to take our list of stop words and **convert it to a set**. A set is a built in function in Python that stores each word without order and holds unique keys. This is important for us given that we need to loop through each word in the input string to determine if the word exists in our long list of stop words.

Next we'll split the input string into a list and then individually check if each word exists in our set. If it doesn't, append it to the new list we created. If it does exist, then don't do anything.

By creating a new list we don't have to modify the existing input list and do not have to worry about inserts and deletes. All we have to do is do inserts.

Last, re-join the new list into a string with a space delimiter and return the value!

```
def stopwords_stripped(paragraph, stopwords):
    stop_set = set(stopwords) #create set
    new_string = [] #create new list

    #split paragraph into a list of words
    words = str(paragraph).lower().split()
    for word in words:
        #append only if the word is not a stop word
        if word not in stopwords:
            new_string.append(word)
    return ' '.join(new_string)
``` -->
