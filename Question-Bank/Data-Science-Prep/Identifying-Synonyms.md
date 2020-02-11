## Problem
Say you are given a very large corpus of words. How would you identify synonyms?

## Solution
We can first find word embeddings through the corpus of words. Word2Vec is a popular algorithm for doing so, and it produces vectors for words based on the context that the words appear in. The word embeddings that are generated are weights on the resulting vectors. The distance between these vectors can be used to measure similarity, for example via cosine similarity or some other similarity measure.

Once we have these word embeddings, we can then run an algorithm like K-means to identify clusters within our word embeddings or run K-nearest neighbors for a particular word we want to find synonyms for.
