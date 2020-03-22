## Problem

This problem was asked by Airbnb.

Say you are tasked with producing a model that can recommend similar listings to an Airbnb user when they are looking at any given listing. What kind of model would you use, what data is needed for that model, and how would you evaluate the model?

## Solution
We can use a model based on a word2vec approach whereby we produce embeddings for each listing. Similar to word embeddings, these embeddings are vectors generated from a much larger feature space. Assuming that we have a rich feature space about listings (location, price, host, etc.), the data required would be sequences of click sessions on listings from various users, in order to have a notion of co-occurrence as is used by word2vec. Similar to having a sliding window approach, there are context listings and a window around them of the listings that were previously clicked, and listings that were clicked after the context listings.

The goal is to learn various characteristics that underpin the various commonalities between all listings. Once the embeddings are produced, we can rank similarity of listings based on the cosine similarity between their embeddings. To evaluate the model, we can run an A/B test with and without the model, and analyze the effects on click-through rates (CTRs) on listings, as well as seeing how many of the recommended listings ended up being ones that the user ultimately booked.
