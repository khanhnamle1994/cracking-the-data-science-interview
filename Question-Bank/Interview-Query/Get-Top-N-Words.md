## Problem

This question was asked by: Indeed

1. Given an example paragraph string and an integer N, write a function that returns the top N frequent words in the posting and the frequencies for each word.

2. What's the function run-time?

**Example:**

```
n = 3
posting = """
Herbal sauna uses the healing properties of herbs in combination with distilled water.
The water evaporates and distributes the effect of the herbs throughout the room.
A visit to the herbal sauna can cause real miracles, especially for colds.
"""

output = [
    ('the', 6),
    ('herbal', 2),
    ('sauna', 2),
]
```

<!-- ## Solution

1. Given we have a string that we need to split into word and then return some ranking of the top frequencies, we know implicitly that we'll have to store each individual word and the count in order to track word frequency rank.

This then sounds like a key value pair problem which means we should use a dictionary as a hashmap to track counts associated with each unique word. If we loop through each word in the paragrah, save each word as a key, and then increment the count for each word as it shows up, we end up with a dictionary with words as the keys and the frequencies as the values.

```
for word in words:
    # if the word exists,
    if word in hashmap.keys():
        # add to the count
        hashmap[word] += 1
    # if it doesn't exist, create the key value pair
    else:
        hashmap[word] = 1
```

Once all values are in the dictionary, we can sort it with a lambda function and return the top n values.

```
def topN(posting, n):
    words = posting.lower().split()
    hashmap = {}
    # iterate through all of the words in the paragraph
    for word in words:
        if word in hashmap.keys():
            hashmap[word] += 1
        else:
            hashmap[word] = 1
    # sort hashmap for top n values
    values = sorted(hashmap.items(), key=lambda x: x[1], reverse=True)
    return values[:n]
```

2. The run-time will be based on two parts, the writing time and retrieval time. Going through each word and storing it in the hashmap can be done in O(n) time where n is equal to the length of the posting.

Retrieval however is the sort and return of the top N values. The fastest the sort can be done is in O(nlogn) time. So the run-time will be equivalent to O(nlogn). -->
