## Problem
In data science, there exists the concept of stemming, which is the heuristic of chopping off the end of a word to clean and bucket it into an easier feature set.

Given a dictionary consisting of many roots and a sentence, stem all the words in the sentence with the root forming it. If a word has many roots can form it, replace it with the root with the shortest length.

**Example**:

```
Input:
roots = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"

Output: "the cat was rat by the bat"
```

<!-- ## Solution
At first it simply looks like we can just loop through each word and check if the root exists in the word and if so, replace the word with the root. But since we are technically stemming the words we have to make sure that the roots are equivalent to the word at it's prefix rather than existing anywhere within the word.

We're given a dictionary of roots with a sentence string. Given we have to check each word, let's try creating a function that takes a word and returns the existing word if it doesn't match a root, or return the root itself.

```
def replace(word, rootset):
    # loop through each subsequent letter
    for i in xrange(1, len(word)):
        # if the word at the letter is equal one word in the rootset
        # return the rootset word
        if word[:i] in rootset:
            return word[:i]
    return word
```

Here we're going through each character in the word starting from the beginning and looping through each letter until the resulting word is either equivalent or not to a root in the rootset. We can create the rootset by just making the list into a set.

```
def replaceWords(roots, sentence):
    rootset = set(roots) #create a set

    def replace(word):
        for i in xrange(1, len(word)):
            if word[:i] in rootset:
                return word[:i]
        return word

    return " ".join(map(replace, sentence.split()))
```

Given we've created the replace function, we can now just map it to splitting the sentence input and re-join the list back into a sentence. -->
