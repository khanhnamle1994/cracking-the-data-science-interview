## Question
This question was asked by: Microsoft

Given an array of words and a maxWidth parameter, format the text such that each line has exactly maxWidth characters. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

Example:

```
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16

Output =
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
```

<!-- ## Solution
Since extra spaces between words should be distributed as evenly as possible, we need to implement round robin logic. Round robin logic can be implemented by iterating over each value in the array, checking if it is over the max width, and then adding spaces to the existing line if it has reached capacity.

The following line implements the round robin logic:

```
for i in range(maxWidth - num_of_letters):
    cur[i%(len(cur)-1 or 1)] += ' '
```

Once you determine that there are only k words that can fit on a given line, you know what the total length of those words is num_of_letters. Then the rest are spaces, and there are (maxWidth - num_of_letters) of spaces.

The "or 1" part is for dealing with the edge case len(cur) == 1.

```
def fullJustify(self, words, maxWidth):
    res = []
    cur= []
    num_of_letters = 0

    for w in words:
        #check if existing words + new words are greater than max width
        if num_of_letters + len(w) + len(cur) > maxWidth:
            #implement round robin logic
            for i in range(maxWidth - num_of_letters):
                cur[i%(len(cur)-1 or 1)] += ' '
            res.append(''.join(cur))
            cur, num_of_letters = [], 0
        cur += [w]
        num_of_letters += len(w)
    return res + [' '.join(cur).ljust(maxWidth)]
``` -->
