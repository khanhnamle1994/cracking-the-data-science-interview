## Question
This question was asked by: Facebook

Given a stream of numbers, select a random number from the stream, with O(1) space in selection.

<!-- ## Solution
We need to prove that every element is picked with 1/n probability where n is the number of items seen so far. For every new stream item x, we pick a random number from 0 to the count-1. If the picked number is count-1, we replace the previous result with x.

```
import random

# A function to randomly select a item from stream[0], stream[1], .. stream[i-1]
def selectRandom(x, count=1, y=0):
    # x is the new value
    # y is the old value, default 0

    # If this is the first element from stream, return it
    if (count == 1):
        res = x;
    else:

        # Generate a random number from 0 to count - 1
        rnd = random.randrange(count);

        # Replace the prev random number with new number with 1/count probability
        if (rnd == count - 1):
            res = x
        else:
            res = y
    return res

# Driver Code
stream = [1, 2, 3, 4];
n = len(stream);

# Use a different seed value for every run.
for i in range (n):
    if i == 0:
        x = stream[0]
    else:
        x = selectRandom(stream[i], i+1, x)
    print("Random number from first", (i + 1), "numbers is", x)
```

To simplify proof, let us first consider the last element, the last element replaces the previously stored result with 1/n probability. So probability of getting last element as result is 1/n.

Let us now talk about second last element. When second last element processed first time, the probability that it replaced the previous result is 1/(n-1). The probability that previous result stays when nth item is considered is (n-1)/n. So probability that the second last element is picked in last iteration is [1/(n-1)] * [(n-1)/n] which is 1/n.

Similarly, we can prove for third last element and others. -->
