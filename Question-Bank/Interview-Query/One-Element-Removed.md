## Problem
There are two lists, list X and list Y. Both lists contain integers from -1000 to 1000 and are identical to each other except that one integer is removed in list Y that exists in list X.

Write a function that takes in both lists and returns the integer that was removed in 0(1) time and O(n) space without using the python set function.

<!-- ## Solution
This question is a definition of a trick question. It's not really a python or algorithms question but more of a brain teaser meant to give you a problem to be solved in a creative way.

The question is asking how you figure out the number that is missing from list Y, which is identical to list X, except that one number is missing. We could loop through one list, create a hashmap, and figure out which element doesn't exist but that wouldn't be done in O(1) time.

Before getting into the coding, think about it logically - how would you find the answer to this?

The quick and simple solution is to **sum up all the numbers in X and sum up all the numbers in Y and subtract the sum of X from the sum of Y**, and that gives you the number that's missing. Because the elements in the list are integers, it adds a different dimension to the problem in creativity rather than the typical approach of data structures and algorithms.

```
def return_missing_integer(list_x, list_y):
    return sum(list_x) - sum(list_y)
```

Always ask follow up questions when given constraints. The interviewer could be holding back assumptions that would not ever be known without asking for more clarification. Some example would be:
- Is the list sorted?
- Is one of the lists the set of all integers from -1000 to 1000?
- Are any built in functions allowed besides the set function? -->
