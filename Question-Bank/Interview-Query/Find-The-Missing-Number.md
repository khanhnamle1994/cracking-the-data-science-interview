## Question
This question was asked by: Microsoft

You have an array of integers of length n spanning 0 to n with one missing. Write a function that returns the missing number in the array.

**Example**:
```
nums = [0,1,2,4,5]
missingNumber(nums) -> 3
```
Complexity of O(N) required.

<!-- ## Solution
There are two ways we can solve this problem. One way through logical iteration and another through mathematical formulation. We can look at both as they both hold O(N) complexity.

The first would be through general iteration through the array. We can pass in the array and create a set which will hold each value in the input array. Then we create a for loop that will span the range from 0 to n, and look to see if each number is in the set we just created. If it isn't, we return the missing number.

```
def missingNumber(nums):
    num_set = set(nums)
    n = len(nums) + 1
    for number in range(n):
        if number not in num_set:
            return number
```

The second solution requires formulating an equation. If we know that one number is supposed to be missing from 0 to n, then we can solve for the missing number by taking the sum of numbers from 0 to n and subtracting it from the sum of the input array with the missing value.

An equation for the sum of numbers from 0 to n is (n + 1) * (n + 2)/2.
Now all we have to do is apply the internal sum function to the input array, and then subtract the values from each other.

```
def missingNumber(nums):
    n = len(nums)
    total = (n + 1)*(n + 2)/2
    sum_of_nums = sum(nums)
    return total - sum_of_nums
``` -->
