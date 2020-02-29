## Problem
Given an array of integers, can you write a function that returns `"True"` if there is a triplet `(a, b, c)` within the array that satisfies `a^2 + b^2 = c^2`?

For example:

```
Input: arr[] = [3, 1, 4, 6, 5]
Output: True
#There is a Pythagorean triplet (3, 4, 5) that exists in the input array.

Input: arr[] = {10, 4, 6, 12, 5}
Output: False
#There is no Pythagorean triplet that exists in the input array.
```

<!-- ## Solution:
[Click here](https://colab.research.google.com/drive/1qPf2XQG9u8Mpt-9AAYOrcxm22iOUICAD#scrollTo=syOxBrEDQaNU) to view this solution in an interactive Colab (Jupyter) notebook.

```
# defining the function
def isTriplet(array, n):
    # here we're staggering the element of the array by increments of 1 and 2
    # so we get a total of 3 elements to test on
    j=0
    for i in range(n - 2):
        for k in range(j + 1, n):
            for j in range(i + 1, n - 1):
                # calculating the square of the array "**" = power of so 3**2 = 9
                x = array[i]**2
                y = array[j]**2
                z = array[k]**2
                # testing to see if our selected triplet passes
                if (x == y + z or y == x + z or z == x + y):
                    return 1

    # if we didn't find a triplet we break out of the loop
    return 0
```

```
# Driver code to test above function  
array = [3, 1, 4, 6, 5]
array_size = len(array)

if(isTriplet(array, array_size)):
    print(True)
else:
    print(False)
``` -->
