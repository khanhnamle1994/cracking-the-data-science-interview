## Question
Suppose you're given a portion of a phone number. Each digit corresponds to letters. Using Python, write code to return all possible combinations the given number could represent.

For example:

```
Input: "24"

Output:
['a', 'g']
['a', 'h']
['a', 'i']
['b', 'g']
['b', 'h']
['b', 'i']
['c', 'g']
['c', 'h']
['c', 'i']
```

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1jc0Ey8filNAXPvNZ90PGUIRArHJZelrC) to view this solution in an interactive Colab (Jupyter) notebook.

```
# First, create table to reference letter
# to digit mappings, starting at 0, 1, etc
digitMap = ["", "", "abc", "def", "ghi", "jkl",  
                    "mno", "pqrs", "tuv", "wxyz"]

#Print all possible letters than can be obtained from input  
def printLetters(number, curr, output, n):
    if(curr == n):
        print(output)
        return

    # Try all 3 possible characters  
    # for current digit in number[],
    # continue looping for remaining digits
    for i in range(len(digitMap[number[curr]])):
        output.append(digitMap[number[curr]][i])
        printLetters(number, curr + 1, output, n)
        #Return last value from list
        output.pop()
        #If 0 or 1, no letters associated
        if(number[curr] == 0 or number[curr] == 1):
            return;  

# A wrapper over printLetters().  
# It creates our output array and  
# calls printLetters()  
def printLettersWrapper(number, n):
    printLetters(number, 0, [], n)

# Driver function
if __name__ == '__main__':
    number = [2, 4]
    n = len(number)
    printLettersWrapper(number, n);
``` -->
