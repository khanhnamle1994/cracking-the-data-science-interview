## Question
Suppose you're given a matrix of 1s and 0s that represents a map of rivers. You can assume that the grid cells in your map are only connected horizontally and vertically (e.g. no diagonal connections). You can assume that 1 represents water (your river) and 0 represents land/your river bank. Each cell has a length of 1 and is square in your map. Given this, write code to determine the perimeter of your river.

Examples:

```
Input: [[1,0]]
Output: 4

Input: [[1,0,1],
        [1,1,1]]
Output: 12
```

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1pdnGnxKPEq6_0XrX_IBN4LHX1vQ8h_1B) to view this solution in an interactive Colab (Jupyter) notebook.

Here we can set up a function to loop through each cell in the grid, then check left, right, up, and down. If any of the cells are on the edge, or next to land, then we increment our 'sides' variable which is tallying up the perimeter.

*If you're having trouble following any of the steps, it could be helpful to just add print(grid[i][j]) statements throughout to walk yourself through what's going on with the actual numbers.*

```
def island_perimeter(grid):
    # Initialize variable to count perimiter
    sides = 0
    num_rows = len(grid)
    num_cols = len(grid[0])
    for i in range(num_rows):
        for j in range(num_cols):
            if grid[i][j] == 1:
                # Check left, if we are at the edge (column wise) or the cell we are checking is water, increment sides
                if j == 0 or grid[i][j - 1] == 0:
                    sides += 1
                # Check right, if we are at the edge (column wise) or the cell we are checking is water, increment sides
                if j == num_cols - 1 or grid[i][j + 1] == 0:
                    sides += 1
                # Check up, if we are at the edge (row wise) or the cell we are checking is water, increment sides
                if i == 0 or grid[i - 1][j] == 0:
                    sides += 1
                # Check down, if we are at the edge (row wise) or the cell we are checking is water, increment sides
                if i == num_rows - 1 or grid[i + 1][j] == 0:
                    sides += 1
    return sides

# Driver code
grid = [[1,0,1],
        [1,1,1]]

island_perimeter(grid)    
#Output: 12
``` -->
