"""
This is the core of deep learning: (1) Take an input and desired output, (2) Search for their correlation
"""

def compute_error(b, m, coordinates):
    """
    m is the coefficient and b is the constant for prediction
    The goal is to find a combination of m and b where the error is as small as possible
    coordinates are the locations
    """
    totalError = 0
    for i in range(0, len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(coordinates))

# Example
error = compute_error(1, 2, [[3, 6], [6, 9], [12, 18]])
print(error)
