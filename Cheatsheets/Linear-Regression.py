"""
Combining the method of least square and gradient descent, you get linear regression.
"""

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))

    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2/n) * (y - (m_current * x + b_current))
        m_gradient += -(2/n) * x * (y - (m_current * x + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, epochs):
    b = starting_b
    m = starting_m

    for i in range(epochs):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]

# Example on price of what/kg and the average price of bread
wheat_and_bread = [
    [0.5, 5], [0.6, 5.5], [0.8, 6], [1.1, 6.8], [1.4, 7]
]
result = gradient_descent_runner(wheat_and_bread, 1, 1, 0.01, 100)
print(result)
