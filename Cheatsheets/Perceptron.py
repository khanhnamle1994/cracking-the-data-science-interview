"""
For every training circle, you start with input data to the left. Initial random weights are added to all the input data. They are then summed up. If the sum is negative, it's translated into 0, otherwise, it's mapped to 1.
"""

from random import choice
from numpy import array, dot, random

unit_step = lambda x: 0 if x < 0 else 1
training_data = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]
w = random.rand(3)
errors = []
eta = 0.2
epochs = 100

for i in range(epochs):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x

for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
