from __future__ import print_function

def fib(int n):
    """Print the Fibonacci series up to n."""
    cdef int a = 0
    cdef int b = 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b
    print()
