## Problem
This problem was asked by Lyft.

How many cards would you expect to draw from a standard deck before seeing the first ace?

## Solution
Although one can do out all the probabilities, it gets a bit messy from an algebra standpoint, so it is easier to have the following intuitive answer. Imagine we have aces A1, A2, A3, and A4. We can draw a line in between them to represent an arbitrary number (including 0) of cards between each ace, with a line before the first ace and after the last.

`| A1 | A2 | A3 | A4 |`

There are 52 - 4 = 48 non-ace cards. Each of those cards is equally likely to be in any of the 5 lines, therefore there should be 48/5 = 9.6 cards in front of the first ace. Therefore, the total expectation of cards drawn until the first ace is seen = 9.6 + 1 (for the ace itself) = 10.6 cards.
