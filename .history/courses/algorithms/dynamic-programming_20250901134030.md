# Fundamental Dynamic Programming in Algorithms

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python\&logoColor=white)](https://www.python.org/)

---

## Overview

Dynamic Programming (DP) is a powerful algorithmic technique used to solve complex problems by breaking them down into simpler sub-problems that overlap. It is particularly effective when the problem exhibits optimal substructure, meaning that the solution to a larger problem can be constructed from optimal solutions to its sub-problems, and overlapping subproblems, where the same smaller problems are solved multiple times. Unlike naive recursive approaches that recompute the same values repeatedly, DP leverages memoization or tabulation to store intermediate results, significantly improving efficiency. This technique is widely applied in problems involving sequences, grids, subsets, and combinatorial optimization, such as Fibonacci numbers, climbing stairs, coin change, longest increasing subsequence, and partitioning problems.

---

## Dynamic Programming Approaches

Dynamic Programming can be implemented using either a top-down or bottom-up approach. The top-down approach, also known as memoization, uses recursion combined with a cache to store the results of sub-problems. This allows the algorithm to compute only the necessary sub-problems, avoiding redundant computations. The bottom-up approach, or tabulation, constructs a solution iteratively from the simplest sub-problems up to the desired final problem, often using an array or table to store intermediate results. Bottom-up DP is generally more space- and time-efficient and is suitable for problems where the recursive structure is straightforward to translate into iterative computation.

---

## Strategy for Solving DP Problems

Solving DP problems requires a structured approach. The first step is understanding the problem and identifying the state, which represents the parameters that define a sub-problem. The second step is formulating the recurrence relation, which expresses how the solution to a problem can be derived from the solutions of its sub-problems. The third step involves defining base cases, which are the smallest sub-problems with known solutions that serve as starting points. After this, one decides whether a top-down or bottom-up implementation is more appropriate, considering efficiency and clarity. Additionally, space optimization techniques such as rolling arrays can often reduce memory usage when only a few previous states are needed to compute the current state.

---

## Common Types of DP Problems

Dynamic Programming problems can be categorized based on their structure and state dimensions. One-dimensional DP problems, such as climbing stairs or house robber, depend on a single variable, often representing a position in a sequence. Two-dimensional DP problems, including unique paths in grids or longest common subsequence, require two parameters, often corresponding to indices in sequences or matrices. Subset or bitmask DP problems, such as partition equal subset sum or target sum, represent subsets compactly using bits. String-based DP problems, like palindromes, decode ways, or edit distance, use string indices as states, while matrix DP problems, including minimum path sum or longest increasing path, define states as positions within a grid. Each type relies on the same principle of building solutions from sub-problems but differs in how the state is represented and traversed.

---

## Implementation in Python

The following example illustrates a bottom-up dynamic programming solution to compute the nth Fibonacci number efficiently. This approach uses an iterative table to store intermediate results, ensuring each sub-problem is computed only once and re-used in subsequent calculations.

```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

print(fibonacci(10)) 
```

---

## Advantages of Dynamic Programming

Dynamic Programming provides a systematic method to tackle complex optimization and counting problems by reusing previously computed results. It reduces the exponential time complexity of naive recursive solutions to polynomial time in many cases, making intractable problems solvable efficiently. Furthermore, DP can be adapted to minimize memory usage with space-optimized variants and is flexible enough to handle a wide variety of problems, from sequences and strings to grids and subsets. Its ability to capture overlapping computations and build optimal solutions incrementally makes it a cornerstone of algorithm design.

---

## References

Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. LeetCode & NeetCode Practice Problems. [https://neetcode.io/](https://neetcode.io/)
