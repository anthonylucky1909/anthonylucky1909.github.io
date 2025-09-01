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

# Important Dynamic Programming Problems (LeetCode Roadmap)

| Status | ⭐ | Problem | Difficulty | Solution (DP Idea) |
|--------|----|---------|------------|--------------------|
| ⬜ | ⭐ | [Stone Game III](https://leetcode.com/problems/stone-game-iii/) | Hard | DFS + memo on index, player’s turn. Compare score difference. |
| ⬜ | ⭐ | [House Robber](https://leetcode.com/problems/house-robber/) | Medium | 1D DP: `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`. |
| ⬜ | ⭐ | [House Robber II](https://leetcode.com/problems/house-robber-ii/) | Medium | Circle case → run House Robber twice (exclude first / last). |
| ⬜ | ⭐ | [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) | Medium | Expand around center OR DP[i][j] = s[i]==s[j] and dp[i+1][j-1]. |
| ⬜ | ⭐ | [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/) | Medium | Expand around center / DP on substrings. Count palindromes. |
| ⬜ | ⭐ | [Decode Ways](https://leetcode.com/problems/decode-ways/) | Medium | `dp[i] = dp[i-1] if s[i]!=0 + dp[i-2] if valid 2-digit`. |
| ⬜ | ⭐ | [Coin Change](https://leetcode.com/problems/coin-change/) | Medium | DFS + memo / Bottom-up: min coins for each amount. |
| ⬜ | ⭐ | [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) | Medium | Track max/min product at each step (due to negatives). |
| ⬜ | ⭐ | [Word Break](https://leetcode.com/problems/word-break/) | Medium | DFS + memo: can split `s[i:]` if word in dict and rest valid. |
| ⬜ | ⭐ | [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) | Medium | DFS + memo on (i, prev) OR `dp[i] = 1 + max(dp[j])`. |
| ⬜ | ⭐ | [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) | Medium | Subset-sum DP (knapsack). Target = total/2. |
| ⬜ | ⭐ | [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/) | Medium | Order matters → DFS + memo count ways. |
| ⬜ | ⭐ | [Perfect Squares](https://leetcode.com/problems/perfect-squares/) | Medium | `dp[i] = 1 + min(dp[i - sq])`. Classic BFS/DP. |
| ⬜ | ⭐ | [Integer Break](https://leetcode.com/problems/integer-break/) | Medium | Max product by splitting. Try all cuts. |
| ⬜ | ⭐ | [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) | Easy | Fibonacci DP: `dp[i] = dp[i-1] + dp[i-2]`. |
| ⬜ | ⭐ | [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) | Easy | `dp[i] = cost[i] + min(dp[i-1], dp[i-2])`. |
| ⬜ | ⭐ | [N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/) | Easy | Similar to Fibonacci but sum of 3 prev values. |
| ⬜ | ⭐ | [Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) | Hard | DFS + memo on each cell with 4 directions. |
| ⬜ | ⭐ | [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/) | Hard | `dp[i][j]`: ways s[i:] matches t[j:]. |
| ⬜ | ⭐ | [Burst Balloons](https://leetcode.com/problems/burst-balloons/) | Hard | Interval DP: `dp[i][j]` = max coins if burst (i..j). |
| ⬜ | ⭐ | [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/) | Hard | `dp[i][j]` = match s[:i], p[:j]. Handle `*` and `.` cases. |
| ⬜ | ⭐ | [Unique Paths](https://leetcode.com/problems/unique-paths/) | Medium | Grid DP: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`. |
| ⬜ | ⭐ | [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/) | Medium | Same as Unique Paths but block cells = 0 ways. |
| ⬜ | ⭐ | [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) | Medium | Grid DP: `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`. |
| ⬜ | ⭐ | [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) | Medium | `dp[i][j]` = if equal → +1, else max(left, up). |
| ⬜ | ⭐ | [Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/) | Medium | Subset-sum DP minimize diff. |
| ⬜ | ⭐ | [Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | Medium | DP with states: hold / sold / cooldown. |
| ⬜ | ⭐ | [Coin Change II](https://leetcode.com/problems/coin-change-ii/) | Medium | Count ways (unbounded knapsack). |
| ⬜ | ⭐ | [Target Sum](https://leetcode.com/problems/target-sum/) | Medium | DFS + memo OR reduce to subset sum. |
| ⬜ | ⭐ | [Interleaving String](https://leetcode.com/problems/interleaving-string/) | Medium | `dp[i][j]` = can s1[:i], s2[:j] form s3[:i+j]. |
| ⬜ | ⭐ | [Stone Game](https://leetcode.com/problems/stone-game/) | Medium | Game DP: compare score left vs right choice. |
| ⬜ | ⭐ | [Stone Game II](https://leetcode.com/problems/stone-game-ii/) | Medium | DFS + memo with `M` parameter (range of moves). |
| ⬜ | ⭐ | [Edit Distance](https://leetcode.com/problems/edit-distance/) | Hard | `dp[i][j]` = min(insert, delete, replace). |

---

## References

Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press. LeetCode & NeetCode Practice Problems. [https://neetcode.io/](https://neetcode.io/)
