class Solution:
    def longestPalindrome(self, s: str) -> str:
        reversed_str = s[::-1]
        

def longestPalindrome(s):
    rs = s[::-1]

    m, n = len(s), len(s)
    soln = [["" for i in range(m)] for j in range(n)]
    for i in range(0, n):
        for j in range(0, m):
            print(rs[:(i+1)])
            print(s[:(j+1)])
            if rs[:(i+1)] == s[:(j+1)]:
                soln[i][j] = soln[i-1][j-1] + rs[i] 
            else:
                soln[i][j] = max(soln[i][j-1], soln[i-1][j])
    print(soln)
    return soln[n-1][m-1]


print(longestPalindrome("babad"))
