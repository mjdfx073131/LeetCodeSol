class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



# 1309
def freqAlphabets(s: str) -> str:
    result = ''
    for i in range(0, len(s)):
        print(s[i])
        if (s[i] == '1' and i < len(s)-2 and s[i+2] == '#'):
            if (s[i+1] == '0'):
                result += 'j'
                i += 2
            elif s[i+1] == '1':
                result += 'k'
                i += 2
            elif s[i+1] == '2':
                result += 'l'
                i += 2
            elif s[i+1] == '3':
                result += 'm'
                i += 2
            elif s[i+1] == '4':
                result += 'n'
                i += 2
            elif s[i+1] == '5':
                result += 'o'
                i += 2
            elif s[i+1] == '6':
                result += 'p'
                i += 2
            elif s[i+1] == '7':
                result += 'q'
                i += 2
            elif s[i+1] == '8':
                result += 'r'
                i += 2
            elif s[i+1] == '9':
                result += 's'
                i += 2
        elif (s[i] == '2' and i < len(s)-2 and s[i+2] == '#'):
            if s[i+1] == '0':
                result += 't'
                i += 2
            elif s[i+1] == '1':
                result += 'u'
                i += 2
            elif s[i+1] == '2':
                result += 'v'
                i += 2
            elif s[i+1] == '3':
                result += 'w'
                i += 2
            elif s[i+1] == '4':
                result += 'x'
                i += 2
            elif s[i+1] == '5':
                result += 'y'
                i += 2
            elif s[i+1] == '6':
                result += 'z'
                i += 2
        else:
            if s[i] == '1':
                result += 'a'
            elif s[i] == '2':
                result += 'b'
            elif s[i] == '3':
                result += 'c'
            elif s[i] == '4':
                result += 'd'
            elif s[i] == '5':
                result += 'e'
            elif s[i] == '6':
                result += 'f'
            elif s[i] == '7':
                result += 'g'
            elif s[i] == '8':
                result += 'h'
            elif s[i] == '9':
                result += 'i'
    return result
# 1299


def replaceElements(arr):
    for i in range(len(arr)):
        arr[i] = max(arr[(i+1):])
    return arr


def flipAndInvertImage(n):
    if (n <= 0):
        return 1
    return flipAndInvertImage(n-1) + flipAndInvertImage(n-2)


def maxSubArray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    twoDArray = [[None for i in range(len(nums))] for i in range (len(nums))]
    #print(twoDArray)
    maxSum = None
    for i in range(0, len(nums)):
        subSum = None
        for j in range(i, len(nums)):
            #print(twoDArray[i][j] == None)
            if (twoDArray[i][j] is None and subSum == None):
                twoDArray[i][j] = nums[i]
                subSum = twoDArray[i][j]
            else:
                twoDArray[i][j] = nums[j] + twoDArray[i][j-1]
                subSum = twoDArray[i][j]
            if (maxSum == None or twoDArray[i][j] > maxSum):
                maxSum = twoDArray[i][j]
    print(twoDArray)
    return maxSum

def rob(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    result = []
    for i in range(0, len(nums)):
        result.append(0)

    def robHelper(result, nums, idx):
        if (idx >= len(nums)):
            return 0
        if result[idx] == 0:
            #result[idx] = nums[idx]
            money = max(nums[idx] + robHelper(result, nums,
                                              idx+2), robHelper(result, nums, idx+1))
            if (money > result[idx]):
                result[idx] = money
            # else:
            #     return result[j]
        return result[idx]
    
    return robHelper(result, nums, 0)

def findUnsortedSubarray(nums) -> int:
    sortedArray = sorted(nums)
    first = 0
    last = len(nums)-1
    while first != len(nums)-1:
        if (nums[first] == sortedArray[first]):
            first += 1
        else:
            slowest = first
            break
    while last != 0:
        if (nums[last] == sortedArray[last]):
            last -= 1
        else:
            largest = last
            break
    if(first - last == len(nums) -1):
        return 0
    return largest - slowest + 1


def reconstruct(people):
    sortedPpl = sorted(people)
    middlePpl = []
    for i in range (0, len(sortedPpl)):
        if (not middlePpl):
            middlePpl.append(sortedPpl[i])
        else:
            if (sortedPpl[i][0] > middlePpl[0][0]):
                middlePpl.insert(0, sortedPpl[i])
            else:
                middlePpl.insert(middlePpl.index(sortedPpl[i-1])+1, sortedPpl[i])
            
    result = []
    for person in middlePpl:
        if (not result):
            result.append(person)
        else:
            result.insert(person[1], person)
    return result

def isPrefixOfWord(sentence, searchWord):
    """
    :type sentence: str
    :type searchWord: str
    :rtype: int
    """
    arr = sentence.split(" ")
    prefixLength = len(searchWord)
    for i in range(len(arr)):
        print(arr[i][: (prefixLength)])
        if len(arr[i]) > prefixLength and arr[i][: prefixLength] == searchWord:
            return (i+1)
    return -1
#Daily Temperatures

def dailyTemperatures(T):
    """
    :type T: List[int]
    :rtype: List[int]
    """
    if (len(T) == 0 or len(T) == 1):
        return [0]
    wait = [0]*len(T)
    stack = []

    for i, x in enumerate(T):
        while stack and x > T[stack[-1]]:
            j = stack.pop()
            wait[j] = i - j
        stack.append(i)

    return wait

def generateParenthesis(nums):
    """
    :type n: int
    :rtype: List[str]
    """
    result = []
    currSubset = []

    def createSubsets(result, idx, currSubset):
        if (idx == len(nums) - 1):
            newSubset = currSubset.copy()
            newSubset.append(nums[idx])
            result.append(newSubset)
            return
        else:
            newSubset = currSubset.copy()
            newSubset.append(nums[idx])
            result.append(newSubset)
            createSubsets(result, idx+1, newSubset)
            if (idx+2 <= len(nums)-1):
                for j in range(idx+2, len(nums)):
                    createSubsets(result, j, newSubset)
    result.append(currSubset)
    for i in range(0, len(nums)):
        createSubsets(result, i, [])
    return result
def countSubstrings(s):
    result = 0
    def checkPalindromic(s):
        return s == s[::-1]

    def checkSubstr(s, currStr, idx):
        nonlocal result
        if (idx == len(s) -1):
            if(checkPalindromic(currStr + s[idx])):
                result +=1
                return
        else:
            newStr = currStr + s[idx]
            if(checkPalindromic(newStr)):
                result +=1
            checkSubstr(s, newStr, idx+1)
    for i in range (len(s)):
        checkSubstr(s, "", i)
    return result

def combinationSum(candidates, target):
    result = []

    def createSum(target, currSum, idx, result, currArr):
        if(currSum == target):
            newSubset = currArr.copy()
            newSubset.append(candidates[idx])
            result.append(newSubset)
            return
        elif(currSum > target or idx >= len(candidates)-1):
            return
        else:
            if (currSum + candidates[idx] <= target):
                newSubset = currArr.copy()
                newSubset.append(candidates[idx])
                # if currSum + candidates[idx] == target:
                #     result.append(newSubset)
                #     return
                createSum(target, currSum + candidates[idx], idx, result, newSubset)
            createSum(target, currSum +
                      candidates[idx+1], idx+1, result, currArr)
    for i in range(len(candidates)):
        createSum(target, 0, i, result, [])
    return result

def minPathSum(grid) -> int:
    m = len(grid)
    n = len(grid[0])
    result = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            if i == 0:
                result[i][j] = result[i][j-1] + grid[i][j]
                continue
            if j == 0:
                result[i][j] = result[i-1][j] + grid[i][j]
                continue
            else:
                result[i][j] = grid[i][j] + min(result[i-1][j], result[i][j-1])
    print(result)
    return result[m-1][n-1]

def numTrees(n):
    result = [1] * (n+2)
    if n == 0:
        return 0
    if n == 1:
        return 1
    for i in range(2, n+1):
        sum = 0
        for j in range(0, i):
            left = j
            right = i-1-j
            sum += (result[left] * result[right])
        result[i] = sum
    return result[n]

def partitionLabels(S):
    dic = {}
    for i in range(len(S)):
        if S[i] not in dic:
            dic[S[i]] = [i]
        else:
            dic[S[i]].append(i)
    print(dic)
    result = []
    currLen = 0
    start = 0
    end = 0
    for char in dic:
        if currLen == 0:
            currLen = dic[char][-1] - dic[char][0]+1
            start = dic[char][0]
            end = dic[char][-1]
        else:
            if dic[char][0] < end:
                if dic[char][-1] > end:
                    end = dic[char][-1]
            else:
                rst = end - start + 1
                result.append(rst)
                currLen = dic[char][-1] - dic[char][0]+1
                start = dic[char][0]
                end = dic[char][-1]
        if (list(dic)[-1] == char):
            rst = end - start + 1
            result.append(rst)
    return result


def diagonalSum(mat):
    if len(mat) == 1:
        return mat[0][0]
    sum = 0

    for i, row in enumerate(mat):
        if i == len(row)-i-1:
            sum += (row[i])
        else:
            sum += (row[i] + row[len(row)-i-1])
    return sum

A = TreeNode(5, TreeNode(4, TreeNode(3, None, None),
                         TreeNode(2, None, None)), TreeNode(1, None, None))
B = TreeNode(5, TreeNode(3, None, None), TreeNode(2, None,None))
C = TreeNode(5, TreeNode(4,None, TreeNode(3, None,None)), None)
print("result")
# print(maxSubArray([-2, -1]))
# print(commonChars(["bella","label","roller"]))
#print(findUnsortedSubarray([2, 6, 4, 8, 10, 9, 15]))
# print(reconstruct([[6, 0], [5, 0], [4, 0], [3, 2], [2, 2], [1, 4]]))
# reconstruct([[9,0],[7,0],[1,9],[3,0],[2,7],[5,3],[6,0],[3,4],[6,2],[5,2]])
# print(dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))
#print(countSubstrings("abc"))
# print(minPathSum([[1,3,1],[1,5,1],[4,2,1]]))
#print(partitionLabels("vhaagbqkaq"))
print(diagonalSum([[1,2,3],[4,5,6],[7,8,9]]))


# 

def solution (G):
    result =[0,0,0]
    for i in range(len(G)):
        if G[i] == "R":
            result[0] +=2
            result[2] +=1
        elif G[i] == "S":
            result[2] +=2
            result[1] +=1
        else:
            result[0] +=1
            result[1] +=2
    return max(result)
