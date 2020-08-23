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

A = TreeNode(5, TreeNode(4, TreeNode(3, None,None), TreeNode(2,None,None)), TreeNode(1, None,None))
B = TreeNode(5, TreeNode(3, None, None), TreeNode(2, None,None))
C = TreeNode(5, TreeNode(4,None, TreeNode(3, None,None)), None)
print("result")
# print(maxSubArray([-2, -1]))
# print(commonChars(["bella","label","roller"]))
#print(findUnsortedSubarray([2, 6, 4, 8, 10, 9, 15]))
print(reconstruct([[6, 0], [5, 0], [4, 0], [3, 2], [2, 2], [1, 4]]))
reconstruct([[9,0],[7,0],[1,9],[3,0],[2,7],[5,3],[6,0],[3,4],[6,2],[5,2]])
