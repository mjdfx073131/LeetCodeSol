from itertools import permutations
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

#155
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.n = 0
        self.stack = []
        self.minimum = []

    def push(self, x: int) -> None:
        if self.n == 0:
            self.stack.append(x)
            self.minimum.append(x)
        else:
            self.stack.append(x)
            self.minimum.append(min(x, self.minimum[self.n-1]))

        self.n += 1

    def pop(self) -> None:
        self.minimum.pop(-1)
        self.n -= 1
        return self.stack.pop(-1)

    def top(self) -> int:
        if self.n == 0:
            return None
        else:
            return self.stack[self.n-1]

    def getMin(self) -> int:
        return self.minimum[self.n-1]




class Solution(object):
    # 104 Maximum depth of binary tree
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if (root == None):
            return 0
        depth = []
        initDepth = 1

        def recursion(currNode, currDepth):
            if currNode.left == None and currNode.right == None:
                depth.append(currDepth)
            else:
                if currNode.left != None:
                    newD = currDepth + 1
                    recursion(currNode.left, newD)
                if currNode.right != None:
                    newD = currDepth + 1
                    recursion(currNode.right, newD)
        recursion(root, initDepth)
        return max(depth)

    def maxDepthAnotherVersion(self, root):
        if(root is None):
            return 0
        else:
            return 1 + max(self.maxDepthAnotherVersion(root.left), self.maxDepthAnotherVersion(root.right))

    # 136 Single Number should have a runtime complexity without using extra memory
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for num in nums:
            if (nums.count(num) > 1):
                continue
            else:
                return num

    def singleNumber1684ms(self, nums):
        copy = []
        for num in nums:
            if num in copy:
                copy.remove(num)
            else:
                copy.append(num)
        return copy[0]

    def singleNumber80ms(self, nums):
        dic = {}
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                dic.pop(num)
        return dic.keys()[0]

    # 206
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        curr = head
        prev = None
        while(curr is not None):
            Next = curr.next
            curr.next = prev
            prev = curr
            curr = Next
        head = prev
        return head

    # 169
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dic = {}
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                dic[num] += 1
        for key in dic.keys():
            if dic[key] > len(nums)/2:
                return key

    # 283
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        count = nums.count(0)
        while (count != 0):
            for i in range(len(nums)-1):
                if nums[i] == 0 and nums[i+1] != 0:
                    nums[i] = nums[i+1]
                    nums[i+1] = 0
                elif nums[i] == 0 and nums[i+1] == 0:
                    continue
            count -= 1

    # 448 find disappear number in an array
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        check = 1
        dic = {}
        result = []
        for i in range(len(nums)):
            dic[i+1] = False
        for num in nums:
            dic[num] = True

        for key in dic.keys():
            if dic[key] == False:
                result.append(key)
        return result

    # 21 merge two sorted lists
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = ListNode(None)
        curr = head
        left = l1
        right = l2
        while (left != None and right != None):
            newLeft = ListNode(left.val)
            newRight = ListNode(right.val)
            if left.val < right.val:
                curr.next = newLeft
                curr = curr.next
                left = left.next
            elif right.val < left.val:
                curr.next = newRight
                curr = curr.next
                right = right.next
            elif right.val == left.val:
                curr.next = newLeft
                curr.next.next = newRight
                curr = curr.next.next
                right = right.next
                left = left.next
        while (left != None):
            curr.next = ListNode(left.val)
            left = left.next
            curr = curr.next
        while (right != None):
            curr.next = ListNode(right.val)
            right = right.next
            curr = curr.next
        return head.next

    # 121 Best tiem to buy and sell stock
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if (len(prices) == 0):
            return 0
        if (len(prices) == 1):
            return 0
        if(all(prices[i] >= prices[i+1] for i in range(len(prices)-1))):
            return 0
        maxProfit = 0
        for i in range(len(prices)-1):
            buy = prices[i]
            for j in range(i+1, len(prices)):
                if (buy < prices[j]):
                    profit = prices[j] - buy
                    if (profit > maxProfit):
                        maxProfit = profit
        return maxProfit

    def maxProfitBetterSolution(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        minPrice = float('inf')
        maxProfit = 0
        for price in prices:
            if price < minPrice:
                minPrice = price
            elif price - minPrice > maxProfit:
                maxProfit = price - minPrice
        return maxProfit

    #543 diameter of binary tree
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        def helper(root):
            if not root:
                return 0,0
            else:
                l,ll = helper(root.left)
                r,rr = helper(root.right)
                return max(l,r)+1, max(l+r,ll,rr)
        a, b = helper(root)
        return max(a-1,b)

    #70 climb stairs each time you can climb either 1 or 2 steps
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        def recursion (CurrSteps):
            nonlocal count 
            if n == 0:
                count +=1
            else:
                if (n-1)> 0:
                    left = CurrSteps -1
                    recursion(left)
                if (n-2) > 0:
                    left = CurrSteps -2
                    recursion (left)
        recursion(n)
        return count
    def dpClimbStairs(self, n):
        if n < 2:
            return n
        dp = [0] * (n+1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

    #101. Symmetric Tree
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if(root == None):
            return True
        if(root.left == None and root.right == None):
            return True
        def isMirror(left, right):
            if (left == None or right == None):
                return left == right
            elif (left.val != right.val):
                return False
            return isMirror(left.left, right.right) and isMirror(left.right, right.left)
        return isMirror(root.left, root.right)


    #53. Maximum Subarray
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # twoDArray = [[None for i in range(len(nums))] for i in range(len(nums))]
        # #print(twoDArray)
        # maxSum = None
        # for i in range(0, len(nums)):
        #     subSum = None
        #     for j in range(i, len(nums)):
        #         #print(twoDArray[i][j] == None)
        #         if (twoDArray[i][j] is None and subSum == None):
        #             twoDArray[i][j] = nums[i]
        #             subSum = twoDArray[i][j]
        #         else:
        #             twoDArray[i][j] = nums[j] + twoDArray[i][j-1]
        #             subSum = twoDArray[i][j]
        #         if (maxSum == None or twoDArray[i][j] > maxSum):
        #             maxSum = twoDArray[i][j]
        # print(twoDArray)
        # return maxSum
        maxSub, curSum = nums[0], 0
        for n in nums:
            if curSum <0:
                curSum = 0
            curSum += n
            maxSub = max(maxSub, curSum)
        return maxSub



    #1. Two Sum
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {}
        for i in range (len(nums)):
            diff = target - nums[i]
            if diff in d:
                return[d[diff], i]
            d[num[i]] = i

    #198
    def rob(self, nums):
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
    #198
    def robDP(self, nums):
        if (len(nums) == 0):
            return 0

        if (len(nums) == 1):
            return nums[0]
        result = []
        result.append(nums[0])
        result.append(max(result[0], nums[1]))
        for i in range(2, len(nums)):
            result.append(max(result[i-2] + nums[i], result[i-1]))
        return result[-1]
            

    #141 Linked List Cycle

    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if(head == None):
            return False
        result = {}
        while (head.next != None):
            if (head not in result):
                result[head] = False
            else:
                return True
            head = head.next
        return False

    def hasCycleAnotherSolution(self, head):
        if(head == None):
            return False
        slow = head
        fast = head.next
        while fast is not None and fast.next is not None:
            if (slow == fast):
                return True
            slow = slow.next
            fast = fast.next.next

    #160 intersection of Two Linked List
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if (headA == None or headB == None):
            return None
        up = headA
        down = headB
        result = set()
        while up is not None:
            result.add(up)
            up = up.next
        while down is not None:
            if (down in result):
                return down
            down = down.next
        return None
    
    #234. Palindrome Linked List
    def isPalindrome(self, head: ListNode) -> bool:
        stack = []
        copy = head
        while(head != None):
            stack.append(head.val)
            head = head.next
        for i in range(len(stack)-1, -1, -1):
            if (stack[i] != copy.val):
                return False
            copy = copy.next
        return True
    
    #20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        stack =[]
        for bracket in s:
            if len(stack) == 0:
                stack.append(bracket)
            else:
                if bracket == "(" or bracket == "{" or bracket=="[":
                    stack.append(bracket)
                elif bracket == ")":
                    if stack[len(stack)-1] == "(":
                        stack.pop()
                    else:
                        return False
                elif bracket == "]":
                    if stack[len(stack)-1] == "[":
                        stack.pop()
                    else:
                        return False
                elif bracket == "}":
                    if stack[len(stack)-1] == "{":
                        stack.pop()
                    else:
                        return False
        if(len(stack) == 0):
            return True
        return False

    #581. Shortest Unsorted Continuous Subarray
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        sortedArray = nums.sort()
        first = 0
        last =len(nums)-1
        while first != last:
            if (nums[first] == sortedArray[first]):
                first +=1
                break
        while first != last:
            if (nums[last] == sortedArray[last]):
                last -=1
                break
        if first == last:
            return 0
        else :
            return last -first +1

    # ***************************Medium******************************

    #338.Counting Bits
    def countBits(self, num: int) -> List[int]:
        result = []
        for i in range (num+1):
            result.append(str(bin(i+1)).count("1"))
        return result

    
        
    #406. Queue Reconstruction by Height
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        #Before: [[9,0],[7,0],[1,9],[3,0],[2,7],[5,3],[6,0],[3,4],[6,2],[5,2]]
        sortedPpl = sorted(people)
        #After first sort: [[1, 9], [2, 7], [3, 0], [3, 4], [5, 2], [5, 3], [6, 0], [6, 2], [7, 0], [9, 0]]
        middlePpl = []
        for i in range (0, len(sortedPpl)):
            if (not middlePpl):
                middlePpl.append(sortedPpl[i])
            else:
                if (sortedPpl[i][0] > middlePpl[0][0]):
                    middlePpl.insert(0, sortedPpl[i])
                else:
                    middlePpl.insert(middlePpl.index(sortedPpl[i-1])+1, sortedPpl[i])
        # After second sort: [[9, 0], [7, 0], [6, 0], [6, 2], [5, 2], [5, 3], [3, 0], [3, 4], [2, 7], [1, 9]]
        result = []
        for person in middlePpl:
            if (not result):
                result.append(person)
            else:
                result.insert(person[1], person)
        return result

    #46. Permutations
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        perm = permutations(nums)
        result = []
        # Print the obtained permutations
        for i in list(perm):
            result.append(list(i))
        return result
    


    #94. Binary Tree Inorder Traversal   Very Important knowledge
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        def traverse (node, result):
            if node is None:
                return
            else:
                traverse(node.left, result)
                result.append(node.val)
                traverse(node.right, result)
        traverse(root, result)
        return result
    

    #739. Daily Temperatures 
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        if (len(T) == 0 or len(T) == 1):
            return [0]
        result = [0]*len(T)
        stack = []

        for i, x in enumerate(T):
            while stack and x > T[stack[-1]]:
                j = stack.pop()
                result[j] = i - j
            stack.append(i)

        return result
        
    #22. Generate Parentheses
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        currOpenBra = 0
        currCloseBra = 0

        def createParenthesis(n, currStr, currOpenBra, currCloseBra, result):
            if len(currStr) == 2*n:
                result.append(currStr)
                return
            else:
                if currOpenBra < n:
                    createParenthesis(
                        n, currStr + "(", currOpenBra + 1, currCloseBra, result)
                if currOpenBra > currCloseBra:
                    createParenthesis(n, currStr + ")",
                                      currOpenBra, currCloseBra + 1, result)
        createParenthesis(n, "", currOpenBra, currCloseBra, result)
        return result
        
    
    #78. Subsets
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
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
    #347. Top K Freduent Elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        result = []
        dic = {}
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                dic[num] += 1
        topK = sorted(dic.values())
        for num in dic:
            if dic[num] in topK[len(topK)-k:]:
                result.append(num)
        return result

    #647. Palindromic Substrings
    def countSubstrings(self, s: str) -> int:
        result = 0
        def checkPalindromic(s):
            return s == s[::-1]

        def checkSubstr(s, currStr, idx, result):
            if (idx == len(s) -1):
                if(checkPalindromic(currStr + s[idx])):
                    result +=1
                    return
            else:
                if(checkPalindromic(currStr + s[idx])):
                    result +=1
                checkSubstr(s, currStr, idx+1, result)
        checkSubstr(s, "", 0, result)
        return result


    #238. Product of Array Except Self 
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        pre = []
        post= []
        for i in range (len(nums)):
            if i ==0 :
                pre.append(1)
            else:
                pre.append(pre[-1]*nums[i-1])
        for j in range (len(nums)-1, -1, -1):
            if j == len(nums)-1:
                post.append(1)
            else:
                post.append(post[-1] * nums[j+1])
        result = []
        for i in range (len(nums)):
            result.append(pre[i]*post[len(nums)-i-1])
        return result             

            
    #49. Group Anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = {}
        for str in strs:
            sortedArr = sorted(str)
            string = ""
            for char in sortedArr:
                string += char
            if string not in dic:
                dic[string] = [str]
            else:
                dic[string].append(str)
        return dic.values()

    #48.Rotate Image
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        for i in range (len(matrix)// 2):
            for j in range (i, len(matrix[i])-1-i):
                temp = matrix[i][j]
                matrix[i][j] = matrix[len(matrix)-1-j][i]
                matrix[len(matrix)-1-j][i] = matrix[len(matrix)-1-i][len(matrix)-1 -j]
                matrix[len(matrix)-1-i][len(matrix)-1-j] = matrix[j][len(matrix)-1-i]
                matrix[j][len(matrix)-1-i] = temp

    #39. Combination Sum
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if target == 0:
            return [[]]
        ans = []
        for i, n in enumerate(candidates):
            if target >= n:
                ans += [[n] +
                        l for l in self.combinationSum(candidates[i:], target - n)]
        return ans


    #287. Find the Duplicate Number

    def findDuplicate(self, nums: List[int]) -> int:
        st = set()
        for num in nums:
            if num not in st:
                st.add(num)
            else:
                return num

    #215. Kth Largest Element in an Array]
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return sorted(nums)[len(nums)-k]

    #102. Binary Tree Order Traversal
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []
        dic = {0: []}
        def bfs(node, level):
            if(node == None):
                return
            dic[level].append(node.val)
            if node.left != None or node.right != None:
                if (level+1) not in dic:
                    dic[level+1] = []
                bfs(node.left, level+1)
                bfs(node.right, level+1)
        bfs(root, 0)
        return dic.values()

    #64. Minimum Path Sum
    def minPathSum(self, grid: List[List[int]]) -> int:
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
    
    def minPathSum1(self, grid: List[List[int]]) -> int:
        result = []
        m = len(grid)
        n = len(grid[0])

        def pathSum(grid, currSum, currX, currY):
            if (currX == len(grid)-1 and currY == len(grid[currX])-1):
                result.append(currSum)
            else:
                currSum += grid[currX][currY]
                if (currX == len(grid)-1):
                    pathSum(grid, currSum, currX, currY+1)
                elif (currY == n-1):
                    pathSum(grid, currSum, currX+1, currY)
                else:
                    pathSum(grid, currSum, currX, currY+1)
                    pathSum(grid, currSum, currX+1, currY)
        pathSum(grid, 0, 0, 0)
        return min(result)

    def minPathSum2(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        result = [[0 for i in range(n)] for j in range(m)]
        result[0][0] = grid[0][0]
        for i in range(1, n):
            result[0][i] = grid[0][i] + result[0][i-1]
        for j in range(1, m):
            result[j][0] = grid[j][0] + result[j-1][0]
        for i in range(1, m):
            for j in range(1, n):
                result[i][j] = grid[i][j] + min(result[i-1][j], result[i][j-1])
        return result[m-1][n-1]

    #62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        if (m == 1 or n == 1):
            return 1
        result = [[0 for i in range(m)] for j in range(n)]
        for i in range(1, m):
            result[0][i] = 1
        for j in range(1, n):
            result[j][0] = 1
        for i in range(1, n):
            for j in range(1, m):
                result[i][j] = result[i-1][j] + result[i][j-1]
        return result[n-1][m-1]

    def uniquePaths2(self, m: int, n: int) -> int:
        result = [[1]*m] * n
        for i in range(1, n):
            for j in range(1, m):
                result[i][j] = result[i-1][j] + result[i][j-1]
        return result[n-1][m-1]


    #96. Unique Binary Search Tree
    def numTrees(self, n: int) -> int:
        '''
        if 0 then 0
        if 1 then 1
        if 2 then we treat 2 as 1,2:
                then 1 there is only 2 on the right then 1
                then 2 there is only 1 on the left then 1
                then sum them up got 2
        if 3 then 1,2,3
            then 1 there are 2,3 on the right, then 2
            then 2 1 on the left 3 on the right then 1
            then 3 there are 1,2 on the left, then 2
            total 5
        '''
        result = [1] * (n+1)
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

    #11. Contain the most water
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) -1
        maxA = 0
        while (left < right):
            area = (right - left) * min (height[right], height[left])
            if area > maxA:
                maxA =area
            if height[right]> height[left]:
                left +=1
            else:
                right -=1
        return maxA
    #73. Partition Labels
    def partitionLabels(self, S: str) -> List[int]:
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
                currLen = dic[char][-1] - dic[char][0]
                start = dic[char][0]
                end = dic[char][-1]
            else:
                if dic[char][0] < end:
                    if dic[char][-1] > end:
                        end = dic[char][-1]
                else:
                    rst = end - start + 1
                    result.append(rst)
                    currLen = dic[char][-1] - dic[char][0]
                    start = dic[char][0]
                    end = dic[char][-1]
        return result

    #337. House Robber III
    def rob(self, root: TreeNode) -> int:
        
