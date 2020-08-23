
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


            

    



