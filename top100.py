
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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
