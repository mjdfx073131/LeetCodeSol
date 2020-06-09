
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
            return 1 + max (self.maxDepthAnotherVersion(root.left), self.maxDepthAnotherVersion(root.right))

    #136 Single Number should have a runtime complexity without using extra memory
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
        copy =[]
        for num in nums:
            if num in copy:
                copy.remove(num)
            else:
                copy.append(num)
        return copy[0]
    def singleNumber80ms(self, nums):
        dic= {}
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                dic.pop(num)
        return dic.keys()[0]
        
    #206
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