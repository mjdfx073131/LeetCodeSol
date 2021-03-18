# Definition for singly-linked list.
from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbersHelper(self, big: ListNode, small: ListNode) -> ListNode:
        digitSum = 0
        carrier = 0
        result = big
        while(big != None):
            if (small != None):
                digitSum = big.val + small.val + carrier
            else:
                digitSum = big.val + carrier
            if (digitSum >= 10):
                carrier = 1
                if big.next != None:
                    big.val = digitSum % 10
                else:
                    big.val = digitSum % 10
                    big.next = ListNode(1)
                    return result
            else:
                carrier = 0
                big.val = digitSum
            if(big.next != None):
                big = big.next
            else:
                return result
            if(small != None):
                small = small.next
            else:
                small = None

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

        l1_len = 0
        l2_len = 0
        l1_copy = l1
        l2_copy = l2

        while (l1_copy != None or l2_copy != None):
            if(l1_copy != None):
                l1_len += 1
                l1_copy = l1_copy.next
            if (l2_copy != None):
                l2_len += 1
                l2_copy = l2_copy.next

        if (l1_len >= l2_len):
            return self.addTwoNumbers(l1, l2)
        else:
            return self.addTwoNumbersHelper(l2,l1)
            

