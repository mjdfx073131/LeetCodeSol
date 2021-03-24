
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head == None:
            return None
        dic = {}
        cur = head
        while cur != None:
            dic[cur] = Node(cur.val, None, None)
            if (cur.next == None):
                dic[cur.next] = None
                break
            cur = cur.next

        print(dic)
        cur = head
        while cur != None:
            dic[cur].random = dic[cur.random]
            cur = cur.next

        cur = head
        while cur != None:
            dic[cur].next = dic[cur.next]
            cur = cur.next
        return dic[head]
