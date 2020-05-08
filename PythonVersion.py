class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    #1266
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        # for i in range(0, len(points)):
        #     for j in range (1, len(points)):
        #         if(points[i][j])
        result = 0 
        firstP = points[0]
        for secondP in points[1:]:
            result += max(abs(firstP[0]-secondP[0]), abs(firstP[1]-secondP[1]))
            firstP = secondP
        return result
    
    #709
    def toLowerCase(self, str: str) -> str:
        return str.lower()

    #1252
    def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
        arr = [[0 for i in range(m)] for j in range(n)]
        for indice in indices:
            for j in range (m):
                arr[indice[0]][j] +=1
            for i in range (n):
                arr[i][indice[1]] +=1
        result = 0
        for i in range (n):
            for j in range (m):
                if arr[i][j] % 2 != 0:
                    result+=1
        return result

    #1304
    def sumZero(self, n: int) -> List[int]:
        arr = []
        if n % 2 == 0:
            j = int(n/2)
            for i in range (int(n/2)):
                arr.append(j)
                j -= 1
            j = int(n/2)
            for i in range (int(n/2),n):
                arr.append(-j)
                j -= 1
        else:
            arr.append(0)
            j = n//2
            for i in range (1,n//2+1):
                arr.append(j)
                j -=1
            j = n//2
            for i in range(n//2, n-1):
                arr.append(-j)
                j -= 1

        return arr
    
    #1309
    def freqAlphabets(self, s: str) -> str:
        result = ''
        i = 0
        while (i < len(s)):
            if (s[i] == '1' and i < len(s)-2 and s[i+2] == '#'):
                if (s[i+1] == '0'):
                    result += 'j'
                    i+=3
                elif s[i+1] == '1':
                    result += 'k'
                    i+=3
                elif s[i+1] == '2':
                    result += 'l'
                    i+=3
                elif s[i+1] == '3':
                    result += 'm'
                    i+=3
                elif s[i+1] == '4':
                    result += 'n'
                    i+=3
                elif s[i+1] == '5':
                    result += 'o'
                    i+=3
                elif s[i+1] == '6':
                    result += 'p'
                    i+=3
                elif s[i+1] == '7':
                    result += 'q'
                    i+=3
                elif s[i+1] == '8':
                    result += 'r'
                    i+=3
                elif s[i+1] == '9':
                    result += 's'
                    i+=3
            elif (s[i] == '2' and i <len(s)-2 and s[i+2] == '#'):
                if s[i+1] == '0':
                    result += 't'
                    i+=3
                elif s[i+1] == '1':
                    result += 'u'
                    i+=3
                elif s[i+1] == '2':
                    result += 'v'
                    i+=3
                elif s[i+1] == '3':
                    result += 'w'
                    i+=3
                elif s[i+1] == '4':
                    result += 'x'
                    i+=3
                elif s[i+1] == '5':
                    result += 'y'
                    i+=3
                elif s[i+1] == '6':
                    result += 'z'
                    i+=3
            else:
                if s[i] == '1':
                    result += 'a'
                    i+=1
                elif s[i] == '2':
                    result += 'b'
                    i+=1
                elif s[i] == '3':
                    result += 'c'
                    i+=1
                elif s[i] == '4':
                    result += 'd'
                    i+=1
                elif s[i] == '5':
                    result += 'e'
                    i+=1
                elif s[i] == '6':
                    result += 'f'
                    i+=1
                elif s[i] == '7':
                    result += 'g'
                    i+=1
                elif s[i] == '8':
                    result += 'h'
                    i+=1
                elif s[i] == '9':
                    result += 'i'
                    i+=1
        return result

    #1299
    def replaceElements(self, arr: List[int]) -> List[int]:
        # for i in range (len(arr)-1):
        #     arr[i] = max(arr[(i+1):])
        # arr[len(arr)-1] = -1

        #this is a faster solution 
        out = [-1]
        greatest = 0
        for num in arr[::-1]:
            if greatest < num:
                greatest = num
            out.append(greatest)
        out.pop()
        return out[::-1]
            
    #804
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        arr = [".-","-...","-.-.","-..",".","..-.","--.","....","..",
        ".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...",
        "-","..-","...-",".--","-..-","-.--","--.."]
        result= []
        for word in words:
            temp = ""
            for i in range (0, len(word)):
                temp += arr[ord(word[i])-97] # since a =97 ASCII
            result.append(temp)
        return len(set(result))
    
    #832
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        result = []
        for list in A:
            list.reverse()
            temp = []
            for i in range (len(list)):
                if list[i] == 0:
                    temp.append(1)
                else:
                    temp.append(0)

            result.append(temp)
        return result
    
    #905
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        result =[]
        for i in range (len(A)):
            if (A[i] % 2 == 0 ):
                result.insert(0,A[i])
            else:
                result.append(A[i])
        return result
    
    #961
    def repeatedNTimes(self, A: List[int]) -> int:
        s = set()
        for x in A:
            if x in s:
                return x
            else:
                s.add(x)

    #657
    def judgeCircle(self, moves: str) -> bool:
        x=y = 0
        for move in moves:
            if move == 'U':
                y+=1
            elif move == 'D':
                y-=1
            elif move == 'L':
                x-=1
            elif move == 'R':
                x+=1
        return (x,y) == (0,0)
  
    #728
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        result =[]
        for i in range (left, right+1):
            temp = i
            bit =0 
            count = 0
            while i >0:
                digit = i % 10
                if digit == 0:
                    bit+=1
                    break
                if temp % digit== 0:
                    count+=1
                bit+=1
                i//=10
            if (bit == count):
                result.append(temp)
        return result
                    
    #617
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 and not t2: return None
        if not t1 or not t2: return t2 or t1
        n = TreeNode(t1.val+ t2.val)
        n.left = self.mergeTrees(t1.left, t2.left)
        n.right = self.mergeTrees(t1.right, t2.right)
        return n
    
    #977
    def sortedSquares(self, A: List[int]) -> List[int]:
        return sorted([x*x for x in A])
        
    #1207
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        s = sorted(arr)
        check = set()
        count = 0
        current = 0
        for i in range (len(s)):
            if (count ==0):
                count += 1
                current = s[i]
            elif s[i] == current:
                count += 1
            elif s[i] != current:
                if count in check:
                    return False
                else:
                    check.add(count)
                    count = 1
                    current = s[i]
            if (i == len(s)-1):
                if count in check:
                    return False
        return True
    def uniqueOccurrencesAnotherFasterWay(self, arr: List[int]) -> bool:
        dic = {}
        for i in arr:
            if i not in dic:
                dic[i] = 1
            else:
                dic[i]+=1
        return len(dic) == len(set(dic.values()))        
                  
    #461
    def hammingDistance(self, x: int, y: int) -> int:
        return (bin(x^y)).count('1')
    
    #852 
    def peakIndexInMountainArray(self, A: List[int]) -> int:
        return A.index(max(A))
    
    #942
    def diStringMatch(self, S: str) -> List[int]:
        # result = [0]
        # for x in S:
        #     lastDigit = result[len(result)-1]
        #     if (x == 'I'):
        #         result.append(lastDigit +1)
        #      elif (x == 'D'):
        #          result.append(lastDigit - 1)
        # return result
        low = 0
        high = len(S)
        result = []
        for x in S:
            if x == 'I':
                result.append(low)
                low+=1
            else:
                result.append(high)
                high -=1
        result.append(low)
        return result

    #700 noRecursionVersion
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if val == root.val:
                break
            elif val < root.val:
                root = root.left
            else:
                root = root.right
        return root

    #589
    def preorder(self, root: 'Node') -> List[int]:
        result = []
        def recursion (root: 'Node'):
            if (root != None):
                result.append(root.val)
                for child in root.children:
                    recursion(child)
        recursion(root)
            
        return result
    #590
    def postorder(self, root: 'Node') -> List[int]:
        result = []
        def recursion (root: 'Node'):
            if (root == None):
                return 
            else:
                for child in root.children:
                    recursion(child)
            result.append(root.val)
        recursion(root)
        return result
    
    #929
    def numUniqueEmails(self, emails: List[str]) -> int:
        result = []
        for email in emails:
            split = email.split('@')
            local = split[0].replace('.', '')
            if '+' in split[0]:
                local = local[:local.index('+')]
            result.append(local + '@' + split[1])
        return len(set(result))

    #811
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        dic = {}
        for domain in cpdomains:
            split = domain.split(" ")
            count = int(split[0])
            subDomain = split[1].split(".")
            i = len(subDomain) - 1
            str = subDomain[i]
            while i >= 0:
                if str not in dic:
                    dic[str]= count
                else:
                    dic[str]+= count
                str = subDomain[i-1] + "." + str
                i-=1
        return ["{} {}".format(dic[k],k) for k in dic]


    #557 thumb up 801
    def reverseWords(self, s: str) -> str:
        split = s.split(" ")
        result = ""
        count = 0
        for word in split:
            result += word[::-1]
            count +=1
            if count != len(split):
                result += " "
        return result 
    def reverseWordsTwoLineVERSION(self, s: str) -> str:
        # from solution online
        words = s.split() 
        return ' '.join([word[::-1] for word in words]) #  I should also write the code in one line

    #1122
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        dic = {}
        notIn = []
        result = []
        for i in arr2:
            dic[i] = []
        for i in arr1:
            if i in arr2:
                dic[i].append(i)
            else:
                notIn.append(i)
        
        for key in dic:
            result += dic[key]
        notIn.sort()
        result += notIn
        return result
    
    #559 dynamic programming
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        if all(child == None for child in root.children):
            return 1
        return 1 + max(self.maxDepth(child) for child in root.children)

    
    #897
    def increasingBST(self, root: TreeNode) -> TreeNode:
        def dfs(cur, pre):
            if cur is None: return pre
            rt = dfs(cur.left, cur)
            cur.left, cur.right = None, dfs(cur.right, pre)
            return rt
        if root is None: return None
        return dfs(root, None)

    #965
    def isUnivalTree(self, root: TreeNode) -> bool:
        check = True
        value = root.val
        def recursion (check: bool, root: TreeNode):
            if root.val != value:
                return check & False
            if root.left == None and root.right == None:
                return
            else:
                return recursion(check, root.left) & recursion(check, root.right)
        recursion(check, root)
        return check
    #1160
    def countCharacters(self, words: List[str], chars: str) -> int:
        result = 0
        for word in words:
            length = len(word)
            for ch in word:
                if word.count(ch) <= chars.count(ch):
                    length -= 1
                else:
                    break
                if length == 0:
                    result += len(word)
        return result

    #1047
    def removeDuplicates(self, S: str) -> str:
        result = []
        for i in S:
            if(result and result[-1]==i):
                result.pop()
            else:
                result.append(i)
        return "".join(result)

    #1002
    def commonChars(self, A: List[str]) -> List[str]:
        result = []
        dic = {}
        for char in A[0]:
            if char not in dic:
                dic[char] = A[0].count(char)
        for i in range(1,len(A)):
            #print(A[i])
            for char in A[i]:
                if char in dic:
                    #print("Before:" + str(min(dic[char],A[i].count(char))))
                    dic[char] = min(dic[char],A[i].count(char))
                    #print("After:" +str(dic[char]))
                for chr in dic:
                    if chr not in A[i]:
                        dic[chr] = 0
            #print("**********")
        for char in dic:
            for i in range (0,dic[char]):
                result.append(char)
        return result
                    

    #1380
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        result = []
        minL = []
        maxL = []
        for row in matrix:
            minL.append(min(row))
        tran = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
        for column in tran:
            maxL.append(max(column))
        for num in minL:
            if num in maxL:
                result.append(num)
        return result

    #1356
    def sortByBits(self, arr: List[int]) -> List[int]:
        binL = [bin(arr[i]) for i in range (len(arr))]
        dic={}
        result = []
        for i in range (len(arr)):
            numOne = binL[i].count("1")
            if (numOne not in dic):
                dic[numOne] = []
                dic[numOne].append(arr[i])
            else:
                dic[numOne].append(arr[i])
        for i in sorted(dic.keys()):
            for num in sorted(dic[i]):
                result.append(num)
        return result

    #1337
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        temp = [i.count(1) for i in mat]
        dic ={}
        result =[]
        for i in range (len(temp)):
            if (temp[i] not in dic):
                dic[temp[i]] = []
            dic[temp[i]].append(i)   
        for i in sorted(dic.keys()):
            for num in sorted(dic[i]):
                result.append(num)
        return result[0:k]           

    #876
    def middleNode(self, head: ListNode) -> ListNode:
        counter = 0
        temp = head
        while temp is not None:
            counter +=1
            temp = temp.next
        
        for i in range (int(counter/2)):
            head = head.next
        return head
    
    #509 memoization
    def fib(self, N: int) -> int:
        if N == 0:
            return 0
        if N == 1:
            return 1
        result = []
        result.append(0)
        result.append(1)
        for i in range (2,N+1):
            result.append(result[i-1]+ result[i-2])
        return result[N]
    
    #1200
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        dic = {}
        arr.sort()
        for i in range (len(arr)-1):
            diff = (arr[i+1] - arr[i]) if (arr[i+1] - arr[i]) >= 0 else -(arr[i+1] - arr[i])
            if (diff not in dic):
                dic[diff] = []
            opt =[]
            opt.append(arr[i])
            opt.append(arr[i+1])
            dic[diff].append(sorted(opt))
        return dic[list(sorted(dic.keys()))[0]]

    #1394
    def findLucky(self, arr: List[int]) -> int:
        result = []
        for num in arr:
            if num == arr.count(num):
                result.append(num)
        if len(result) == 0:
            return -1
        return max(result)

    #344
    def reverseString(self, s: List[str]) -> None:

        s[:] = [s[-i] for i in range(1,len(s)+1)]
    
    #821
    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        counter = 1
        leftR =[]
        for char in S:
            if char != C:
                leftR.append(counter)
                counter+=1
            else:
                leftR.append(0)
                counter = 1
        counter = 1
        rightL = []
        for char in range(len(S)-1,-1,-1):
            if S[char] != C:
                rightL.append(counter)
                counter+=1
            else:
                rightL.append(0)
                counter = 1
        result = []
        rightL.reverse()
        record = []
        for i in range (len(rightL)):
            if rightL[i] == 0:
                record.append(i)
        for i in range(len(record)):
            if i == 0:
                result[:record[i]+1] = rightL[:record[i]+1]
            if i == len(record)-1:
                result[record[i]:] = leftR[record[i]:]
            else:
                for j in range (record[i]+1,record[i+1]+1):
                    result.append(min(leftR[j],rightL[j]))
        return result
    
    #1022
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        binNum = []   
        def findPath(self, root):
            if root is None:
                return
            else 
    
    #476
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        binNum = (bin(num)).replace("0b","")
        result = ""
        for digit in binNum:
            if (digit == "1"):
                result += "0"
            else:
                result += "1"
        return int(result,2)