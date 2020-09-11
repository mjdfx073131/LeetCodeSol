class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # 1266
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

    # 709
    def toLowerCase(self, str: str) -> str:
        return str.lower()

    # 1252
    def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
        arr = [[0 for i in range(m)] for j in range(n)]
        for indice in indices:
            for j in range(m):
                arr[indice[0]][j] += 1
            for i in range(n):
                arr[i][indice[1]] += 1
        result = 0
        for i in range(n):
            for j in range(m):
                if arr[i][j] % 2 != 0:
                    result += 1
        return result

    # 1304
    def sumZero(self, n: int) -> List[int]:
        arr = []
        if n % 2 == 0:
            j = int(n/2)
            for i in range(int(n/2)):
                arr.append(j)
                j -= 1
            j = int(n/2)
            for i in range(int(n/2), n):
                arr.append(-j)
                j -= 1
        else:
            arr.append(0)
            j = n//2
            for i in range(1, n//2+1):
                arr.append(j)
                j -= 1
            j = n//2
            for i in range(n//2, n-1):
                arr.append(-j)
                j -= 1

        return arr

    # 1309
    def freqAlphabets(self, s: str) -> str:
        result = ''
        i = 0
        while (i < len(s)):
            if (s[i] == '1' and i < len(s)-2 and s[i+2] == '#'):
                if (s[i+1] == '0'):
                    result += 'j'
                    i += 3
                elif s[i+1] == '1':
                    result += 'k'
                    i += 3
                elif s[i+1] == '2':
                    result += 'l'
                    i += 3
                elif s[i+1] == '3':
                    result += 'm'
                    i += 3
                elif s[i+1] == '4':
                    result += 'n'
                    i += 3
                elif s[i+1] == '5':
                    result += 'o'
                    i += 3
                elif s[i+1] == '6':
                    result += 'p'
                    i += 3
                elif s[i+1] == '7':
                    result += 'q'
                    i += 3
                elif s[i+1] == '8':
                    result += 'r'
                    i += 3
                elif s[i+1] == '9':
                    result += 's'
                    i += 3
            elif (s[i] == '2' and i < len(s)-2 and s[i+2] == '#'):
                if s[i+1] == '0':
                    result += 't'
                    i += 3
                elif s[i+1] == '1':
                    result += 'u'
                    i += 3
                elif s[i+1] == '2':
                    result += 'v'
                    i += 3
                elif s[i+1] == '3':
                    result += 'w'
                    i += 3
                elif s[i+1] == '4':
                    result += 'x'
                    i += 3
                elif s[i+1] == '5':
                    result += 'y'
                    i += 3
                elif s[i+1] == '6':
                    result += 'z'
                    i += 3
            else:
                if s[i] == '1':
                    result += 'a'
                    i += 1
                elif s[i] == '2':
                    result += 'b'
                    i += 1
                elif s[i] == '3':
                    result += 'c'
                    i += 1
                elif s[i] == '4':
                    result += 'd'
                    i += 1
                elif s[i] == '5':
                    result += 'e'
                    i += 1
                elif s[i] == '6':
                    result += 'f'
                    i += 1
                elif s[i] == '7':
                    result += 'g'
                    i += 1
                elif s[i] == '8':
                    result += 'h'
                    i += 1
                elif s[i] == '9':
                    result += 'i'
                    i += 1
        return result

    # 1299
    def replaceElements(self, arr: List[int]) -> List[int]:
        # for i in range (len(arr)-1):
        #     arr[i] = max(arr[(i+1):])
        # arr[len(arr)-1] = -1

        # this is a faster solution
        out = [-1]
        greatest = 0
        for num in arr[::-1]:
            if greatest < num:
                greatest = num
            out.append(greatest)
        out.pop()
        return out[::-1]

    # 804
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        arr = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..",
               ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...",
               "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        result = []
        for word in words:
            temp = ""
            for i in range(0, len(word)):
                temp += arr[ord(word[i])-97]  # since a =97 ASCII
            result.append(temp)
        return len(set(result))

    # 832
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        result = []
        for list in A:
            list.reverse()
            temp = []
            for i in range(len(list)):
                if list[i] == 0:
                    temp.append(1)
                else:
                    temp.append(0)

            result.append(temp)
        return result

    # 905
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        result = []
        for i in range(len(A)):
            if (A[i] % 2 == 0):
                result.insert(0, A[i])
            else:
                result.append(A[i])
        return result

    # 961
    def repeatedNTimes(self, A: List[int]) -> int:
        s = set()
        for x in A:
            if x in s:
                return x
            else:
                s.add(x)

    # 657
    def judgeCircle(self, moves: str) -> bool:
        x = y = 0
        for move in moves:
            if move == 'U':
                y += 1
            elif move == 'D':
                y -= 1
            elif move == 'L':
                x -= 1
            elif move == 'R':
                x += 1
        return (x, y) == (0, 0)

    # 728
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        result = []
        for i in range(left, right+1):
            temp = i
            bit = 0
            count = 0
            while i > 0:
                digit = i % 10
                if digit == 0:
                    bit += 1
                    break
                if temp % digit == 0:
                    count += 1
                bit += 1
                i //= 10
            if (bit == count):
                result.append(temp)
        return result

    # 617
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 and not t2:
            return None
        if not t1 or not t2:
            return t2 or t1
        n = TreeNode(t1.val + t2.val)
        n.left = self.mergeTrees(t1.left, t2.left)
        n.right = self.mergeTrees(t1.right, t2.right)
        return n

    # 977
    def sortedSquares(self, A: List[int]) -> List[int]:
        return sorted([x*x for x in A])

    # 1207
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        s = sorted(arr)
        check = set()
        count = 0
        current = 0
        for i in range(len(s)):
            if (count == 0):
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
                dic[i] += 1
        return len(dic) == len(set(dic.values()))

    # 461
    def hammingDistance(self, x: int, y: int) -> int:
        return (bin(x ^ y)).count('1')

    # 852
    def peakIndexInMountainArray(self, A: List[int]) -> int:
        return A.index(max(A))

    # 942
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
                low += 1
            else:
                result.append(high)
                high -= 1
        result.append(low)
        return result

    # 700 noRecursionVersion
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if val == root.val:
                break
            elif val < root.val:
                root = root.left
            else:
                root = root.right
        return root

    # 589
    def preorder(self, root: 'Node') -> List[int]:
        result = []

        def recursion(root: 'Node'):
            if (root != None):
                result.append(root.val)
                for child in root.children:
                    recursion(child)
        recursion(root)

        return result
    # 590

    def postorder(self, root: 'Node') -> List[int]:
        result = []

        def recursion(root: 'Node'):
            if (root == None):
                return
            else:
                for child in root.children:
                    recursion(child)
            result.append(root.val)
        recursion(root)
        return result

    # 929
    def numUniqueEmails(self, emails: List[str]) -> int:
        result = []
        for email in emails:
            split = email.split('@')
            local = split[0].replace('.', '')
            if '+' in split[0]:
                local = local[:local.index('+')]
            result.append(local + '@' + split[1])
        return len(set(result))

    # 811
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
                    dic[str] = count
                else:
                    dic[str] += count
                str = subDomain[i-1] + "." + str
                i -= 1
        return ["{} {}".format(dic[k], k) for k in dic]

    # 557 thumb up 801
    def reverseWords(self, s: str) -> str:
        split = s.split(" ")
        result = ""
        count = 0
        for word in split:
            result += word[::-1]
            count += 1
            if count != len(split):
                result += " "
        return result

    def reverseWordsTwoLineVERSION(self, s: str) -> str:
        # from solution online
        words = s.split()
        # I should also write the code in one line
        return ' '.join([word[::-1] for word in words])

    # 1122
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

    # 559 dynamic programming
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        if all(child == None for child in root.children):
            return 1
        return 1 + max(self.maxDepth(child) for child in root.children)

    # 897
    def increasingBST(self, root: TreeNode) -> TreeNode:
        def dfs(cur, pre):
            if cur is None:
                return pre
            rt = dfs(cur.left, cur)
            cur.left, cur.right = None, dfs(cur.right, pre)
            return rt
        if root is None:
            return None
        return dfs(root, None)

    # 965
    def isUnivalTree(self, root: TreeNode) -> bool:
        check = True
        value = root.val

        def recursion(check: bool, root: TreeNode):
            if root.val != value:
                return check & False
            if root.left == None and root.right == None:
                return
            else:
                return recursion(check, root.left) & recursion(check, root.right)
        recursion(check, root)
        return check

    # 1160
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

    # 1047
    def removeDuplicates(self, S: str) -> str:
        result = []
        for i in S:
            if(result and result[-1] == i):
                result.pop()
            else:
                result.append(i)
        return "".join(result)

    # 1002
    def commonChars(self, A: List[str]) -> List[str]:
        result = []
        dic = {}
        for char in A[0]:
            if char not in dic:
                dic[char] = A[0].count(char)
        for i in range(1, len(A)):
            # print(A[i])
            for char in A[i]:
                if char in dic:
                    #print("Before:" + str(min(dic[char],A[i].count(char))))
                    dic[char] = min(dic[char], A[i].count(char))
                    #print("After:" +str(dic[char]))
                for chr in dic:
                    if chr not in A[i]:
                        dic[chr] = 0
            # print("**********")
        for char in dic:
            for i in range(0, dic[char]):
                result.append(char)
        return result

    # 1380

    def luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        result = []
        minL = []
        maxL = []
        for row in matrix:
            minL.append(min(row))
        tran = [[matrix[j][i]
                 for j in range(len(matrix))] for i in range(len(matrix[0]))]
        for column in tran:
            maxL.append(max(column))
        for num in minL:
            if num in maxL:
                result.append(num)
        return result

    # 1356
    def sortByBits(self, arr: List[int]) -> List[int]:
        binL = [bin(arr[i]) for i in range(len(arr))]
        dic = {}
        result = []
        for i in range(len(arr)):
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

    # 1337
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        temp = [i.count(1) for i in mat]
        dic = {}
        result = []
        for i in range(len(temp)):
            if (temp[i] not in dic):
                dic[temp[i]] = []
            dic[temp[i]].append(i)
        for i in sorted(dic.keys()):
            for num in sorted(dic[i]):
                result.append(num)
        return result[0:k]

    # 876
    def middleNode(self, head: ListNode) -> ListNode:
        counter = 0
        temp = head
        while temp is not None:
            counter += 1
            temp = temp.next

        for i in range(int(counter/2)):
            head = head.next
        return head

    # 509 memoization
    def fib(self, N: int) -> int:
        if N == 0:
            return 0
        if N == 1:
            return 1
        result = []
        result.append(0)
        result.append(1)
        for i in range(2, N+1):
            result.append(result[i-1] + result[i-2])
        return result[N]

    # 1200
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        dic = {}
        arr.sort()
        for i in range(len(arr)-1):
            diff = (arr[i+1] - arr[i]) if (arr[i+1] -
                                           arr[i]) >= 0 else -(arr[i+1] - arr[i])
            if (diff not in dic):
                dic[diff] = []
            opt = []
            opt.append(arr[i])
            opt.append(arr[i+1])
            dic[diff].append(sorted(opt))
        return dic[list(sorted(dic.keys()))[0]]

    # 1394
    def findLucky(self, arr: List[int]) -> int:
        result = []
        for num in arr:
            if num == arr.count(num):
                result.append(num)
        if len(result) == 0:
            return -1
        return max(result)

    # 344
    def reverseString(self, s: List[str]) -> None:

        s[:] = [s[-i] for i in range(1, len(s)+1)]

    # 821
    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        counter = 1
        leftR = []
        for char in S:
            if char != C:
                leftR.append(counter)
                counter += 1
            else:
                leftR.append(0)
                counter = 1
        counter = 1
        rightL = []
        for char in range(len(S)-1, -1, -1):
            if S[char] != C:
                rightL.append(counter)
                counter += 1
            else:
                rightL.append(0)
                counter = 1
        result = []
        rightL.reverse()
        record = []
        for i in range(len(rightL)):
            if rightL[i] == 0:
                record.append(i)
        for i in range(len(record)):
            if i == 0:
                result[:record[i]+1] = rightL[:record[i]+1]
            if i == len(record)-1:
                result[record[i]:] = leftR[record[i]:]
            else:
                for j in range(record[i]+1, record[i+1]+1):
                    result.append(min(leftR[j], rightL[j]))
        return result

    # 476
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        binNum = (bin(num)).replace("0b", "")
        result = ""
        for digit in binNum:
            if (digit == "1"):
                result += "0"
            else:
                result += "1"
        return int(result, 2)

    # 500
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        topRow = list("qwertyuiop")
        middleRow = list("asdfghjkl")
        bottomRow = list("zxcvbnm")
        result = []
        for word in words:
            topCheck = 0
            middleCheck = 0
            bottomCheck = 0
            for char in word:
                if char.lower() in topRow:
                    topCheck += 1
                elif char.lower() in middleRow:
                    middleCheck += 1
                else:
                    bottomCheck += 1
            if ((topCheck or middleCheck or bottomCheck) == len(word)):
                result.append(word)
        return result

    # 766
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        # bottomM = len(matrix) - 1 # for row index use
        # result = True
        # if(bottomM / 2 == 0):
        #     for n in range (1, len(matrix[0])-1):
        #         sum = 0
        #         print("current column:" + str(n))
        #         for i in range (0,n+1):
        #             print(matrix[bottomM-i][n-i])
        #             if (i >= bottomM+1):
        #                 print("reach break")
        #                 break
        #             sum += matrix[bottomM-i][n-i]
        #         print(sum)
        #         if (sum / matrix[bottomM][n] == n+1):
        #             result = result and True
        #             print("reach true")
        #         else:
        #             result = result and False
        #             print("reach false")
        #     print("finish bottom")
        #     m = 0
        #     for n in range (len(matrix[0])-2, 0,-1):
        #         sum = 0
        #         print("current column:" + str(n))
        #         for i in range (0,bottomM-n+2):
        #             #print(matrix[m+i][n+i])
        #             if (i >= bottomM+1):
        #                 print("reach break")
        #                 break
        #             sum += matrix[m+i][n+i]
        #         if (sum / (bottomM+1 - n+1) == matrix[m][n]):
        #             result = result and True
        #             print("reach true")
        #         else:
        #             result = result and False
        #             print("reach false")
        # else:
        #     for n in range (1, len(matrix[0])):
        #         sum = 0
        #         print("current column:" + str(n))
        #         for i in range (0,n+1):
        #             print(matrix[bottomM-i][n-i])
        #             sum += matrix[bottomM-i][n-i]
        #         print(sum)
        #         if (sum / matrix[bottomM][n] == n+1):
        #             result = result and True
        #             print("reach true")
        #         else:
        #             result = result and False
        #             print("reach false")
        #     print("finish bottom")
        #     m = 0
        #     for n in range (len(matrix[0])-2, 0,-1):
        #         sum = 0
        #         print("current column:" + str(n))
        #         for i in range (0,bottomM-n+2):
        #             #print(matrix[m+i][n+i])
        #             if (i >= bottomM+1):
        #                 print("reach break")
        #                 break
        #             sum += matrix[m+i][n+i]
        #         if (sum / (bottomM+1 - n+1) == matrix[m][n]):
        #             result = result and True
        #             print("reach true")
        #         else:
        #             result = result and False
        #             print("reach false")
        # return result
        rowNum, columnNum = len(matrix), len(matrix[0])
        for row in range(rowNum-1):
            for column in range(columnNum - 1):
                if matrix[row][column] != matrix[row+1][column+1]:
                    return False
        return True

    # 1446
    def maxPower(self, s):
        """
        :type s: str
        :rtype: int
        """
        if (len(s) == 0):
            return 0
        if (len(s) == 1):
            return 1
        arr = []
        result = 1
        for i in range(1, len(s)):
            if(s[i] == s[i-1]):
                result += 1
                arr.append(result)
            else:
                arr.append(result)
                result = 1
        return max(arr)

    # 463
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid:
            return 0
        m, n = len(grid), len(grid[0])
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] > 0:
                    edge = 4
                    for a, b in [[i-1, j], [i+1, j], [i, j-1], [i, j+1]]:
                        if 0 <= a < m and 0 <= b < n and grid[a][b] > 0:
                            edge -= 1
                    res += edge
        return res

    # 1078  Occurrences After Bigram
    def findOcurrences(self, text, first, second):
        """
        :type text: str
        :type first: str
        :type second: str
        :rtype: List[str]
        """
        result = []
        count = 0
        print(text.split(" "))
        for word in text.split(" "):
            print(word)
            if (count == 2):
                result.append(word)
                if (word == first):
                    count = 1
                else:
                    count = 0
                continue
            if (word == first):
                count = 1
                print("hit first")
            elif (word == second and count == 1):
                count = 2
                print("hit second")
            else:
                print("hit else")
                count = 0
            print(str(count))
        return result

    # 784

    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        result = []

        def recursion(currStr, idx):
            if (idx == len(S)-1):
                if (S[idx].isdigit()):
                    result.append(currStr+S[idx])
                    return
                if (S[idx].isalpha()):
                    result.append(currStr+S[idx])
                    result.append(currStr+S[idx].swapcase())
                    return
            else:
                if (S[idx].isdigit()):
                    currStr += S[idx]
                    idx += 1
                    recursion(currStr, idx)
                elif (S[idx].isalpha()):
                    copyStr = currStr
                    currStr += S[idx]
                    copyStr += S[idx].swapcase()
                    idx += 1
                    recursion(currStr, idx)
                    recursion(copyStr, idx)

        recursion("", 0)
        return result

    # 1470
    def shuffle(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: List[int]
        """
        result = []
        for i in range(0, n):
            result.append(nums[i])
            result.append(nums[n+i])
        return result

    # 1342
    def numberOfSteps(self, num):
        """
        :type num: int
        :rtype: int
        """
        step = 0
        while (num != 0):
            if num % 2 == 0:
                num /= 2
            else:
                num -= 1
            step += 1
        return step

    # 1365
    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = []
        for num in nums:
            count = 0
            for i in range(0, len(nums)):
                if num != nums[i] and num > nums[i]:
                    count += 1
            result.append(count)
        return result

    # 1431
    def kidsWithCandies(self, candies, extraCandies):
        """
        :type candies: List[int]
        :type extraCandies: int
        :rtype: List[bool]
        """
        result = []
        for candy in candies:
            if (candy + extraCandies) >= max(candies):
                result.append(True)
            else:
                result.append(False)
        return result

    # 1313
    def decompressRLElist(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        result = []
        for i in range(0, len(nums)-1, 2):
            for j in range(0, nums[i]):
                result.append(nums[i+1])
        return result

    # 1389
    def createTargetArray(self, nums, index):
        """
        :type nums: List[int]
        :type index: List[int]
        :rtype: List[int]
        """
        result = []
        for i in range(0, len(nums)):
            result.insert(index[i], nums[i])
        return result

    # 1450
    def busyStudent(self, startTime, endTime, queryTime):
        """
        :type startTime: List[int]
        :type endTime: List[int]
        :type queryTime: int
        :rtype: int
        """
        result = 0
        for i in range(0, len(startTime)):
            if startTime[i] <= queryTime <= endTime[i]:
                result += 1
        return result

    # 1464
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = []
        for i in range(0, len(nums)-1):
            for j in range(i+1, len(nums)):
                result.append((nums[i]-1) * (nums[j]-1))
        return max(result)

    # 226 Thumb up   invert Tree
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if (root == None):
            return root
        result = TreeNode(root.val)

        def recursion(curr, result):
            if (curr.left == None and curr.right == None):
                result = TreeNode(curr.val)
                return
            else:
                if (curr.left != None):
                    result.right = TreeNode(curr.left.val)
                    #result.val = curr.val
                    recursion(curr.left, result.right)
                # print(curr)
                if (curr.right != None):
                    # print(curr.right.val)
                    result.left = TreeNode(curr.right.val)
                    #result.val = curr.val
                    recursion(curr.right, result.left)
        recursion(root, result)
        return result

    # 682
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        totalSum = 0
        nums = []
        for op in ops:
            if op == "C":
                totalSum -= nums[len(nums)-1]
                score = nums.pop()
            elif op == "D":
                score = 2 * nums[len(nums)-1]
                nums.append(score)
                totalSum += score
            elif op == "+":
                score = nums[len(nums)-1] + nums[len(nums)-2]
                nums.append(score)
                totalSum += score
            else:
                nums.append(int(op))
                totalSum += int(op)
        return totalSum

    # 496
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        result = []
        if(len(nums2) == 0):
            return result
        dic = {}
        for i in range(len(nums2)-1):
            key = nums2[i]
            for j in range(i+1, len(nums2)):
                if (nums2[j] > key):
                    dic[key] = nums2[j]
                    break
                else:
                    dic[key] = -1
        dic[nums2[len(nums2)-1]] = -1
        for num in nums1:
            result.append(dic[num])
        return result

    # 867
    def transpose(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        tran = [[A[j][i]
                 for j in range(len(A))] for i in range(len(A[0]))]
        return tran

    #824
    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        words = S.split(" ")
        result =""
        vowels = ["a","e","i","o","u"]
        for i in range (len(words)):
            word = words[i]
            if(word[0].lower() in vowels):
                word += "ma"
                word += ("a"*(i+1))
                if (i != len(words)-1):
                    word += " "
                result +=word
            else:
                newWord = word[1: ] + word[0] + "ma" + "a"*(i+1)
                if (i != len(words)-1):
                    newWord += " "
                result +=newWord
        return result

    #1480
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """ 
        result = []
        for num in nums:
            if len(result) == 0:
                result.append(num)
            else:
                result.append(result[len(result)-1]+ num)
        return result

    #1512
    def numIdenticalPairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = 0
        for i in range (len(nums)-1):
            for j in range(i+1, len(nums)):
                if (nums[i] == nums[j]):
                    result +=1
        return result
        
    #1351
    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        result = 0
        for row in grid:
            for i in row:
                if i < 0:
                    result +=1
        return result

    #1528
    def restoreString(self, s, indices):
        """
        :type s: str
        :type indices: List[int]
        :rtype: str
        """
        result = ""
        sortedArr = [None for i in range(len(s))]
        for i in range(len(indices)):
            sortedArr[indices[i]] = s[i]
        for str in sortedArr:
            result += str
        return result
    #884
    def uncommonFromSentences(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: List[str]
        """
        dic = {}
        for word in A.split(" "):
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1
        for word in B.split(" "):
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1
        result =[]
        for word in dic:
            if dic[word] == 1:
                result.append(word)
        return result
    

    #1436

    def destCity(self, paths):
        """
        :type paths: List[List[str]]
        :rtype: str
        """
        start = []
        end = []
        for path in paths:
            start.append(path[0])
            end.append(path[1])
        path = paths[0]
        while True:
            if path[1] in start:
                for search in paths:
                    if (search[0] == path[1]):
                        path = search
            else:
                return path[1]
    
    #1455
    def isPrefixOfWord(self, sentence, searchWord):
        """
        :type sentence: str
        :type searchWord: str
        :rtype: int
        """
        arr = sentence.split(" ")
        prefixLength = len(searchWord)
        for i in range(len(arr)):
            if len(arr[i]) >= prefixLength and arr[i][: prefixLength] == searchWord:
                return (i+1)
        return -1
    
    #1550
    def threeConsecutiveOdds(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        i = 0
        while i < len(arr):
            if arr[i] % 2 != 0 and i+2 < len(arr):
                if arr[i+1] % 2 != 0:
                    if arr[i+2] % 2 != 0:
                        return True
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return False

    #1572
    def diagonalSum(self, mat: List[List[int]]) -> int:
        sum = 0
        for i, row in enumerate(mat):
            if i == len(row)-i-1:
                sum += (row[i])
            else:
                sum += (row[i] + row[len(row)-i-1])
        return sum

    #1413           
    def minStartValue(self, nums: List[int]) -> int:
        stepSum = 1
        maxSum = 0
        for num in nums:
            stepSum += (1 + (-num))
            if stepSum > maxSum:
                maxSum = stepSum
        if maxSum == 0:
            return 1
        else:
            return maxSum

    #1486. XOR Operation in an Array
    def xorOperation(self, n: int, start: int) -> int:
        step = 1
        result = start
        while (step < n):
            result = result ^ (start+2*step)
            step += 1
        return result

    #561. Array Partition I
    def arrayPairSum(self, nums: List[int]) -> int:
        result = 0
        sortedArr = sorted(nums)
        for i in range(0, len(sortedArr), 2):
            result += sortedArr[i]
        return result

    #1460. Make Two Arrays Equal by Reversing Sub-arrays
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return sorted(target) == sorted(arr)

    #1475. Final Prices With a Special Discount in a Shop
    def finalPrices(self, prices: List[int]) -> List[int]:
        result = []
        for i in range(len(prices) - 1):
            price = prices[i]
            for j in range(i+1, len(prices)):
                if prices[j] <= prices[i]:
                    result.append(price - prices[j])
                    break
                if j == len(prices) - 1:
                    result.append(price)
            if (i == len(prices) - 2):
                result.append(prices[-1])
        return result
