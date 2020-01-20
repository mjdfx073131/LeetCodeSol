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
        return str.lower();

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
