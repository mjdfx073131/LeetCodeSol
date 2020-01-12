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
            
        
                  
