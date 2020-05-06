#1309
def freqAlphabets(s: str) -> str:
    result = ''
    for i in range (0,len(s)):
        print(s[i])
        if (s[i] == '1' and i < len(s)-2 and s[i+2] == '#'):
            if (s[i+1] == '0'):
                result += 'j'
                i+=2
            elif s[i+1] == '1':
                result += 'k'
                i+=2
            elif s[i+1] == '2':
                result += 'l'
                i+=2
            elif s[i+1] == '3':
                result += 'm'
                i+=2
            elif s[i+1] == '4':
                result += 'n'
                i+=2
            elif s[i+1] == '5':
                result += 'o'
                i+=2
            elif s[i+1] == '6':
                result += 'p'
                i+=2
            elif s[i+1] == '7':
                result += 'q'
                i+=2
            elif s[i+1] == '8':
                result += 'r'
                i+=2
            elif s[i+1] == '9':
                result += 's'
                i+=2
        elif (s[i] == '2' and i <len(s)-2 and s[i+2] == '#'):
            if s[i+1] == '0':
                result += 't'
                i+=2
            elif s[i+1] == '1':
                result += 'u'
                i+=2
            elif s[i+1] == '2':
                result += 'v'
                i+=2
            elif s[i+1] == '3':
                result += 'w'
                i+=2
            elif s[i+1] == '4':
                result += 'x'
                i+=2
            elif s[i+1] == '5':
                result += 'y'
                i+=2
            elif s[i+1] == '6':
                result += 'z'
                i+=2
        else:
            if s[i] == '1':
                result += 'a'
            elif s[i] == '2':
                result += 'b'
            elif s[i] == '3':
                result += 'c'
            elif s[i] == '4':
                result += 'd'
            elif s[i] == '5':
                result += 'e'
            elif s[i] == '6':
                result += 'f'
            elif s[i] == '7':
                result += 'g'
            elif s[i] == '8':
                result += 'h'
            elif s[i] == '9':
                result += 'i'
    return result
#1299
def replaceElements(arr):
        for i in range (len(arr)):
            arr[i] = max(arr[(i+1):])
        return arr
    
def flipAndInvertImage(A):
        # result = []
        # for list in A:
        #     list.reverse()
        #     for i in range (len(list)):
        #         if list[i] == 0:
        #             list[i] == 1
        #         else:
        #             list[i] == 0

        #     result.append(list)
        # return result
    result = []
    dic = {}
    for char in A[0]:
        if char not in dic:
            dic[char] = A[0].count(char)
    for i in range(1,len(A)):
        print(A[i])
        for char in A[i]:
            if char in dic:
                print("Before:" + str(min(dic[char],A[i].count(char))))
                dic[char] = min(dic[char],A[i].count(char))
                print("After:" +str(dic[char]))
            for chr in dic:
                if chr not in A[i]:
                    dic[chr] = 0
        print("**********")
    for char in dic:
        for i in range (0,dic[char]):
            result.append(char)
    return result

#1002
# def commonChars(self, A: List[str]):
#     result = []
#     dic = {}
#     for char in A[0]:
#         if char not in dic:
#             dic[char] = A[0].count(char)
#     for i in range(1,len(A)-1):
#         print(A[i])
#         for char in A[i]:
#             if char in dic:
#                 print("Before:"+ min(dic[char],A[i].count(char)))
#                 dic[char] = min(dic[char],A[i].count(char))
#                 print("After:"+ dic[char])
#     for char in dic:
#         for i in range (0,dic[char]):
#             result.append(char)
#     return result
print(flipAndInvertImage(["cool","lock","cook"]))
# print(commonChars(["bella","label","roller"]))
