# 1309
def freqAlphabets(s: str) -> str:
    result = ''
    for i in range(0, len(s)):
        print(s[i])
        if (s[i] == '1' and i < len(s)-2 and s[i+2] == '#'):
            if (s[i+1] == '0'):
                result += 'j'
                i += 2
            elif s[i+1] == '1':
                result += 'k'
                i += 2
            elif s[i+1] == '2':
                result += 'l'
                i += 2
            elif s[i+1] == '3':
                result += 'm'
                i += 2
            elif s[i+1] == '4':
                result += 'n'
                i += 2
            elif s[i+1] == '5':
                result += 'o'
                i += 2
            elif s[i+1] == '6':
                result += 'p'
                i += 2
            elif s[i+1] == '7':
                result += 'q'
                i += 2
            elif s[i+1] == '8':
                result += 'r'
                i += 2
            elif s[i+1] == '9':
                result += 's'
                i += 2
        elif (s[i] == '2' and i < len(s)-2 and s[i+2] == '#'):
            if s[i+1] == '0':
                result += 't'
                i += 2
            elif s[i+1] == '1':
                result += 'u'
                i += 2
            elif s[i+1] == '2':
                result += 'v'
                i += 2
            elif s[i+1] == '3':
                result += 'w'
                i += 2
            elif s[i+1] == '4':
                result += 'x'
                i += 2
            elif s[i+1] == '5':
                result += 'y'
                i += 2
            elif s[i+1] == '6':
                result += 'z'
                i += 2
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
# 1299


def replaceElements(arr):
    for i in range(len(arr)):
        arr[i] = max(arr[(i+1):])
    return arr


def flipAndInvertImage(S):
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
                currStr+=S[idx]
                idx+=1
                recursion(currStr, idx)
            elif (S[idx].isalpha()):
                copyStr = currStr
                currStr +=S[idx]
                copyStr +=S[idx].swapcase()
                idx+=1
                recursion(currStr, idx)
                recursion(copyStr, idx)

    recursion("", 0)
    return result


print("result")
print(flipAndInvertImage("a1b2"))
# print(commonChars(["bella","label","roller"]))
