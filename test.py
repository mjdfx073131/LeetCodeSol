class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



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


def flipAndInvertImage(n):
    if (n <= 0):
        return 1
    return flipAndInvertImage(n-1) + flipAndInvertImage(n-2)
A = TreeNode(5, TreeNode(4, TreeNode(3, None,None), TreeNode(2,None,None)), TreeNode(1, None,None))
B = TreeNode(5, TreeNode(3, None, None), TreeNode(2, None,None))
C = TreeNode(5, TreeNode(4,None, TreeNode(3, None,None)), None)
print("result")
print(flipAndInvertImage(2))
# print(commonChars(["bella","label","roller"]))
