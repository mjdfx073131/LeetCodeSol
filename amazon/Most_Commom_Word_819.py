from typing import List

import re

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        word_count = {}
        trimmed_word = re.split("[\s!?.;',]", paragraph.lower())
        print(trimmed_word)
        for word in trimmed_word:
            if word not in banned and word.isalnum():
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        result = ''
        maxNum = 0
        for word in word_count:
            if word_count[word] > maxNum:
                maxNum = word_count[word]
                result = word
        return result


    """ Intuition split the paragraph using regex """

