from spellchecker import SpellChecker

spell = SpellChecker(distance=1)
spell.word_frequency.load_words(['riki'])

def calculate_hamming_distance(str1, str2):
    distance = 0
    if len(str1) != len(str2):
        return -1
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            distance += 1
    return distance


def correct_text(line):
    word_list = line.split(" ")
    print("Unknown words:", spell.unknown(word_list))
    for i in range(len(word_list)):
        # candidate = spell.candidates(word_list[i])
        # print("Candidates for ", word_list[i], ":", candidate)
        # correction = spell.correction(word_list[i])
        hamming_candidates = [word for word in spell.candidates(word_list[i]) if len(word) == len(word_list[i])]
        probability = spell.word_probability(hamming_candidates[0])
        correction = hamming_candidates[0]
        for word in hamming_candidates:
            if spell.word_probability(word) > probability:
                probability = spell.word_probability(word)
                correction = word
        word_list[i] = correction

        # TO-DO: only make a change with identical hamming-distance
        # if len(word_list[i]) == len(correction):
        #     word_list[i] = correction
    return " ".join(word_list)


print("Corrected Version: ", correct_text(input("Enter sentence: ")))
