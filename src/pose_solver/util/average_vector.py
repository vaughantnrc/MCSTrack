# This is a very simple function for averaging translations
# when it is not desired to use numpy (for whatever reason)
from typing import List


def average_vector(
    translations: List[List[float]]
) -> List[float]:
    sum_translations: List[float] = [0.0, 0.0, 0.0]
    for translation in translations:
        for i in range(0, 3):
            sum_translations[i] += translation[i]
    translation_count = len(translations)
    return [
        sum_translations[0] / translation_count,
        sum_translations[1] / translation_count,
        sum_translations[2] / translation_count]
