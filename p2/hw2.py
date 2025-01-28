import math
import argparse
from typing import List
import sys

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename: str) -> dict:
    shred_list = [0] * 26
    # count the occurance of every character
    with open (filename, encoding="utf-8") as f:
        content = f.read()
        content = content.upper()
        for char in content:
            if char >= 'A' and char <= 'Z':
                shred_list[ord(char)-ord('A')] += 1
    # print Q1
    print("Q1")
    for idx, cnt in enumerate(shred_list):
        print(f"{chr(ord('A') + idx)} {cnt}")
    return shred_list

def compute_fy_preprocess(shred_list: List, english_char_prob: float, spanish_char_prob: float) -> None:
    # X1 log e1
    res1 = shred_list[0] * math.log(english_char_prob[0])
    # X1 log s1
    res2 = shred_list[0] * math.log(spanish_char_prob[0])
    print("Q2")
    print(f"{res1:.4f}")
    print(f"{res2:.4f}")

def compute_Fy(shred_list: List, english_prior: float, spanish_prior: float, english_char_prob: List, spanish_char_prob: List) -> set:
    Fy_eng = math.log(english_prior)
    Fy_spa = math.log(spanish_prior)
    for idx in range(26):
        Fy_eng += shred_list[idx] * math.log(english_char_prob[idx])
        Fy_spa += shred_list[idx] * math.log(spanish_char_prob[idx])
    print("Q3")
    print(f"{Fy_eng:.4f}")
    print(f"{Fy_spa:.4f}")
    return Fy_eng, Fy_spa

def compute_y_cond_prob_given_x(Fy_eng: float, Fy_spa: float) -> None:
    if Fy_spa - Fy_eng >= 100:
        Py_eng_given_X = 0
    elif Fy_spa - Fy_eng <= -100:
        Py_eng_given_X = 1
    else:
        Py_eng_given_X = 1 / (1 + math.e ** (Fy_spa - Fy_eng))
    print("Q4")
    print(f"{Py_eng_given_X:.4f}")

def main():
    parser = argparse.ArgumentParser(description="arguments of hw2.py")
    parser.add_argument("letter_file", help="The input path of letter file, ex: samples/letter0.txt", default="letter.txt")
    parser.add_argument("english_prior", help="The input of english prior probability, ex: 0.6", default=0.6)
    parser.add_argument("spanish_prior", help="The input of english prior probability, ex: 0.4", default=0.4)
    args = parser.parse_args()
    shred_list = shred(args.letter_file)
    (e,s) = get_parameter_vectors()
    compute_fy_preprocess(shred_list, e, s)
    Fy_eng, Fy_spa = compute_Fy(shred_list, float(args.english_prior), float(args.spanish_prior), e, s)
    compute_y_cond_prob_given_x(Fy_eng, Fy_spa)

if __name__ == "__main__":
    main()
