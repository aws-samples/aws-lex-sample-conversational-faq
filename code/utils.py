#!/usr/bin/env python3

import os
import re
import string
import sys as _sys


def split_into_sentences(text):
    # Define the regular expression pattern for sentence splitting
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'

    # Split the text into sentences using the regular expression
    sentences = re.split(pattern, text)

    return sentences


def str2bool(v):
    """
    String to boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return False
        # raise argparse.ArgumentTypeError("Boolean value expected.")


def normalize_text(text):
    """
    Normalize text for exact match.
    """
    switch_list = [(" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"), (" ' ", "'")]
    new_text = text.replace("\n", " ").strip()
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace("  ", " ")
    tokens = new_text.split(" ")
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ("i", "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        # elif tokens[i] in "?.!" and i < len(tokens) - 1:
        #     tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = " ".join(tokens)
    new_text = " " + new_text + " "
    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])
    new_text = new_text.strip()
    new_text = new_text.replace("  ", " ")
    return new_text


def normalize_reply(text: str) -> str:
    """
    Normalize response text.
    """
    if text[:4].lower() in ["yes,", "yes ", "yes."]:
        text = text[4:].strip()
    elif text[:3].lower() in ["no,", "no ", "no."]:
        text = text[3:].strip()
    if text[0] in string.punctuation and text[0] not in "({[":
        text = text[1:].strip()
    new_text = normalize_text(text)
    if new_text and new_text[-1] not in "!.?)\"'":
        new_text += "."
    return new_text


def uppercase(string: str) -> str:
    """
    Make the first character of the string uppercase, if the string is non-empty.
    """
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]