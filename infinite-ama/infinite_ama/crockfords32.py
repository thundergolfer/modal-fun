# -*- coding: utf-8 -*-
#
# This file is part of base32-lib
# Copyright (C) 2019 CERN.
# Copyright (C) 2019 Northwestern University,
#                    Galter Health Sciences Library & Learning Center.

# base32-lib is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.

"""Generate, encode and decode random base32 identifiers.

This encoder/decoder:
- uses Douglas Crockford Base32 encoding: https://www.crockford.com/base32.html
- allows for ISO 7064 checksum
- encodes the checksum using only characters in the base32 set
  (only digits in fact)
- produces string that are URI-friendly (no '=' or '/' for instance)

This is based on:
- https://github.com/datacite/base32-url
- https://github.com/jbittel/base32-crockford
"""

import random
import string

import six

# NO i, l, o or u
ENCODING_CHARS = "0123456789abcdefghjkmnpqrstvwxyz"
DECODING_CHARS = {c: i for i, c in enumerate(ENCODING_CHARS)}


def encode(number, split_every=0, min_length=0, checksum=False):
    """Encodes `number` to URI-friendly Douglas Crockford base32 string.

    :param number: number to encode
    :param split_every: if provided, insert '-' every `split_every` characters
                        going from left to right
    :param checksum: append modulo 97-10 (ISO 7064) checksum to string
    :returns: A random Douglas Crockford base32 encoded string composed only
              of valid URI characters.
    """
    assert isinstance(number, six.integer_types)

    if number < 0:
        raise ValueError("Invalid 'number'. Must be >= 0.")

    if split_every < 0:
        raise ValueError("Invalid 'split_every'. Must be >= 0.")

    encoded = ""
    original_number = number
    if number == 0:
        encoded = "0"
    else:
        while number > 0:
            remainder = number % 32
            number //= 32  # quotient of integer division
            encoded = ENCODING_CHARS[remainder] + encoded

    if checksum:
        # NOTE: 100 * original_number is used because datacite also uses it
        computed_checksum = 97 - ((100 * original_number) % 97) + 1
        encoded_checksum = "{:02d}".format(computed_checksum)
        encoded += encoded_checksum

    if min_length > 0:
        # 0-pad beginning of string to obtain minimum desired length
        encoded = encoded.zfill(min_length)

    if split_every > 0:
        splits = [
            encoded[i : i + split_every] for i in range(0, len(encoded), split_every)
        ]
        encoded = "-".join(splits)

    return encoded


def generate(length=8, split_every=0, checksum=False):
    """Generate random base32 string.

    :param length: non-hyphen identifier length *including* checksum
    :param split_every: hyphenates every that many characters
    :param checksum: computes and appends ISO-7064 checksum
    :returns: identifier as a string
    """
    if checksum and length < 3:
        raise ValueError("Invalid 'length'. Must be >= 3 if checksum enabled.")

    generator = random.SystemRandom()
    length_no_checksum = length - 2 if checksum else length
    # takes at most length*5 bits to express, but could take less
    number = generator.getrandbits(length_no_checksum * 5)
    return encode(
        number,
        split_every=split_every,
        min_length=length,  # ensures desired length (*including* checksum)
        checksum=checksum,
    )


def normalize(encoded):
    """Returns normalized encoded string.

    - string is lowercased
    - '-' are removed
    - I,i,l,L decodes to the digit 1
    - O,o decodes to the digit 0

    :param encoded: string to decode
    :returns: normalized string.
    """
    table = (
        "".maketrans("IiLlOo", "111100")
        if six.PY3
        else string.maketrans("IiLlOo", "111100")
    )
    encoded = encoded.replace("-", "").translate(table).lower()

    if not all([c in ENCODING_CHARS for c in encoded]):
        raise ValueError("'encoded' contains undecodable characters.")

    return encoded


def decode(encoded, checksum=False):
    """Decodes `encoded` string (via above) to a number.

    The string is normalized before decoding.

    If `checksum` is enabled, raises a ValueError on checksum error.

    :param encoded: string to decode
    :param checksum: extract checksum and validate
    :returns: original number.
    """
    if checksum:
        encoded_checksum = encoded[-2:]
        encoded = encoded[:-2]

    encoded = normalize(encoded)

    number = 0
    for i, c in enumerate(reversed(encoded)):
        number += DECODING_CHARS[c] * (32**i)

    if checksum:
        verification_checksum = int(encoded_checksum, 10)
        # NOTE: 100 * number is used because datacite also uses it
        computed_checksum = 97 - ((100 * number) % 97) + 1

        if verification_checksum != computed_checksum:
            raise ValueError("Invalid checksum.")

    return number
