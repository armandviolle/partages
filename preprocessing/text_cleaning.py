# from : https://github.com/getalp/Flaubert/blob/master/tools/clean_text.py

import logging
import re
import unicodedata

import six

logger = logging.getLogger(__name__)


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.

    Parameters
    ----------
    text : str
        The input text to convert.

    Returns
    -------
    str
        The converted Unicode text.
    """

    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding="utf-8", errors="strict"):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def normalize_unicode(text):
    """
    Normalize unicode underlying representation

    Parameters
    ----------
    text : str
        The input text to normalize.

    Returns
    -------
    str
        The normalized text.
    """
    text = unicodedata.normalize("NFC", text)

    return text


def read_codepage(text, codepage="cp863"):
    """
    Keep only characters belonging to the character set of a language

    Parameters
    ----------
    text:
        input text
    code page:
        for each language (Example: Code page 863 is the code page used to write French Canadian language, see https://www.ascii-codes.com/cp863.html)

    Returns
    -------
    str
        The text with characters only from the specified codepage.
    """
    text = text.encode(codepage, "ignore").decode(codepage)
    text = text.encode("utf-8").decode("utf-8")

    return text


def rm_spaces(text):
    """
    Remove multiple spaces

    Parameters
    ----------
    text : str
        The input text to process.

    Returns
    -------
    str
        The text with multiple spaces replaced by a single space.
    """
    pattern = re.compile(r"( ){2,}")
    text = re.sub(pattern, r" ", text)

    return text


def process_url_html(text):
    """
    Remove URLs in text

    Parameters
    ----------
    text : str
        The input text to process.

    Returns
    -------
    str
        The text with URLs removed.
    """
    pattern = re.compile(r"(?:www|http)\S+|<\S+|\w+\/*>")
    text = re.sub(pattern, "", text)

    return text


def cleaner(text, rm_new_lines=False, do_lower=False):
    """
    Clean up an input text

    Parameters
    ----------
    text : str
        The input text to clean.
    rm_new_lines : bool, optional
        If True, remove new line characters. Default is False.
    do_lower : bool, optional
        If True, convert text to lowercase. Default is False.

    Returns
    -------
    str
        The cleaned text.
    """
    # Convert and normalize the unicode underlying representation
    text = convert_to_unicode(text)
    text = normalize_unicode(text)

    # Normalize whitespace characters and remove carriage return
    if rm_new_lines:
        remap = {ord("\f"): " ", ord("\r"): "", ord("\n"): "", ord("\t"): ""}
        text = text.translate(remap)
    else:
        remap = {ord("\f"): " ", ord("\r"): ""}
        text = text.translate(remap)

    # Normalize URL links
    text = process_url_html(text)

    # remove multiple spaces in text
    text = rm_spaces(text)

    if do_lower:
        text = text.lower()

    return text
