import os
import re

def get_abs_path(rel_path):
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, rel_path)

def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url