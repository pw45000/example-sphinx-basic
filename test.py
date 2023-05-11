import base64
import json

import torch


def flatten_nested_dictionary(a_dictionary: dict):
    """
    Given a multidimensional dictionary, collapse it into a single dimensional dictionary.
    
    :description: Flattens a multidimensional dictionary into a 1d dictionary. This means all keys and values
    are at the same level, regardless of nesting. Why? Because this is to aid in simply checking for settings,
    not actually manipulating values. So, for instance, this function will be called such that we can let's say
    find 'max_length' like such: if max_length in flattened_dict.keys() regardless of its nesting.
    :param: a_dictionary- the dictionary to collapse.
    :type dict
    :return: A 1-d dictionary with all the keys and values stored in one place.
    :acknowledgements: Adapted from
    https://stackoverflow.com/questions/52081545/python-3-flattening-nested-dictionaries-and-lists-within-dictionaries
    """
    out = {}
    for key, val in a_dictionary.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = flatten_nested_dictionary(subdict).items()
                out.update({key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out

def decode_encoded_tokenized_tensor(a_encoded_tokens):
    """
    Decode a base64 encoded tokenized tensor into a tensor.
    
    :description: In this function's case, it decodes a base64 encoded tokenized tensor into a tensor, such that
    it can be fed into the backend's model. It does not detokenize the list, as there is no tokenizer on the backend.
    :param: a_encoded_tokens- The encoded base64 string sent back from the backend to decode.
    :return: torch.Tensor, representing the tokenized chat history.
    :acknowledgements: Antony Mercurio, who recommended me to utilize this method and gave me portions of the code.
    """
    partial_decoded_history = base64.b64decode(a_encoded_tokens)
    list_chat_history = json.loads(partial_decoded_history.decode("utf-8"))
    return torch.tensor([list_chat_history])

def get_encoded_str_from_token_list(message):
    """
    Gets the base 64 encoded string given a list of tokens.
    
    :description: Encodes a list of tokens into a base 64 string. Why? Simple. Easy serialization of complex data.
    My friend, Anthony Mercurio, recommended me this as he saw it being used by a start-up named NovelAI for
    performance reasons. I can see why, as the only other way of serializing a stupidly large list would be using
    the python pickle library or making a large multidimensional list, which would require quite the lot of string
    manipulation. So, I chose the lesser of the evils and decided to use base64 as my serialization algorithm.
    Now, there is quite a lot of translation that goes on, so let me break it down:
    1. First, the list itself is converted to a json readable string so it can be broken down into raw bytes.
    2. The string is then turned into bytes to prepare it for base64 encoding.
    3. Then, the encoded base64 are turned into a string.
    4. This might seem counterintuitive, but the string is then decoded for utf-8. Why? So it can be sent via json.
       Otherwise, there'd be no way to send it over, and I've confirmed the string itself is still base64 encoded.
    :param message: The list of tokens representing an arbitrary message. Typically is the entire chat history in
    practice, however.
    :return: str, representing the original message now encoded into base64.
    :acknowledgements: Antony Mercurio, who recommended me to utilize this method and gave me portions of the code.
    """
    encoded_str = json.dumps(
        base64.b64encode(bytes(json.dumps(message.tolist()), encoding='utf-8')).decode("utf-8"))
    return encoded_str
