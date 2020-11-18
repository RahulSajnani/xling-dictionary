import json
import argparse

def read_json_file(file_path):

    '''
    Function to read json file

    Arguments:
        file_path

    Returns:
        json object
    '''

    with open(file_path) as fp:

        input_dictionary = json.load(fp)


    return input_dictionary

if __name__ == "__main__":

    ############################# Argument Parser

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help = "Training dictionary path", required = True)
    args = parser.parse_args()

    #############################################

    """
    MAX phrase length = 64
    """

    train_dictionary = read_json_file(args.path)

    language_dictionary = {}
    ids = 0

    length = 0
    for i in train_dictionary:

        phrase = i["Source_text"]
        p_len = len(phrase.split())

        if p_len > length:
            length = p_len
        if i["Target_ID"] == "EN":
            print(phrase, i["Target_keyword"])
        language_idx = i["Source_ID"]
        language_idx_t = i["Target_ID"]
        if language_idx not in language_dictionary:
            language_dictionary[language_idx] = ids
            ids = ids + 1

        if language_idx_t not in language_dictionary:
            language_dictionary[language_idx_t] = ids
            ids = ids + 1

        # print(i)
    # print(language_dictionary)
    # print(length)
