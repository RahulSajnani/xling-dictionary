from helper_functions import *
from create_index import *


def create_embeddings(dictionary_path, output_directory, encoder_cache,  validation_path = None, test_path = None):
    '''
    Creates embeddings for every language
    '''

    lang_map = {'HI': 'hi', 'BE': 'bn', 'GU': 'gu', 'OD': 'or', 'PU': 'pa', 'EN': 'en', 'MA': 'mr'}

    train_dictionary = read_json_file(dictionary_path)

    words_dictionary = {}

    for lang in lang_map:
        words_dictionary[lang_map[lang]] = []
    redundant_dict = {}
    # separate language target words
    prev_word = ""
    for l_dict in train_dictionary:


        target_id = l_dict["Target_ID"]
        target_word = l_dict["Target_keyword"]

        if redundant_dict.get(target_word) is None:
            redundant_dict[target_word] = 1
            words_dictionary[lang_map[target_id]].append(target_word)

    if validation_path is not None:

        print(validation_path)
        validation_dictionary = read_json_file(validation_path)
        prev_word = ""

        for l_dict in validation_dictionary:

            target_id = l_dict["Target_ID"]
            target_word = l_dict["Target_keyword"]



            #if prev_word == target_word:
            #    continue
            #prev_word = target_word

            if redundant_dict.get(target_word) is None:
                redundant_dict[target_word] = 1
                words_dictionary[lang_map[target_id]].append(target_word)

    if test_path is not None:

        print(test_path)
        validation_dictionary = read_json_file(test_path)
        prev_word = ""

        for l_dict in validation_dictionary:

            target_id = l_dict["Target_ID"]
            target_word = l_dict["Target_keyword"]

            if redundant_dict.get(target_word) is None:
                redundant_dict[target_word] = 1
                words_dictionary[lang_map[target_id]].append(target_word)



    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # create embeddings
    for lang in lang_map:

        create_index(words_dictionary[lang_map[lang]],
                    os.path.join(output_directory, lang_map[lang] + ".index"),
                    os.path.join(output_directory, lang_map[lang] + ".vocab"),
                    encoder_cache)


        with open(os.path.join(output_directory, lang_map[lang] + ".vocab"), 'w') as fp:
            for listitem in words_dictionary[lang_map[lang]]:
                fp.write('%s\n' % listitem)

if __name__ == "__main__":

    ############################# Argument Parser

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Training dictionary path", required = True)
    parser.add_argument("--output", help = "Output directory to save embeddings", required = True)
    parser.add_argument("--cache", help = "Bert cache", required = True)
    parser.add_argument("--val", help = "validation", required = False, default=None)
    parser.add_argument("--test", help = "path to test json", required = False, default=None)

    args = parser.parse_args()

    #############################################

    create_embeddings(args.input, args.output, args.cache, args.val, args.test)
