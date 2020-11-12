from helper_functions import *
from create_index import *


def create_embeddings(dictionary_path, output_directory, encoder_cache,  validation_path = None):
    '''
    Creates embeddings for every language
    '''

    lang_map = {'HI': 'hi', 'BE': 'bn', 'GU': 'gu', 'OD': 'or', 'PU': 'pa', 'EN': 'en', 'MA': 'mr'}

    train_dictionary = read_json_file(dictionary_path)

    words_dictionary = {}

    for lang in lang_map:
        words_dictionary[lang_map[lang]] = []

    # separate language target words    
    for l_dict in train_dictionary:
        
        target_id = l_dict["Target_ID"]
        target_word = l_dict["Target_keyword"]
        words_dictionary[lang_map[target_id]].append(target_word)
    
    if validation_path is not None:

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # create embeddings
    for lang in lang_map:

        create_index(words_dictionary[lang_map[lang]], 
                    os.path.join(output_directory, lang_map[lang] + ".index"),
                    os.path.join(output_directory, lang_map[lang] + ".vocab"),
                    encoder_cache)

if __name__ == "__main__":

    ############################# Argument Parser 

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "Training dictionary path", required = True)
    parser.add_argument("--output", help = "Output directory to save embeddings", required = True)
    parser.add_argument("--cache", help = "Bert cache", required = True)
    args = parser.parse_args()

    #############################################

    create_embeddings(args.input, args.output, args.cache)