import json
import logging
import random


def load_data(data_file):
    logging.info('loading JSON file at {}'.format(data_file))
    with open(data_file, 'r') as json_file:
        return json.load(json_file)

def main():
    data_dict = load_data('./data/data.json')
    keys = list(data_dict.keys())
    random.shuffle(keys)
    shuffled_data_dict = dict()
    for key in keys:
        shuffled_data_dict.update({key:data_dict[key]})
    print(json.dumps(shuffled_data_dict))


if __name__ == "__main__":
    main()
