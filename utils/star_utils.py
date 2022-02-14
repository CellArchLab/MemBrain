import re
from collections import OrderedDict
import numpy as np


def read_star_file_as_dict(star_path):
    """
    opens a star file and returns it as a Python dictionary.
    Keys of the dictionary correspond to titles of properties in star file.
    Using the keys, one can access lists, giving the respective property of each instance in the star file
    """
    data_lists = read_star_file(star_path)
    out_dict = OrderedDict()
    for list in data_lists:
        out_dict[list[0][1:]] = list[1:]
    return out_dict


def find_tomo_mb_idx(tomo_token, mb_token, star_dict, stack_token=None):
    """
    Searches a star file for a combination of tomo and membrane token. (optionally also stack token)
    The index is returned.
    """
    tomo_tokens = star_dict['tomoToken']
    mb_tokens = star_dict['mbToken']
    if stack_token is not None:
        stack_tokens = star_dict['stackToken']
    for i in range(len(tomo_tokens)):
        if tomo_tokens[i] == tomo_token and mb_tokens[i] == mb_token and (stack_token is None or stack_tokens[i] == stack_token):
            return i
    return None


def read_star_file(star_path, with_idcs=False):
    """
    Star file is read as a list of lists.
    The use of this function is not recommended.
    """
    read_names = False
    read_data = False
    data_lists = []
    key_count = 0
    key_count_dict = {}
    with open(star_path) as star_file:
        for line in star_file:
            if line.startswith('loop_'):
                read_names = True
                continue
            if read_names:
                if not line.startswith('_'):
                    read_names = False
                    read_data = True
                else:
                    data_key = line[:line.find(' ')]
                    data_lists.append([data_key])
                    key_count_dict[data_key] = key_count
                    key_count += 1
            if read_data:
                if '\t' in line:
                    separator = '\t'
                else:
                    separator = ' '
                # split = re.split('\W+', line)
                split = line.split(separator)
                split = [entry for entry in split if entry != '']
                if len(split) == 1 and split[0] == '\n':
                    break
                split = [entry for entry in split if entry != '\n']
                for i in range(len(data_lists) - 1):
                    # print split

                    data_lists[i].append(split[i])
                if '\n' in split[len(split)- 1]:
                    split[len(split) - 1] = split[len(split) - 1][:-1]
                data_lists[len(data_lists) - 1].append(split[len(split) - 1])
    if with_idcs:
        return data_lists, key_count_dict
    return data_lists


def write_star_file_from_dict(star_path, star_dict):
    """
    Given a Python dictionary of the correct format, a star file is created at the specified path.
    Correct dictionary format:
    - Keys correspond to property titles in the star files
    - each key contains a list. These lists need to be of the same length!
    """
    with open(star_path, 'w') as star_file:
        star_file.writelines('data_\n')
        star_file.writelines('\n')
        star_file.writelines('loop_\n')
        for i, key in enumerate(star_dict.keys()):
            star_file.writelines('_' + key + ' #' + str(i + 1) + '\n')
        data_len = len(star_dict[list(star_dict.keys())[0]])
        for j in range(data_len):
            line = ''
            for key in star_dict.keys():
                line = line + str(star_dict[key][j]) + '\t'
            line = line[:-1] + '\n'
            star_file.writelines(line)


def merge_line_from_star_dict_to_star_dict(line_idx, in_star_dict, out_star_dict):
    for key, entry in in_star_dict.items():
        cur_entry = entry[line_idx]
        if key in out_star_dict.keys():
            out_star_dict[key].append(cur_entry)
        else:
            out_star_dict[key] = [cur_entry]
    return out_star_dict


def write_star_file(star_path, star_lists):
    '''
    Writes list into file
    :param star_path: output file name
    :param star_list: list containing lists of entries, e.g. [[_rlnMicrographName, ../../../../../real_data/mics/tomo_mb1_bin2.mrc, ...] [_psSegRot, ...] ...]
    :return:
    '''

    with open(star_path, 'w') as star_file:
        star_file.writelines('data_\n')
        star_file.writelines('\n')
        star_file.writelines('loop_\n')
        for i in range(len(star_lists)):
            star_file.writelines(star_lists[i][0] + ' #' + str(i + 1) + '\n')
        for j in range(len(star_lists[0]) - 1):
            line = ''
            for i in range(len(star_lists)):
                line = line + str(star_lists[i][j + 1]) + '\t'
            line = line[:-1] + '\n'
            star_file.writelines(line)


def add_or_change_column_to_star_file(star_path, column_title, column_values):
    """
    Given a columns title, and column values (need to be a list), the star file is adjusted:
    Either a column with the specified title is created or -- if it already exists -- replaced.
    column_values needs to be a list of the same length as the star file
    """
    star_dict = read_star_file_as_dict(star_path)
    assert isinstance(column_values, list)
    assert len(star_dict[list(star_dict.keys())[0]]) == len(column_values), 'needed length ' + str(len(star_dict[list(star_dict.keys())[0]])) + \
                                                                                       ' given length ' + str(len(column_values))
    # assert isinstance(column_values[0], str)
    star_dict[column_title] = column_values
    write_star_file_from_dict(star_path, star_dict)

def copy_star_file(in_star, out_star):
    """
    Creates a copy of the specified star file at the given path.
    """
    star_dict = read_star_file_as_dict(in_star)
    write_star_file_from_dict(out_star, star_dict)


def remove_line_from_dict(star_dict, line_number):
    """
    Removes one entry from the star dictionary.
    """
    for key in star_dict.keys():
        cur_list = star_dict[key]
        del cur_list[line_number]
