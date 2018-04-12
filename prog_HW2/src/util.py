def print_dict(my_dict):
    for key in my_dict:
        print(key, " : ", my_dict[key])

def print_dict_tofile(my_dict):
    with open('log_file.txt', 'w') as log_file:
        for key in my_dict:
            print(key, " : ", my_dict[key], file=log_file)

