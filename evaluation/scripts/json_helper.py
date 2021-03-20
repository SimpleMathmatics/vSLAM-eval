import json


class JsonHelper:
    @staticmethod
    def add_json(path_to_file, key, value):
        try:
            with open(path_to_file) as file:
                data = json.load(file)
            data[key] = value
            with open(path_to_file, 'w') as json_file:
                json.dump(data, json_file)

        except IOError:
            data = {key: value}
            with open(path_to_file, 'a') as json_file:
                json.dump(data, json_file)
