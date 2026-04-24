import json


def is_flat_list(lst):
    return all(not isinstance(x, (list, dict)) for x in lst)


def format_json(obj, indent=0, indent_step=2):
    space = ' ' * indent

    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            formatted_v = format_json(v, indent + indent_step, indent_step)
            items.append(f'\n{space}{" " * indent_step}"{k}": {formatted_v}')
        return '{' + ','.join(items) + f'\n{space}' + '}'

    elif isinstance(obj, list):
        if is_flat_list(obj):
            # 单行数组
            return '[' + ', '.join(json.dumps(x) for x in obj) + ']'
        else:
            items = []
            for x in obj:
                items.append(f'\n{space}{" " * indent_step}{format_json(x, indent + indent_step, indent_step)}')
            return '[' + ','.join(items) + f'\n{space}' + ']'

    else:
        return json.dumps(obj)


if __name__ == '__main__':

    # 示例
    data = {
        "a": [1, 2, 3],
        "b": [[1, 2], [3, 4]],
        "c": {"d": [5, 6, 7], "e": [{"x": 1}, {"y": 2}]}
    }

    print(format_json(data))