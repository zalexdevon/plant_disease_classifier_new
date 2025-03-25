def get_params_transform_list_to_1_value(param_grid):
    values = [item[0] for item in param_grid.values()]
    return dict(zip(param_grid.keys(), values))


param_grid = {
    "C": [1],
    "A": [2],
}

print(get_params_transform_list_to_1_value(param_grid))
