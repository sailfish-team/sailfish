from sailfish import sym

def get_grid_from_config(config):
    for x in sym.KNOWN_GRIDS:
        if x.__name__ == config.grid:
            return x

    return None
