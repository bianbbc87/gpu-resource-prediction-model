def normalize_flops(flops, total):
    return flops / total if total > 0 else 0
