def get_divider_str(msg: str, length: int = 100, line_ends: str = ''):
    space_left = length - 2 - len(msg)
    if space_left <= 0:
        return msg
    elif space_left % 2 == 0:
        left, right = space_left // 2, space_left // 2
    else:
        left, right = space_left // 2, space_left // 2 + 1
    return f"{line_ends}{left * '='} {msg} {right * '='}{line_ends}"
