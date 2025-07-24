from datetime import datetime


def validate_data(start, end):
    if end < start:
        return False
    else:
        return True
    