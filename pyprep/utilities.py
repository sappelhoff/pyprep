
def union(list1, list2):
    return list(set(list1 + list2))


def set_diff(list1, list2):
    return list(set(list1) - set(list2))


def intersect(list1, list2):
    return list(set(list1).intersection(set(list2)))

