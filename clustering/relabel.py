def relabel(labels):
    """
    Relabel cluster labels so that they are in ascending order.
    """

    appearance_order = {}
    num_found = 0

    for label in labels:
        if label not in appearance_order:
            appearance_order[label] = num_found
            num_found += 1

    for i, old_label in enumerate(labels):
        labels[i] = appearance_order[old_label]
    return labels
