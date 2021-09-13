def fix_general_label_error(labels, type, slots):
    label_dict = dict([ (l[0], l[1]) for l in labels]) if type else dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels]) 
    return label_dict