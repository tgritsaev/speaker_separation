model_path = "3-gram.arpa"
lower_model_path = "lower_3-gram.arpa"

with open(model_path, 'r') as f1:
    with open(lower_model_path, "w") as f2:
        for line in f1:
            f2.write(line.lower())