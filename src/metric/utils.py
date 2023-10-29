import editdistance
# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    if not target_text:
        if predicted_text:
            return 1
        return 0
    target_text_splitted = target_text.split(' ')
    predicted_text_splitted = predicted_text.split(' ')
    return editdistance.eval(target_text_splitted, predicted_text_splitted) / len(target_text_splitted)
