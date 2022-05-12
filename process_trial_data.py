### Series of functions that deal with data that has been sectioned into trials


def find_true_trial_end_samples(n_trials,true_len_trial):
    """Identifies and saves the time samples immediately following the last
    time sample in a trial--immediately after so that you can slice with it
    and know that you will select all samples up to but not including, the time
    you have input"""

    true_trial_ends_not_inclusive = []
    for i in range(1,n_trials+1):
        true_trial_ends_not_inclusive.append((true_len_trial*i)-1)

    return true_trial_ends_not_inclusive


# def trial_labels():
