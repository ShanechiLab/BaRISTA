from enum import Enum

def enumval_formatter(subject, trial_list):
    return [f"S{subject}_{trial}" for trial in trial_list]

def holdout_enumval_formatter(subject, trial_list):
    return [f"HS{subject}_{trial}" for trial in trial_list]

class BrainTreebankAvailableSessions(Enum):
    SUBJ_1: list = enumval_formatter("1", ["0", "2"])
    SUBJ_2: list = enumval_formatter("2", ["0", "1", "2", "3", "4", "5"])
    SUBJ_3: list = enumval_formatter("3", ["1", "2"])
    SUBJ_4: list = enumval_formatter("4", ["1", "2"])
    SUBJ_5: list = enumval_formatter("5", ["0"])
    SUBJ_6: list = enumval_formatter("6", ["0", "1"])
    SUBJ_7: list = enumval_formatter("7", ["1"])
    SUBJ_8: list = enumval_formatter("8", ["0"])
    SUBJ_9: list = enumval_formatter("9", ["0"])
    SUBJ_10: list = enumval_formatter("10", ["1"])

    ## Heldout trials.
    HOLDSUBJ_1: list = holdout_enumval_formatter("1", ["1"])
    HOLDSUBJ_2: list = holdout_enumval_formatter("2", ["6"])
    HOLDSUBJ_3: list = holdout_enumval_formatter("3", ["0"])
    HOLDSUBJ_4: list = holdout_enumval_formatter("4", ["0"])
    HOLDSUBJ_6: list = holdout_enumval_formatter("6", ["4"])
    HOLDSUBJ_7: list = holdout_enumval_formatter("7", ["0"])
    HOLDSUBJ_10: list = holdout_enumval_formatter("10", ["0"])