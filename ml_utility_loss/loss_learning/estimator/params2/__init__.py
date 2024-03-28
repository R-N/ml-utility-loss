from . import default, contraceptive, treatment, insurance
contraceptive.TRIAL_QUEUE_EXT = [
    *contraceptive.TRIAL_QUEUE,
    *insurance.TRIAL_QUEUE,
    *treatment.TRIAL_QUEUE,
]
insurance.TRIAL_QUEUE_EXT = [
    *insurance.TRIAL_QUEUE,
    *treatment.TRIAL_QUEUE,
    *contraceptive.TRIAL_QUEUE,
]
treatment.TRIAL_QUEUE_EXT = [
    *treatment.TRIAL_QUEUE,
    *contraceptive.TRIAL_QUEUE,
    *insurance.TRIAL_QUEUE,
]
