from . import default, contraceptive, treatment, insurance, iris
contraceptive.TRIAL_QUEUE_EXT = [
    *contraceptive.TRIAL_QUEUE,
    *insurance.TRIAL_QUEUE,
    *treatment.TRIAL_QUEUE,
    *iris.TRIAL_QUEUE,
]
insurance.TRIAL_QUEUE_EXT = [
    *insurance.TRIAL_QUEUE,
    *treatment.TRIAL_QUEUE,
    *contraceptive.TRIAL_QUEUE,
    *iris.TRIAL_QUEUE,
]
treatment.TRIAL_QUEUE_EXT = [
    *treatment.TRIAL_QUEUE,
    *contraceptive.TRIAL_QUEUE,
    *insurance.TRIAL_QUEUE,
    *iris.TRIAL_QUEUE,
]
iris.TRIAL_QUEUE_EXT = [
    *iris.TRIAL_QUEUE,
    *contraceptive.TRIAL_QUEUE,
    *treatment.TRIAL_QUEUE,
    *insurance.TRIAL_QUEUE,
]
contraceptive.TRIAL_QUEUE_EXT = contraceptive.sanitize_queue(
    contraceptive.TRIAL_QUEUE_EXT,
    PARAM_SPACE=contraceptive.PARAM_SPACE,
    DEFAULTS=contraceptive.DEFAULTS,
    FORCE=contraceptive.FORCE,
    MINIMUMS=contraceptive.MINIMUMS,
)
insurance.TRIAL_QUEUE_EXT = insurance.sanitize_queue(
    insurance.TRIAL_QUEUE_EXT,
    PARAM_SPACE=insurance.PARAM_SPACE,
    DEFAULTS=insurance.DEFAULTS,
    FORCE=insurance.FORCE,
    MINIMUMS=insurance.MINIMUMS,
)
treatment.TRIAL_QUEUE_EXT = treatment.sanitize_queue(
    treatment.TRIAL_QUEUE_EXT,
    PARAM_SPACE=treatment.PARAM_SPACE,
    DEFAULTS=treatment.DEFAULTS,
    FORCE=treatment.FORCE,
    MINIMUMS=treatment.MINIMUMS,
)
iris.TRIAL_QUEUE_EXT = iris.sanitize_queue(
    iris.TRIAL_QUEUE_EXT,
    PARAM_SPACE=iris.PARAM_SPACE,
    DEFAULTS=iris.DEFAULTS,
    FORCE=iris.FORCE,
    MINIMUMS=iris.MINIMUMS,
)
