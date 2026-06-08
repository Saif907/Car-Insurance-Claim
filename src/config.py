"""
Configuration and schema definitions for the Car Insurance Claim project.
"""

# Categorical mapping definitions
EDUCATION_RANK = [['<High School', 'High School', 'Bachelors', 'Masters', 'PhD']]

# Feature lists
NUMERICAL_COLS = [
    'num_young_drivers', 'age', 'num_of_children', 'years_job_held_for',
    'income', 'value_of_home', 'commute_dist', 'vehicle_value',
    'policy_tenure', '5_year_total_claims_value', '5_year_num_of_claims',
    'license_points', 'vehicle_age'
]

CAT_COLS_ORD = ['highest_education']
CAT_COLS_BIN = ['single_parent', 'married', 'gender', 'type_of_use', 'licence_revoked', 'address_type']
CAT_COLS_ONE_HOT = ['occupation', 'vehicle_type']

ALL_FEATURES = NUMERICAL_COLS + CAT_COLS_ORD + CAT_COLS_BIN + CAT_COLS_ONE_HOT

TARGET_CLASS = 'claim_flag'
TARGET_REG = 'clm_amt'

# Columns to drop to avoid multicollinearity
COLS_TO_DROP = ['cat_one_hot__occupation_Blue Collar', 'cat_one_hot__vehicle_type_Minivan']
