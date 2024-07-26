NUMERIC_FEAT = [
    "age",
    "region_code",
    "annual_premium",
    "policy_sales_channel",
    "vintage",
]

CATEGORY_FEAT = dict.fromkeys(
    [
        "gender",
        "driving_license",
        "previously_insured",
        "vehicle_age",
        "vehicle_damage",
    ]
)

FEATS = [*NUMERIC_FEAT, *CATEGORY_FEAT.keys()]

print("all features:", FEATS)

TARGET = "response"
