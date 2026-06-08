from pydantic import BaseModel, Field

class InsuranceClaimInput(BaseModel):
    num_young_drivers: float
    age: float
    num_of_children: float
    years_job_held_for: float
    income: float
    value_of_home: float
    commute_dist: float
    vehicle_value: float
    policy_tenure: float
    five_year_total_claims_value: float = Field(..., alias="5_year_total_claims_value")
    five_year_num_of_claims: float = Field(..., alias="5_year_num_of_claims")
    license_points: float
    vehicle_age: float
    highest_education: str
    single_parent: str
    married: str
    gender: str
    type_of_use: str
    licence_revoked: str
    address_type: str
    occupation: str
    vehicle_type: str

class InsuranceClaimOutput(BaseModel):
    claim_probability: float
    claim_decision: int
    estimated_claim_value: float
