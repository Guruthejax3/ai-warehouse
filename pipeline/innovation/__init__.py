"""Innovation features — damage prediction, incident reports, pallet stability, WMS/CCTV integration."""

from pipeline.innovation.damage_prediction import predict_damage
from pipeline.innovation.incident_report import generate_incident_report
from pipeline.innovation.pallet_stability import assess_pallet_stability

__all__ = ["predict_damage", "generate_incident_report", "assess_pallet_stability"]
