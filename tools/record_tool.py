import json
from langchain.tools import Tool
from utils.database_ops import get_patient_record, update_patient_record

def add_or_update_history(input_str: str) -> str:
    """Input: JSON with keys patient_id and data (object to merge)."""
    try:
        payload = json.loads(input_str)
        pid = payload.get("patient_id")
        data = payload.get("data", {})
        if not pid or not isinstance(data, dict):
            return "Invalid payload. Expected patient_id and data (dict)."
        update_patient_record(pid, data)
        return f"Medical history for patient {pid} updated."
    except json.JSONDecodeError:
        return "Invalid JSON format for payload."
    except Exception as e:
        return f"Failed to update patient history: {e}"

def retrieve_history(patient_id: str) -> str:
    rec = get_patient_record(patient_id)
    return json.dumps(rec, indent=2) if rec else "Patient record not found."

def get_record_tools():
    return [
        Tool(
            name="Manage Medical Records",
            func=add_or_update_history,
            description=(
                "Add or update patient medical history. Input is a JSON string with keys: "
                "patient_id and data (object to merge)."
            ),
        ),
        Tool(
            name="Retrieve Medical History",
            func=retrieve_history,
            description="Retrieve a patient's full medical history. Input is the patient_id string.",
        ),
    ]
