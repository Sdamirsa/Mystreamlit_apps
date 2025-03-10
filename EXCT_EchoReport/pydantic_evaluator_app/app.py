
import streamlit as st
import pandas as pd
import json
import datetime
import numpy as np
from io import BytesIO
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import base64

###############################################################################
# Pydantic Models (for reference)
###############################################################################
class Z_score_of_reported_numeric(BaseModel):
    z_score: Optional[float] = None
    type_of_z_score: Optional[str] = None

class NumericValue(BaseModel):
    numeric: Optional[float] = None
    unit: Optional[str] = None
    z_score: Optional[List[Z_score_of_reported_numeric]] = None

class Bool_with_Other(str, Enum):
    TRUE = "True or yes"
    FALSE = "False or No"
    OTHER = "Other"

class DilationSeverity(str, Enum):
    APLASTIC = "Aplastic"
    HYPOPLASTIC = "Hypoplastic"
    SMALL = "Small"
    NORMAL = "Normal"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class HypertrophySeverity(str, Enum):
    NO = "No"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderately-Severely depressed"
    SEVERE = "Severe"

class StenosisSeverity(str, Enum):
    NO = "No"
    TRACE_TRIVIAL = "Trace/Trivial"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class RegurgitationSeverity(str, Enum):
    NO = "No"
    TRACE_TRIVIAL = "Trace/Trivial"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class StructuralStatus(str, Enum):
    NORMAL = "Structurally normal"
    ABNORMAL = "Structurally abnormal"
    OTHER = "Other"

class SystolicDiastolicFunction(str, Enum):
    NORMAL = "Normal"
    LOW_NORMAL = "Low normal"
    MILDLY_DEPRESSED = "Mildly depressed"
    MILD_MODERATE_DEPRESSED = "Mild-Moderately depressed"
    MODERATELY_DEPRESSED = "Moderately depressed"
    MODERATE_SEVERE_DEPRESSED = "Moderately-Severely depressed"
    SEVERELY_DEPRESSED = "Severely depressed"
    AKINETIC = "Akinetic"
    HYPERDYNAMIC = "Hyperdynamic"
    OTHER = "Other"

class PressureSeverity(str, Enum):
    NORMAL = "Normal"
    MILD_ELEVATED = "Mildly elevated"
    MODERATE_ELEVATED = "Moderately elevated"
    SEVERE_ELEVATED = "Severely elevated"
    LESS_HALF_SYSTEMIC = "Less than half systemic"
    HALF_SYSTEMIC = "Half systemic"
    APPROACHING_SYSTEMIC = "Approaching systemic"
    SYSTEMIC = "Systemic"
    SUPRASYSTEMIC = "Suprasystemic"

class Size(str, Enum):
    TINY = "Tiny"
    SMALL = "Small"
    SMALL_MODERATE = "Small-moderate"
    MODERATE = "Moderate"
    MODERATE_LARGE = "Moderate-large"
    LARGE = "Large"

class ASD_type(str, Enum):
    SECUNDUM = "Secundum"
    PRIMUM = "Primum"
    PFO = "Patent Foramen Ovale"
    SUPERIOR_SINUS_VENOSUS = "Superior sinus venosus"
    CORONARY_SINUS = "Coronary sinus"
    SURGICAL = "Surgically created atrial septal defect"
    INFERIOR_SINUS_VENOSUS = "Inferior sinus venosus"
    OTHER = "Other"

class VSD_type(str, Enum):
    PERIMEMBRANOUS = "Perimembranous"
    ANTERIOR_MALALIGNMENT = "Anterior malalignment"
    POSTERIOR_MALALIGNMENT = "Posterior malalignment"
    MID_MUSCULAR = "Mid muscular"
    POSTERIOR_MUSCULAR = "Posterior muscular"
    APICAL_MUSCULAR = "Apical muscular"
    INLET = "Inlet"
    OUTLET_DOUBLY_COMMITTED = "Outlet/doubly-committed juxta-arterial"
    PERIMEMBRANOUS_INLET = "Perimembranous inlet"
    OUTLET_SUBAORTIC = "Outlet/subaortic"
    CONOVENTRICULAR = "Conoventricular"
    OTHER = "Other"

class RA_info(BaseModel):
    RA_dilation: Optional[DilationSeverity] = None

class LA_info(BaseModel):
    LA_dilation: Optional[DilationSeverity] = None
    LA_volume_indexed: Optional[NumericValue] = None

class Atria(BaseModel):
    RA: Optional[RA_info] = None
    LA: Optional[LA_info] = None

class RVSizeStructure(BaseModel):
    RV_dilation: Optional[DilationSeverity] = None
    RV_hypertrophy: Optional[HypertrophySeverity] = None

class RVFunction(BaseModel):
    RV_systolic_function: Optional[SystolicDiastolicFunction] = None

class RV_info(BaseModel):
    RV_size_structure: Optional[RVSizeStructure] = None
    RV_function: Optional[RVFunction] = None

class LVSizeStructure(BaseModel):
    LV_dilation: Optional[DilationSeverity] = None
    LV_hypertrophy: Optional[HypertrophySeverity] = None
    LV_volume_systole: Optional[NumericValue] = None
    LV_volume_diastole: Optional[NumericValue] = None

class LVFunction(BaseModel):
    LV_systolic_function: Optional[SystolicDiastolicFunction] = None
    LV_systolic_function_other: Optional[str] = None
    LVEF: Optional[NumericValue] = None

class LV_info(BaseModel):
    LV_size_structure: Optional[LVSizeStructure] = None
    LV_function: Optional[LVFunction] = None

class Ventricles(BaseModel):
    RV: Optional[RV_info] = None
    LV: Optional[LV_info] = None

class TricuspidValve(BaseModel):
    TV_structural_status: Optional[StructuralStatus] = None
    TV_structural_status_other: Optional[str] = None
    TV_regurgitation_severity: Optional[RegurgitationSeverity] = None

class PulmonaryValve(BaseModel):
    PV_annulus_size: Optional[NumericValue] = None
    PV_stenosis_severity: Optional[StenosisSeverity] = None
    PV_structural_status: Optional[StructuralStatus] = None
    PV_structural_status_other: Optional[str] = None
    PV_regurgitation_severity: Optional[RegurgitationSeverity] = None
    PV_pressure_gradient: Optional[NumericValue] = None

class MitralValve(BaseModel):
    MV_stenosis_severity: Optional[StenosisSeverity] = None
    MV_structural_status: Optional[StructuralStatus] = None
    MV_structural_status_other: Optional[str] = None
    MV_regurgitation_severity: Optional[RegurgitationSeverity] = None

class AorticValve(BaseModel):
    AV_structural_status: Optional[StructuralStatus] = None
    AV_structural_status_other: Optional[str] = None
    AV_leaflets: Optional[int] = None
    AV_stenosis_severity: Optional[StenosisSeverity] = None
    AV_regurgitation_severity: Optional[RegurgitationSeverity] = None
    AV_peak_pressure_gradient: Optional[NumericValue] = None
    AV_mean_pressure_gradient: Optional[NumericValue] = None

class Valves(BaseModel):
    tricuspid: Optional[TricuspidValve] = None
    pulmonary: Optional[PulmonaryValve] = None
    mitral: Optional[MitralValve] = None
    aortic: Optional[AorticValve] = None

class Aorta(BaseModel):
    arch_sidedness: Optional[str] = None
    aortic_root_size: Optional[NumericValue] = None
    ascending_aorta_diameter: Optional[NumericValue] = None
    aortic_isthmus_size: Optional[NumericValue] = None
    coarctation: Optional[bool] = None
    coarctation_gradient: Optional[NumericValue] = None

class GreatVessels(BaseModel):
    aorta: Optional[Aorta] = None

class PulmonaryHypertension(BaseModel):
    severity: Optional[str] = None
    TR_jet_gradient: Optional[NumericValue] = None
    IVS_flattening_in_systole: Optional[bool] = None

class ASD(BaseModel):
    atrial_communication_present: Optional[bool] = None
    asd_types: Optional[List[ASD_type]] = None
    atrial_communication_present_other: Optional[str] = None
    size: Optional[Size] = None
    direction_of_flow: Optional[str] = None

class VSD(BaseModel):
    ventricular_communication_present: Optional[bool] = None
    vsd_types: Optional[List[VSD_type]] = None
    ventricular_communication_present_other: Optional[str] = None
    size: Optional[Size] = None
    direction_of_flow: Optional[str] = None
    peak_gradient: Optional[NumericValue] = None

class PDA(BaseModel):
    present: Optional[bool] = None
    direction_of_flow: Optional[str] = None
    size: Optional[NumericValue] = None
    peak_gradient: Optional[NumericValue] = None

class SurgicalHistory(BaseModel):
    prior_surgical_interventions: Optional[str] = None

class EchoReport(BaseModel):
    atria: Optional[Atria] = None
    ventricles: Optional[Ventricles] = None
    valves: Optional[Valves] = None
    great_vessels: Optional[GreatVessels] = None
    pHTN: Optional[PulmonaryHypertension] = None
    asd: Optional[ASD] = None
    vsd: Optional[VSD] = None
    pda: Optional[PDA] = None
    surgical_history: Optional[SurgicalHistory] = None

    class Config:
        use_enum_values = True


###############################################################################
# Constants & Utility Functions
###############################################################################
EVALUATION_LABELS = ["missing", "spurious", "incorrect", "correct but incomplete", "correct"]
EVALUATION_COLORS = {
    "missing": "#FF9999",  # Light red
    "spurious": "#FFD699",  # Light orange
    "incorrect": "#FFFF99",  # Light yellow
    "correct but incomplete": "#99CCFF",  # Light blue
    "correct": "#99FF99"    # Light green
}

def parse_pydantic_path_and_insert(dictionary, path_parts, value):
    """
    Safely build nested dictionaries from path_parts,
    ignoring or overwriting any non-dict nodes en route.
    """
    current = dictionary
    for i, part in enumerate(path_parts):
        is_last = (i == len(path_parts) - 1)
        if is_last:
            # Final part: store the actual value
            current[part] = value
        else:
            # Intermediate part: ensure current[part] is a dict
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

def build_pydantic_dictionary_from_row(row):
    """
    Build nested dict from columns containing '_Genrator_...',
    then parse into EchoReport if desired.
    """
    data_dict = {}
    for col_name in row.index:
        if "_Genrator_" in col_name:
            after_gen = col_name.split("_Genrator_", 1)[1]
            path_parts = after_gen.split("_")
            cell_value = row[col_name]
            
            # Handle NaN values
            if pd.isna(cell_value):
                cell_value = None
                
            parse_pydantic_path_and_insert(data_dict, path_parts, cell_value)
    return data_dict

def column_is_skipped(col_name, skip_list):
    """
    Return True if col_name ends with any string in skip_list (case-insensitive).
    """
    col_name_lower = col_name.lower()
    for skip_ending in skip_list:
        skip_ending = skip_ending.strip().lower()
        if skip_ending and col_name_lower.endswith(skip_ending):
            return True
    return False

def group_fields_by_category(fields):
    """
    Group fields by their top-level category (atria, ventricles, etc.)
    Returns a dictionary of category -> list of field names
    """
    categories = {}
    for field in fields:
        parts = field.split("_Genrator_")[1].split("_")
        category = parts[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(field)
    return categories

def get_field_display_name(field):
    """
    Convert field name to a more readable format
    """
    if "_Genrator_" in field:
        parts = field.split("_Genrator_")[1].split("_")
        return " → ".join(parts)
    return field

def to_excel_download(df):
    """
    Generate Excel download without xlsxwriter dependency
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="evaluation_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel file</a>'
    return href


###############################################################################
# Callback Functions & Session State
###############################################################################
def init_session_state():
    """
    Initialize session state variables if they don't exist.
    """
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "all_Genrator_cols" not in st.session_state:
        st.session_state["all_Genrator_cols"] = []
    if "skip_list" not in st.session_state:
        st.session_state["skip_list"] = []
    if "current_row_index" not in st.session_state:
        st.session_state["current_row_index"] = 0
    if "current_field_index" not in st.session_state:
        st.session_state["current_field_index"] = 0
    if "current_category" not in st.session_state:
        st.session_state["current_category"] = None
    if "unstructured_col" not in st.session_state:
        st.session_state["unstructured_col"] = "Report"
    if "unstructured_text_height" not in st.session_state:
        st.session_state["unstructured_text_height"] = 500
    if "evaluation_column_height" not in st.session_state:
        st.session_state["evaluation_column_height"] = 600
    if "text_column_height" not in st.session_state:
        st.session_state["text_column_height"] = 600
    if "show_json_for_current_row" not in st.session_state:
        st.session_state["show_json_for_current_row"] = False
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "by_field"  # Options: "by_field", "by_category"
    if "search_term" not in st.session_state:
        st.session_state["search_term"] = ""
    if "evaluation_stats" not in st.session_state:
        st.session_state["evaluation_stats"] = {}
    if "show_all_fields" not in st.session_state:
        st.session_state["show_all_fields"] = False


def load_excel_file(uploaded):
    """
    Read the Excel file and store in session_state.
    Also ensure a DateOfReview column and any needed evaluation columns.
    """
    try:
        df = pd.read_excel(uploaded)
        
        # Ensure DateOfReview
        if "DateOfReview" not in df.columns:
            df["DateOfReview"] = None

        # Identify Genrator columns
        Genrator_cols = [c for c in df.columns if "_Genrator_" in c]
        
        # Create a copy of the dataframe to avoid fragmentation warnings
        df_copy = df.copy()
        
        # Ensure an evaluation column for each Genrator col
        for gc in Genrator_cols:
            eval_col = gc + "_evaluation"
            if eval_col not in df_copy.columns:
                df_copy[eval_col] = None

        st.session_state["df"] = df_copy
        st.session_state["all_Genrator_cols"] = Genrator_cols
        st.session_state["current_row_index"] = 0
        st.session_state["current_field_index"] = 0
        
        # Set initial category if using category view
        if Genrator_cols:
            categories = group_fields_by_category(Genrator_cols)
            if categories:
                st.session_state["current_category"] = next(iter(categories))
        
        # Calculate initial stats
        calculate_evaluation_stats()
        
        return True
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return False


def load_json_file(json_file):
    """
    Load previously saved JSON that contains the entire DataFrame (including evaluations).
    """
    try:
        # Read the file content as string
        if hasattr(json_file, 'read'):
            content = json_file.read().decode('utf-8')
        else:
            content = json_file
            
        # Parse JSON
        data = json.loads(content)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Identify Genrator columns (re-check in case structure changed)
        Genrator_cols = [c for c in df.columns if "_Genrator_" in c]
        
        # Create a copy of the dataframe to avoid fragmentation warnings
        df_copy = df.copy()
        
        # Ensure an evaluation column for each Genrator col
        for gc in Genrator_cols:
            eval_col = gc + "_evaluation"
            if eval_col not in df_copy.columns:
                df_copy[eval_col] = None

        if "DateOfReview" not in df_copy.columns:
            df_copy["DateOfReview"] = None

        st.session_state["df"] = df_copy
        st.session_state["all_Genrator_cols"] = Genrator_cols
        st.session_state["current_row_index"] = 0
        st.session_state["current_field_index"] = 0
        
        # Set initial category if using category view
        if Genrator_cols:
            categories = group_fields_by_category(Genrator_cols)
            if categories:
                st.session_state["current_category"] = next(iter(categories))
        
        # Calculate initial stats
        calculate_evaluation_stats()
        
        return True
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return False


def previous_row():
    if st.session_state["df"] is None:
        return
    if st.session_state["current_row_index"] > 0:
        st.session_state["current_row_index"] -= 1
        st.session_state["current_field_index"] = 0


def next_row():
    if st.session_state["df"] is None:
        return
    max_row = len(st.session_state["df"]) - 1
    if st.session_state["current_row_index"] < max_row:
        st.session_state["current_row_index"] += 1
        st.session_state["current_field_index"] = 0


def update_current_row_index(new_val):
    """
    For the numeric input that sets the row index directly.
    """
    if st.session_state["df"] is None:
        return
    df_len = len(st.session_state["df"])
    if 0 <= new_val < df_len:
        st.session_state["current_row_index"] = new_val
        st.session_state["current_field_index"] = 0


def set_evaluation(label, field_col=None):
    """
    Callback for clicking an evaluation label button.
    This saves the evaluation for the current field, then moves to the next field.
    If field_col is provided, use that specific field instead of the current one.
    """
    if st.session_state["df"] is None:
        return

    df = st.session_state["df"]
    row_idx = st.session_state["current_row_index"]
    if not (0 <= row_idx < len(df)):
        return

    skip_patterns = st.session_state["skip_list"]
    non_skipped_cols = [
        c for c in st.session_state["all_Genrator_cols"]
        if not column_is_skipped(c, skip_patterns)
    ]
    
    if field_col:
        # Use the provided field column
        current_col = field_col
    else:
        # Use the current field index
        fld_idx = st.session_state["current_field_index"]
        if not (0 <= fld_idx < len(non_skipped_cols)):
            return
        current_col = non_skipped_cols[fld_idx]
    
    eval_col = current_col + "_evaluation"
    df.at[row_idx, eval_col] = label

    # If we're using the current field index (not a specific field),
    # and we're in field-by-field view mode, move to the next field
    if not field_col and st.session_state["view_mode"] == "by_field":
        fld_idx = st.session_state["current_field_index"]
        if fld_idx < len(non_skipped_cols) - 1:
            st.session_state["current_field_index"] += 1
    
    # Update evaluation stats
    calculate_evaluation_stats()


def mark_reviewed():
    """
    Callback to mark the entire row as reviewed by setting current timestamp in 'DateOfReview'.
    """
    if st.session_state["df"] is not None:
        row_idx = st.session_state["current_row_index"]
        if 0 <= row_idx < len(st.session_state["df"]):
            st.session_state["df"].at[row_idx, "DateOfReview"] = datetime.datetime.now().isoformat()


def toggle_show_json():
    """
    Toggle whether to show JSON for the current row.
    """
    st.session_state["show_json_for_current_row"] = not st.session_state["show_json_for_current_row"]


def toggle_show_all_fields():
    """
    Toggle whether to show all fields at once.
    """
    st.session_state["show_all_fields"] = not st.session_state["show_all_fields"]


def export_to_excel():
    """
    Generate an Excel file for download from the current DataFrame.
    """
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        
        try:
            # Try to use openpyxl instead of xlsxwriter
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Write the main data
                df.to_excel(writer, index=False, sheet_name="Data")
                
                # Create a summary sheet with evaluation statistics
                stats_df = pd.DataFrame({
                    "Label": list(st.session_state["evaluation_stats"].keys()),
                    "Count": list(st.session_state["evaluation_stats"].values())
                })
                stats_df.to_excel(writer, sheet_name="Evaluation Summary", index=False)
                
                # Add a sheet with metadata
                metadata = pd.DataFrame({
                    "Property": ["Export Date", "Total Records", "Total Fields", "Fields Evaluated"],
                    "Value": [
                        datetime.datetime.now().isoformat(),
                        len(df),
                        len(st.session_state["all_Genrator_cols"]),
                        sum(1 for _, row in df.iterrows() for col in st.session_state["all_Genrator_cols"] 
                            if pd.notna(row.get(f"{col}_evaluation")))
                    ]
                })
                metadata.to_excel(writer, sheet_name="Metadata", index=False)
            
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="evaluation_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Download Excel file</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error exporting to Excel: {str(e)}")
            
            # Fallback to CSV
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="evaluation_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV file instead</a>'
            st.markdown("Excel export failed. You can download as CSV instead:", unsafe_allow_html=True)
            st.markdown(href, unsafe_allow_html=True)


def export_all_json():
    """
    Build EchoReport JSON for all rows and provide it for download.
    """
    if st.session_state["df"] is None:
        return
    
    df = st.session_state["df"]
    all_reports = []
    
    for i, row_data in df.iterrows():
        nested_dict = build_pydantic_dictionary_from_row(row_data)
        try:
            report_obj = EchoReport(**nested_dict)
            # Add metadata to the JSON
            report_json = json.loads(report_obj.json())
            report_json["_metadata"] = {
                "row_index": i,
                "date_of_review": row_data.get("DateOfReview"),
                "evaluations": {
                    col.replace("_Genrator_", ""): row_data.get(f"{col}_evaluation")
                    for col in st.session_state["all_Genrator_cols"]
                    if pd.notna(row_data.get(f"{col}_evaluation"))
                }
            }
            all_reports.append(report_json)
        except Exception as e:
            all_reports.append({
                "_error": str(e),
                "row_index": i,
                "raw_data": nested_dict
            })

    # Add global metadata
    metadata = {
        "export_date": datetime.datetime.now().isoformat(),
        "total_records": len(df),
        "evaluation_summary": st.session_state["evaluation_stats"]
    }
    
    export_data = {
        "metadata": metadata,
        "reports": all_reports
    }

    big_json = json.dumps(export_data, indent=2)
    
    # Create download link
    b64 = base64.b64encode(big_json.encode()).decode()
    filename = f"echo_reports_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON file</a>'
    st.markdown(href, unsafe_allow_html=True)


def export_entire_df_to_json_button():
    """
    Provide a download button for the entire DataFrame as JSON (orient="records"),
    so the user can reload it later to continue evaluation.
    """
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        df_json = df.to_json(orient="records")
        
        # Create download link
        b64 = base64.b64encode(df_json.encode()).decode()
        filename = f"evaluation_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON file</a>'
        st.markdown(href, unsafe_allow_html=True)


def calculate_evaluation_stats():
    """
    Calculate statistics about evaluations across the dataset
    """
    if st.session_state["df"] is None:
        return
    
    df = st.session_state["df"]
    eval_cols = [col for col in df.columns if col.endswith("_evaluation")]
    
    # Count occurrences of each evaluation label
    stats = {label: 0 for label in EVALUATION_LABELS}
    stats["not evaluated"] = 0
    
    for col in eval_cols:
        for val in df[col]:
            if pd.isna(val):
                stats["not evaluated"] += 1
            else:
                stats[val] = stats.get(val, 0) + 1
    
    st.session_state["evaluation_stats"] = stats


def set_category(category):
    """
    Set the current category for category view mode
    """
    st.session_state["current_category"] = category
    # Use st.rerun() instead of experimental_rerun
    st.rerun()


def search_fields(search_term):
    """
    Update search term in session state
    """
    st.session_state["search_term"] = search_term


###############################################################################
# UI Components
###############################################################################
def render_evaluation_buttons(field_col=None):
    """
    Render the evaluation buttons for a field
    If field_col is provided, use that field instead of the current one
    """
    cols = st.columns(len(EVALUATION_LABELS))
    for i, label in enumerate(EVALUATION_LABELS):
        with cols[i]:
            if field_col:
                st.button(label, key=f"eval_{field_col}_{label}", 
                         on_click=set_evaluation, args=(label, field_col))
            else:
                st.button(label, key=f"eval_{label}", 
                         on_click=set_evaluation, args=(label,))


def render_field_value_and_evaluation(field_col, row_data):
    """
    Render a field's value and its current evaluation status
    """
    current_value = row_data[field_col]
    eval_col = field_col + "_evaluation"
    current_eval = row_data[eval_col]
    
    # Format the display name
    display_name = get_field_display_name(field_col)
    
    # Background color based on evaluation
    bg_color = EVALUATION_COLORS.get(current_eval, "#FFFFFF") if pd.notna(current_eval) else "#FFFFFF"
    
    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <p><strong>{display_name}</strong></p>
            <p>Value: {current_value if pd.notna(current_value) else 'None'}</p>
            <p>Evaluation: {current_eval if pd.notna(current_eval) else 'Not evaluated'}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    return render_evaluation_buttons(field_col)


def render_category_fields(category, fields, row_data):
    """
    Render all fields for a specific category
    """
    st.subheader(f"Category: {category}")
    
    for field in fields:
        render_field_value_and_evaluation(field, row_data)


def render_field_by_field_view(non_skipped_cols, row_data):
    """
    Render the field-by-field view for evaluation
    """
    if not non_skipped_cols:
        st.warning("No fields to evaluate after applying skip patterns.")
        return
    
    fld_idx = st.session_state["current_field_index"]
    if fld_idx >= len(non_skipped_cols):
        st.session_state["current_field_index"] = 0
        fld_idx = 0
        
    current_field_col = non_skipped_cols[fld_idx]
    current_value = row_data[current_field_col]
    current_eval_col = current_field_col + "_evaluation"
    current_eval_label = row_data[current_eval_col]
    
    # Field navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if fld_idx > 0:
            if st.button("◀ Previous Field"):
                st.session_state["current_field_index"] -= 1
                st.rerun()  # Use st.rerun() instead of experimental_rerun
    
    with col2:
        st.write(f"**Field {fld_idx+1} of {len(non_skipped_cols)}**")
        
    with col3:
        if fld_idx < len(non_skipped_cols) - 1:
            if st.button("Next Field ▶"):
                st.session_state["current_field_index"] += 1
                st.rerun()  # Use st.rerun() instead of experimental_rerun
    
    # Field display name
    display_name = get_field_display_name(current_field_col)
    st.markdown(f"### {display_name}")
    
    # Current value and evaluation
    st.markdown(
        f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
        f"<p><strong>Extracted value:</strong> {current_value if pd.notna(current_value) else 'None'}</p>"
        f"<p><strong>Current evaluation:</strong> {current_eval_label if pd.notna(current_eval_label) else 'Not evaluated'}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Evaluation buttons
    st.write("### Evaluate:")
    render_evaluation_buttons()


def render_category_view(non_skipped_cols, row_data):
    """
    Render the category-based view for evaluation
    """
    if not non_skipped_cols:
        st.warning("No fields to evaluate after applying skip patterns.")
        return
    
    # Group fields by category
    categories = group_fields_by_category(non_skipped_cols)
    
    # Category selector
    current_category = st.session_state["current_category"]
    if current_category not in categories:
        current_category = next(iter(categories))
        st.session_state["current_category"] = current_category
    
    # Category navigation
    st.write("### Select Category:")
    category_cols = st.columns(min(len(categories), 4))
    for i, (category, _) in enumerate(categories.items()):
        col_idx = i % 4
        with category_cols[col_idx]:
            is_active = category == current_category
            button_style = "primary" if is_active else "secondary"
            if st.button(category, key=f"cat_{category}", type=button_style):
                set_category(category)
                # st.rerun() is called inside set_category
    
    # Show fields for current category
    st.write("---")
    st.write(f"### Category: {current_category}")
    
    for field in categories[current_category]:
        render_field_value_and_evaluation(field, row_data)


def render_search_view(non_skipped_cols, row_data):
    """
    Render a search-based view of fields
    """
    search_term = st.text_input(
        "Search fields:", 
        value=st.session_state["search_term"],
        key="search_input"
    )
    
    if search_term != st.session_state["search_term"]:
        search_fields(search_term)
    
    if search_term:
        # Filter fields by search term
        matching_fields = [
            col for col in non_skipped_cols
            if search_term.lower() in col.lower()
        ]
        
        if not matching_fields:
            st.info(f"No fields match the search term: '{search_term}'")
        else:
            st.write(f"Found {len(matching_fields)} matching fields:")
            for field in matching_fields:
                render_field_value_and_evaluation(field, row_data)
    else:
        st.info("Enter a search term to find specific fields")


def render_all_fields_view_in_sidebar(non_skipped_cols, row_data):
    """
    Render all fields at the bottom of the sidebar for evaluation
    """
    if not non_skipped_cols:
        st.sidebar.warning("No fields to evaluate after applying skip patterns.")
        return
    
    st.sidebar.write("---")
    st.sidebar.subheader("Field Evaluation")
    
    # Group fields by category for better organization
    categories = group_fields_by_category(non_skipped_cols)
    
    for category, fields in categories.items():
        with st.sidebar.expander(f"{category} ({len(fields)} fields)", expanded=False):
            for field in fields:
                current_value = row_data[field]
                eval_col = field + "_evaluation"
                current_eval = row_data[eval_col]
                
                # Format the display name
                display_name = get_field_display_name(field)
                
                # Background color based on evaluation
                bg_color = EVALUATION_COLORS.get(current_eval, "#FFFFFF") if pd.notna(current_eval) else "#FFFFFF"
                
                st.sidebar.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 8px; border-radius: 5px; margin-bottom: 8px;">
                        <p><strong>{display_name}</strong></p>
                        <p style="font-size:0.9em">Value: {current_value if pd.notna(current_value) else 'None'}</p>
                        <p style="font-size:0.9em">Evaluation: {current_eval if pd.notna(current_eval) else 'Not evaluated'}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Evaluation buttons in sidebar
                eval_cols = st.sidebar.columns(len(EVALUATION_LABELS))
                for i, label in enumerate(EVALUATION_LABELS):
                    with eval_cols[i]:
                        st.sidebar.button(
                            label, 
                            key=f"sidebar_eval_{field}_{label}", 
                            on_click=set_evaluation, 
                            args=(label, field)
                        )


def render_all_fields_view(non_skipped_cols, row_data):
    """
    Render all fields at once for evaluation
    """
    if not non_skipped_cols:
        st.warning("No fields to evaluate after applying skip patterns.")
        return
    
    # Group fields by category for better organization
    categories = group_fields_by_category(non_skipped_cols)
    
    for category, fields in categories.items():
        with st.expander(f"{category} ({len(fields)} fields)", expanded=True):
            for field in fields:
                render_field_value_and_evaluation(field, row_data)


def render_evaluation_progress(df, non_skipped_cols, row_idx):
    """
    Render a progress bar showing evaluation completion
    """
    if df is None or row_idx >= len(df):
        return
    
    row_data = df.iloc[row_idx]
    total_fields = len(non_skipped_cols)
    evaluated_fields = sum(
        1 for col in non_skipped_cols
        if pd.notna(row_data.get(f"{col}_evaluation"))
    )
    
    if total_fields > 0:
        progress = evaluated_fields / total_fields
        st.progress(progress)
        st.write(f"**Evaluated: {evaluated_fields}/{total_fields} fields ({progress:.1%})**")


def render_evaluation_stats():
    """
    Render a chart of evaluation statistics
    """
    if not st.session_state["evaluation_stats"]:
        return
    
    stats = st.session_state["evaluation_stats"]
    
    # Create a DataFrame for the chart
    chart_data = pd.DataFrame({
        "Label": list(stats.keys()),
        "Count": list(stats.values())
    })
    
    # Use st.bar_chart
    st.write("### Evaluation Statistics")
    st.bar_chart(chart_data.set_index("Label"))


###############################################################################
# Main app wrapped in a try/except
###############################################################################
def app():
    st.set_page_config(page_title="Echo Report Evaluation", layout="wide")
    init_session_state()

    # Custom CSS for container styling
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

    # ============= SIDEBAR =============
    st.sidebar.title("Echo Report Evaluation")
    
    # Move file upload to sidebar
    st.sidebar.subheader("Load Data")
    
    # Excel file uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"], key="excel_upload")
    if uploaded_file is not None:
        if st.sidebar.button("Load Excel"):
            if load_excel_file(uploaded_file):
                st.sidebar.success("Excel file loaded successfully!")
    
    # JSON file uploader in sidebar
    st.sidebar.write("**Or load previously saved JSON** (with evaluations):")
    json_uploaded = st.sidebar.file_uploader("JSON DataFrame", type=["json"], key="json_upload")
    if json_uploaded is not None:
        if st.sidebar.button("Load JSON"):
            if load_json_file(json_uploaded):
                st.sidebar.success("Successfully loaded JSON for evaluation continuation.")
    
    st.sidebar.write("---")
    st.sidebar.subheader("Configuration")
    
    # 1) Name of the Unstructured Text Column
    st.session_state["unstructured_col"] = st.sidebar.text_input(
        "Unstructured Text Column Name",
        value=st.session_state["unstructured_col"]
    )

    # 2) Configurable column heights
    st.sidebar.subheader("Column Heights")
    st.session_state["evaluation_column_height"] = st.sidebar.number_input(
        "Evaluation Column Height (px)",
        min_value=300,
        max_value=2000,
        value=st.session_state["evaluation_column_height"],
        step=50
    )
    
    st.session_state["text_column_height"] = st.sidebar.number_input(
        "Text Column Height (px)",
        min_value=300,
        max_value=2000,
        value=st.session_state["text_column_height"],
        step=50
    )
    
    # 3) Configurable text-area height
    st.session_state["unstructured_text_height"] = st.sidebar.number_input(
        "Unstructured Text Height (px)",
        min_value=100,
        max_value=2000,
        value=st.session_state["unstructured_text_height"],
        step=50
    )

    # 4) Comma-separated skip patterns
    skip_str = st.sidebar.text_input("Skip columns ending with (comma-separated)", "")
    # Convert user input to a list of patterns
    st.session_state["skip_list"] = [x.strip() for x in skip_str.split(",") if x.strip()]
    
    # 5) View mode selection
    view_mode = st.sidebar.radio(
        "View Mode",
        options=["Field by Field", "By Category", "Search Fields", "Show All Fields"],
        index=0
    )
    # Map radio button selection to session state value
    view_mode_map = {
        "Field by Field": "by_field",
        "By Category": "by_category",
        "Search Fields": "search",
        "Show All Fields": "all_fields"
    }
    st.session_state["view_mode"] = view_mode_map[view_mode]

    # Show evaluation stats in sidebar
    if st.session_state["df"] is not None:
        st.sidebar.write("---")
        with st.sidebar.expander("Evaluation Statistics", expanded=True):
            stats = st.session_state["evaluation_stats"]
            for label, count in stats.items():
                st.sidebar.write(f"**{label}**: {count}")

    # ============= MAIN UI =============
    st.title("Echo Report Evaluation")

    # If no DataFrame is loaded, stop
    if st.session_state["df"] is None:
        st.info("Please upload and load an Excel file or load a JSON from the sidebar to begin.")
        return

    df = st.session_state["df"]
    unstructured_col = st.session_state["unstructured_col"]
    all_gen_cols = st.session_state["all_Genrator_cols"]

    # Filter out columns that end with skip patterns
    skip_patterns = st.session_state["skip_list"]
    non_skipped_cols = [c for c in all_gen_cols if not column_is_skipped(c, skip_patterns)]

    max_row = len(df) - 1
    # Bound checks for the row index
    if st.session_state["current_row_index"] < 0:
        st.session_state["current_row_index"] = 0
    if st.session_state["current_row_index"] > max_row:
        st.session_state["current_row_index"] = max_row

    row_idx = st.session_state["current_row_index"]
    row_data = df.iloc[row_idx]  # safer indexing

    st.write("---")  # line before row nav
    
    # Row navigation
    nav_c1, nav_c2, nav_c3 = st.columns([1,2,1])
    with nav_c1:
        st.button("◀ Previous Row", on_click=previous_row)
    
    with nav_c2:
        # Row selector with progress indicator
        new_index = st.number_input(
            f"Current Row ({row_idx+1}/{len(df)})",
            min_value=0,
            max_value=max_row,
            value=row_idx,
            step=1,
            on_change=update_current_row_index,
            args=(int(st.session_state.get("new_row_index", row_idx)),),
            key="new_row_index"
        )
        
        # Show evaluation progress for this row
        render_evaluation_progress(df, non_skipped_cols, row_idx)

    with nav_c3:
        st.button("Next Row ▶", on_click=next_row)
    
    # Show review date if available
    review_date = row_data.get("DateOfReview")
    if pd.notna(review_date):
        st.info(f"This row was reviewed on: {review_date}")
    
    st.write("---")  # line after row nav

    # Create a two-column layout
    left_col, right_col = st.columns([1, 1])
    
    # Special handling for "Show All Fields" view mode
    if st.session_state["view_mode"] == "all_fields":
        # Left column with scrollable fields
        with left_col.container(height=st.session_state["evaluation_column_height"], border=True):
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button("Mark Row as Reviewed", on_click=mark_reviewed)
            with col2:
                st.button("Show/Hide JSON", on_click=toggle_show_json)
            
            st.subheader("Field Evaluation")
            
            # Group fields by category
            categories = group_fields_by_category(non_skipped_cols)
            
            # Render fields by category
            for category, fields in categories.items():
                with st.expander(f"{category} ({len(fields)} fields)", expanded=True):
                    for field in fields:
                        render_field_value_and_evaluation(field, row_data)
        
        # Right column with unstructured text
        with right_col.container(height=st.session_state["text_column_height"], border=True):
            st.subheader("Unstructured Text")
            if unstructured_col in df.columns:
                unstructured_text = row_data[unstructured_col]
                st.text_area(
                    "Report content",
                    value=str(unstructured_text) if pd.notna(unstructured_text) else "",
                    height=st.session_state["unstructured_text_height"],
                    key="unstructured_text_area"
                )
            else:
                st.error(f"Column '{unstructured_col}' not found in the data.")
                
            # Show JSON if requested
            if st.session_state["show_json_for_current_row"]:
                with st.expander("Pydantic JSON", expanded=True):
                    try:
                        nested_dict = build_pydantic_dictionary_from_row(row_data)
                        report_obj = EchoReport(**nested_dict)
                        st.json(json.loads(report_obj.json()))
                    except Exception as e:
                        st.error(f"Error building EchoReport: {e}")
                        st.json(nested_dict)  # Show the raw dictionary instead
    else:
        # Use regular two-column layout for other view modes
        with left_col.container(height=st.session_state["evaluation_column_height"], border=True):
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button("Mark Row as Reviewed", on_click=mark_reviewed)
            with col2:
                st.button("Show/Hide JSON", on_click=toggle_show_json)
            
            # Show different views based on view mode
            if st.session_state["view_mode"] == "by_field":
                render_field_by_field_view(non_skipped_cols, row_data)
            elif st.session_state["view_mode"] == "by_category":
                render_category_view(non_skipped_cols, row_data)
            elif st.session_state["view_mode"] == "search":
                render_search_view(non_skipped_cols, row_data)
            
            # Show JSON if requested
            if st.session_state["show_json_for_current_row"]:
                with st.expander("Pydantic JSON", expanded=True):
                    try:
                        nested_dict = build_pydantic_dictionary_from_row(row_data)
                        report_obj = EchoReport(**nested_dict)
                        st.json(json.loads(report_obj.json()))
                    except Exception as e:
                        st.error(f"Error building EchoReport: {e}")
                        st.json(nested_dict)  # Show the raw dictionary instead

        with right_col.container(height=st.session_state["text_column_height"], border=True):
            st.subheader("Unstructured Text")
            if unstructured_col in df.columns:
                unstructured_text = row_data[unstructured_col]
                st.text_area(
                    "Report content",
                    value=str(unstructured_text) if pd.notna(unstructured_text) else "",
                    height=st.session_state["unstructured_text_height"],
                    key="unstructured_text_area"
                )
            else:
                st.error(f"Column '{unstructured_col}' not found in the data.")

    # Export options at bottom
    st.write("---")
    st.subheader("Export Options")
    export_c1, export_c2, export_c3 = st.columns([1,1,1])
    with export_c1:
        if st.button("Export to Excel"):
            export_to_excel()
    with export_c2:
        if st.button("Export All Rows to JSON"):
            export_all_json()
    with export_c3:
        if st.button("Export DataFrame as JSON"):
            export_entire_df_to_json_button()


###############################################################################
# Wrap main app in a try-except to catch errors & let user download partial data
###############################################################################
def main():
    try:
        app()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("You can download a JSON of your data so far:")
        if "df" in st.session_state and st.session_state["df"] is not None:
            try:
                df_json = st.session_state["df"].to_json(orient="records")
                b64 = base64.b64encode(df_json.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="partial_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Download partial data as JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as download_error:
                st.error(f"Error creating download link: {download_error}")

if __name__ == "__main__":
    main()