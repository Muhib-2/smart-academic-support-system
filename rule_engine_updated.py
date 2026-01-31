import os
import pandas as pd
from typing import Dict, List, Tuple, Any

# ============================================================
# ✅ LECTURER REQUIREMENT (PART B - EXPERT SYSTEM)
# - Real human expert (AIU) validates rules (in report)
# - More than 10 IF–THEN rules
# - Demonstrate Forward + Backward chaining (can be simple)
# ============================================================


def parse_completed_courses(completed_courses_str: str):
    if pd.isna(completed_courses_str) or not str(completed_courses_str).strip():
        return set()
    return set([c.strip() for c in str(completed_courses_str).split(",")])


class AdvisingRuleEngine:
    """
    Rule-based Academic Advising Expert System.

    NOTE:
    - Rules below are GENERAL advising rules.
    - In your report, you MUST validate these rules with an AIU expert and record their details.
    """

    def __init__(self, courses_csv: str = "data/courses.csv", prereq_csv: str = "data/prerequisites.csv"):
        self.courses_csv = courses_csv
        self.prereq_csv = prereq_csv

        self.courses = None
        self.prereq = None

        # ✅ FIX: load each file only if it exists (never crash)
        if os.path.exists(courses_csv):
            self.courses = pd.read_csv(courses_csv)

        if os.path.exists(prereq_csv):
            self.prereq = pd.read_csv(prereq_csv)

    # ---------- Optional course recommendation helpers ----------
    def _prereqs_for(self, course_code: str):
        if self.prereq is None:
            return set()
        if "course_code" not in self.prereq.columns or "prerequisite_course_code" not in self.prereq.columns:
            return set()

        rows = self.prereq[self.prereq["course_code"] == course_code]
        return set(rows["prerequisite_course_code"].tolist())

    def _recommend_courses(self, student_profile: dict, max_credits: int) -> Tuple[List[Dict[str, Any]], int]:
        if self.courses is None:
            return [], 0

        # Safety: required columns check
        if "course_code" not in self.courses.columns:
            return [], 0

        completed = parse_completed_courses(student_profile.get("completed_courses", ""))

        recommended = []
        for _, row in self.courses.iterrows():
            code = row["course_code"]

            if code in completed:
                continue

            prereqs = self._prereqs_for(code)
            if prereqs.issubset(completed):
                recommended.append(
                    {
                        "course_code": code,
                        "course_name": row.get("course_name", ""),
                        "credits": int(row.get("credits", 3)),
                    }
                )

        selected = []
        total = 0
        for c in recommended:
            if total + c["credits"] <= max_credits:
                selected.append(c)
                total += c["credits"]

        return selected, total

    # ---------- Core Expert System Rules ----------
    def forward_chain(self, facts: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        advice = {
            "max_credits": 18,
            "actions": [],
            "warnings": [],
            "support": [],
        }
        trace: List[str] = []

        cgpa = float(facts.get("cgpa", 0) or 0)
        attendance = float(facts.get("attendance", 100) or 100)
        failed_courses = int(facts.get("failed_courses", 0) or 0)
        predicted_class = str(facts.get("predicted_class", "")).upper()
        stress_flag = bool(facts.get("stress_flag", False))
        working_hours = float(facts.get("working_hours", 0) or 0)

        # Rule 1
        if failed_courses >= 3:
            advice["max_credits"] = min(advice["max_credits"], 12)
            advice["warnings"].append("Multiple failed courses detected. Reduce workload and prioritise retakes.")
            advice["actions"].append("Retake the most critical failed courses first.")
            trace.append("R1: IF failed_courses>=3 THEN max_credits<=12 + retake priority")

        # Rule 2
        if failed_courses >= 1:
            advice["actions"].append("Meet academic advisor to plan retake strategy.")
            trace.append("R2: IF failed_courses>=1 THEN meet academic advisor")

        # Rule 3
        if cgpa < 2.0:
            advice["max_credits"] = min(advice["max_credits"], 12)
            advice["warnings"].append("CGPA below 2.0. You may be at risk of probation; reduce credit load.")
            advice["support"].append("Academic counselling session recommended.")
            trace.append("R3: IF cgpa<2.0 THEN max_credits<=12 + counselling")

        # Rule 4
        if 2.0 <= cgpa < 2.5:
            advice["max_credits"] = min(advice["max_credits"], 15)
            advice["warnings"].append("CGPA between 2.0 and 2.5: keep a balanced workload.")
            trace.append("R4: IF 2.0<=cgpa<2.5 THEN max_credits<=15")

        # Rule 5
        if attendance < 70:
            advice["warnings"].append("Low attendance: this strongly affects performance.")
            advice["actions"].append("Create a weekly attendance plan and track missed classes.")
            trace.append("R5: IF attendance<70 THEN attendance improvement plan")

        # Rule 6
        if attendance < 50:
            advice["max_credits"] = min(advice["max_credits"], 12)
            advice["support"].append("Discuss attendance issues with advisor/counsellor.")
            trace.append("R6: IF attendance<50 THEN max_credits<=12 + advisor/counsellor")

        # Rule 7
        if predicted_class in {"D", "FAIL", "AT-RISK", "LOW"}:
            advice["max_credits"] = min(advice["max_credits"], 12)
            advice["warnings"].append("Predicted low performance category. Prioritise core subjects and support.")
            advice["support"].append("Join a study group / peer tutoring.")
            trace.append("R7: IF predicted_class low THEN max_credits<=12 + tutoring")

        # Rule 8
        if predicted_class in {"C"} and cgpa < 2.5:
            advice["max_credits"] = min(advice["max_credits"], 15)
            trace.append("R8: IF predicted_class=C and cgpa<2.5 THEN max_credits<=15")

        # Rule 9
        if stress_flag:
            advice["support"].append("Stress management: counselling + time management workshop.")
            advice["actions"].append("Reduce non-essential commitments for 2 weeks.")
            trace.append("R9: IF stress_flag THEN counselling + time management")

        # Rule 10
        if working_hours >= 20:
            advice["warnings"].append("Working long hours may reduce study time.")
            advice["actions"].append("Consider reducing working hours during exam weeks.")
            trace.append("R10: IF working_hours>=20 THEN reduce work hours during exams")

        # Rule 11
        if cgpa >= 3.5 and attendance >= 80 and failed_courses == 0:
            advice["max_credits"] = max(advice["max_credits"], 18)
            advice["actions"].append("You may take advanced electives if prerequisites are met.")
            trace.append("R11: IF high performance THEN allow advanced electives")

        # Rule 12 (always)
        advice["actions"].append("Use a weekly study timetable (at least 2 hours per credit).")
        trace.append("R12: ALWAYS recommend study timetable")

        return advice, trace

    def backward_chain(self, goal: str, facts: Dict[str, Any]) -> Tuple[bool, List[str]]:
        trace: List[str] = []
        cgpa = float(facts.get("cgpa", 0) or 0)
        attendance = float(facts.get("attendance", 100) or 100)
        failed_courses = int(facts.get("failed_courses", 0) or 0)
        predicted_class = str(facts.get("predicted_class", "")).upper()
        stress_flag = bool(facts.get("stress_flag", False))

        if goal == "reduce_credits":
            trace.append("Goal: reduce_credits?")
            if failed_courses >= 3:
                trace.append("Proved by R1 (failed_courses>=3).")
                return True, trace
            if cgpa < 2.5:
                trace.append("Proved by R3/R4 (cgpa<2.5).")
                return True, trace
            if attendance < 50:
                trace.append("Proved by R6 (attendance<50).")
                return True, trace
            if predicted_class in {"D", "FAIL", "AT-RISK", "LOW"}:
                trace.append("Proved by R7 (predicted low).")
                return True, trace
            trace.append("Not proved: conditions not met.")
            return False, trace

        if goal == "needs_counselling":
            trace.append("Goal: needs_counselling?")
            if cgpa < 2.0:
                trace.append("Proved by R3 (cgpa<2.0).")
                return True, trace
            if stress_flag:
                trace.append("Proved by R9 (stress_flag=True).")
                return True, trace
            if attendance < 50:
                trace.append("Proved by R6 (attendance<50).")
                return True, trace
            trace.append("Not proved.")
            return False, trace

        if goal == "needs_study_group":
            trace.append("Goal: needs_study_group?")
            if predicted_class in {"D", "FAIL", "AT-RISK", "LOW"}:
                trace.append("Proved by R7 (predicted low).")
                return True, trace
            if failed_courses >= 1:
                trace.append("Proved by R2 (failed_courses>=1 implies extra support).")
                return True, trace
            trace.append("Not proved.")
            return False, trace

        trace.append(f"Unknown goal: {goal}")
        return False, trace

    def advise(self, student_profile: dict, predicted_class: str) -> dict:
        facts = dict(student_profile)
        facts["predicted_class"] = predicted_class

        # ✅ FIX #4: map UCI columns to rule fields
        g3 = facts.get("G3", 0)
        try:
            g3 = float(g3)
        except Exception:
            g3 = 0.0
        facts["cgpa"] = round((g3 / 20.0) * 4.0, 2)

        failures = facts.get("failures", 0)
        try:
            failures = int(failures)
        except Exception:
            failures = 0
        facts["failed_courses"] = failures

        absences = facts.get("absences", 0)
        try:
            absences = float(absences)
        except Exception:
            absences = 0.0
        facts["attendance"] = max(0.0, 100.0 - (absences * 2.0))

        facts.setdefault("stress_flag", False)
        facts.setdefault("working_hours", 0)

        advice, forward_trace = self.forward_chain(facts)

        reduce_credits, bc1 = self.backward_chain("reduce_credits", facts)
        needs_counselling, bc2 = self.backward_chain("needs_counselling", facts)
        needs_study_group, bc3 = self.backward_chain("needs_study_group", facts)

        selected_courses, total_credits = self._recommend_courses(student_profile, advice["max_credits"])

        return {
            "predicted_class": predicted_class,
            "max_credits": advice["max_credits"],
            "warnings": advice["warnings"],
            "actions": advice["actions"],
            "support": advice["support"],
            "selected_courses": selected_courses,
            "total_credits": total_credits,
            "forward_chaining_trace": forward_trace,
            "backward_chaining": {
                "reduce_credits": {"result": reduce_credits, "trace": bc1},
                "needs_counselling": {"result": needs_counselling, "trace": bc2},
                "needs_study_group": {"result": needs_study_group, "trace": bc3},
            },
        }
