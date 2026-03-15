#pragma once
#include <string>
#include <vector>

enum class ObstetricState {
    UNKNOWN = 0,
    ANTENATAL,
    INTRAPARTO,
    POSTPARTO
};

struct ExtractedData {
    ObstetricState state = ObstetricState::UNKNOWN;
    bool postpartum_bleeding = false;
    bool severe_bleeding = false;
    bool antepartum_bleeding = false;
    bool hypertension_symptoms = false;
    bool seizure = false;
    bool preeclampsia_suspect = false;
    bool eclampsia_suspect = false;
    bool sepsis_suspect = false;
    bool rpm_suspect = false;
    bool preterm_labor_suspect = false;
    int gest_weeks = -1;
    std::vector<std::string> red_flags;
    std::vector<std::string> contradictions;
};

struct Medication {
    std::string name;
    std::string dose;
    std::string route;
    std::string note;
};

struct ClinicalPlan {
    ObstetricState state = ObstetricState::UNKNOWN;
    std::string severity;
    std::vector<std::string> diagnoses;
    std::vector<std::string> immediate_steps;
    std::vector<Medication> medications;
    std::vector<std::string> monitoring;
    std::vector<std::string> escalation;
    std::vector<std::string> questions;
    std::vector<std::string> warnings;
    std::vector<std::string> evidence;
    std::vector<std::string> required_keywords;
    std::vector<std::string> banned_terms;
};

class ClinicalEngine {
public:
    ClinicalPlan build_plan(const std::string& user_input, const std::string& rag_context);
    ObstetricState detect_state(const std::string& user_input) const;

private:
    ExtractedData extract(const std::string& user_input, const std::string& rag_context);
    ClinicalPlan plan_postpartum_hemorrhage(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_preeclampsia(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_eclampsia(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_sepsis_obstetrica(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_rpm(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_preterm_labor(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_antepartum_bleeding(const ExtractedData& data, const std::string& rag_context);
    ClinicalPlan plan_fallback(const ExtractedData& data);
    std::string filter_rag_context(const std::string& rag_context, ObstetricState state, bool* removed_incompatible) const;
    std::vector<std::string> extract_evidence(const std::string& rag_context, ObstetricState state) const;
    std::string to_lower_ascii(const std::string& s) const;
};
