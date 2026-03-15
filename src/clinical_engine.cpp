#include "clinical_engine.h"
#include "text_norm.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <regex>

static bool contains_any(const std::string& hay, const std::vector<std::string>& needles) {
    for (const auto& n : needles) {
        if (hay.find(n) != std::string::npos) return true;
    }
    return false;
}

static std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string line;
    while (std::getline(iss, line)) {
        out.push_back(line);
    }
    return out;
}

static void normalize_list(std::vector<std::string>& items) {
    for (auto& it : items) {
        it = normalize_spanish_lower(it);
    }
}

std::string ClinicalEngine::to_lower_ascii(const std::string& s) const {
    return normalize_spanish_lower(s);
}

ObstetricState ClinicalEngine::detect_state(const std::string& user_input) const {
    const std::string u = to_lower_ascii(user_input);
    const bool postpartum = contains_any(u, {
        "posparto", "postparto", "post parto", "puerperio", "alumbramiento",
        "luego de dar a luz", "acaba de dar a luz", "acaba de parir", "pario",
        "recien parida", "puerpera", "puerpera inmediata"
    });
    const bool intraparto = contains_any(u, {
        "trabajo de parto", "contracciones cada", "dilatacion", "borramiento",
        "bolsa rota", "rom", "ruptura de membranas", "expulsivo"
    });
    const bool antenatal = contains_any(u, {
        "embarazada", "gestante", "embarazo", "control prenatal",
        "semanas de gestacion", "semanas de embarazo"
    });

    if (postpartum) return ObstetricState::POSTPARTO;
    if (intraparto) return ObstetricState::INTRAPARTO;
    if (antenatal) return ObstetricState::ANTENATAL;
    return ObstetricState::UNKNOWN;
}

std::string ClinicalEngine::filter_rag_context(const std::string& rag_context, ObstetricState state, bool* removed_incompatible) const {
    if (removed_incompatible) *removed_incompatible = false;
    if (rag_context.empty()) return "";

    std::vector<std::string> blocked;
    if (state == ObstetricState::POSTPARTO) {
        blocked = {
            "feto", "fcf", "doppler", "viabilidad", "rciu",
            "restriccion de crecimiento", "cordon", "embarazo", "aborto",
            "monitorizacion fetal", "ecografia fetal", "embarazo en curso",
            "cardiotocografia", "bienestar fetal", "liquido amniotico"
        };
    } else if (state == ObstetricState::ANTENATAL) {
        blocked = {"posparto", "postparto", "puerperio"};
    }

    auto lines = split_lines(rag_context);
    std::vector<std::string> kept;
    bool removed = false;
    int kept_bullets = 0;
    for (const auto& line : lines) {
        std::string l = to_lower_ascii(line);
        if (!blocked.empty() && contains_any(l, blocked)) {
            removed = true;
            continue;
        }
        kept.push_back(line);
        if (line.rfind("- ", 0) == 0) kept_bullets++;
    }

    if (removed_incompatible) *removed_incompatible = removed;
    if (kept_bullets == 0) return "";

    std::string out;
    for (size_t i = 0; i < kept.size(); ++i) {
        out += kept[i];
        if (i + 1 < kept.size()) out += "\n";
    }
    return out;
}

std::vector<std::string> ClinicalEngine::extract_evidence(const std::string& rag_context, ObstetricState state) const {
    std::vector<std::string> evidence;
    if (rag_context.empty()) return evidence;

    auto lines = split_lines(rag_context);
    std::vector<std::string> keys = {
        "hemorragia posparto", "hemorragia postparto", "atonia", "uteroton",
        "masaje uterino", "4t", "tranexam", "balon uterino", "transfusion",
        "preeclampsia", "eclampsia", "proteinuria", "hipertension", "sulfato de magnesio",
        "sepsis", "antibiotico", "fiebre", "infeccion", "shock",
        "ruptura de membranas", "rom", "parto prematuro", "prematuro",
        "placenta previa", "desprendimiento", "dppni"
    };

    for (const auto& line : lines) {
        if (line.rfind("- ", 0) != 0) continue;
        std::string l = to_lower_ascii(line);
        if (keys.empty() || contains_any(l, keys)) {
            evidence.push_back(line.substr(2));
        }
        if (evidence.size() >= 4) break;
    }
    return evidence;
}

ExtractedData ClinicalEngine::extract(const std::string& user_input, const std::string& rag_context) {
    ExtractedData data;
    const std::string u = to_lower_ascii(user_input);
    const std::string r = to_lower_ascii(rag_context);

    const bool has_postpartum_terms = contains_any(u, {
        "posparto", "postparto", "post parto", "puerperio", "alumbramiento",
        "luego de dar a luz", "acaba de dar a luz", "acaba de parir", "pario",
        "recien parida", "puerpera", "puerpera inmediata"
    });
    const bool has_antenatal_terms = contains_any(u, {
        "embarazada", "gestante", "embarazo", "control prenatal",
        "semanas de gestacion", "semanas de embarazo"
    });

    data.state = detect_state(user_input);
    if (has_postpartum_terms && has_antenatal_terms) {
        data.contradictions.push_back("Entrada mezcla posparto con embarazo en curso; se prioriza posparto.");
        data.state = ObstetricState::POSTPARTO;
    }

    const bool has_bleeding = contains_any(u, {"sangrado", "hemorrag", "sangrando"});
    data.postpartum_bleeding = has_bleeding && (data.state == ObstetricState::POSTPARTO || has_postpartum_terms);
    data.antepartum_bleeding = has_bleeding && (data.state == ObstetricState::ANTENATAL || data.state == ObstetricState::INTRAPARTO);
    if (contains_any(u, {"placenta previa", "desprendimiento", "dppni"})) {
        data.antepartum_bleeding = true;
    }

    if (!data.postpartum_bleeding && (data.state == ObstetricState::POSTPARTO || data.state == ObstetricState::UNKNOWN)) {
        data.postpartum_bleeding =
            contains_any(r, {"hemorragia posparto", "hemorragia postparto", "atonia uterina"});
    }

    data.severe_bleeding = contains_any(u, {"mucho", "abundante", "chorro", "incontrolable", "no para", "empapa", "profuso"});
    data.hypertension_symptoms = contains_any(u, {"vision borrosa", "cefalea", "dolor epigastrico", "presion alta", "hipertension", "fosfenos"});
    data.seizure = contains_any(u, {"convulsion", "convulsiones", "seizure"});

    // Gestational weeks extraction (simple)
    {
        std::regex rgx(R"((\d{1,2})\s*(semanas|sem))");
        std::smatch m;
        if (std::regex_search(u, m, rgx) && m.size() >= 2) {
            data.gest_weeks = std::stoi(m[1].str());
        }
    }

    data.preeclampsia_suspect =
        contains_any(u, {"preeclampsia", "hipertension en el embarazo"}) ||
        (has_antenatal_terms && data.hypertension_symptoms) ||
        contains_any(u, {"proteinuria", "edema", "dolor epigastrico", "cefalea intensa"});

    data.eclampsia_suspect =
        contains_any(u, {"eclampsia"}) ||
        (data.seizure && (has_antenatal_terms || data.state == ObstetricState::INTRAPARTO || data.state == ObstetricState::POSTPARTO));

    data.sepsis_suspect =
        contains_any(u, {"sepsis", "septic", "infeccion generalizada"}) ||
        (contains_any(u, {"fiebre", "escalofrios"}) && contains_any(u, {"mal olor", "dolor uterino", "loquios fetidos", "secrecion fetida"}));

    data.rpm_suspect =
        contains_any(u, {"bolsa rota", "ruptura de membranas", "rom", "liquido amniotico", "salida de liquido"});

    data.preterm_labor_suspect =
        contains_any(u, {"amenaza de parto prematuro", "parto prematuro", "prematuro"}) ||
        (contains_any(u, {"contracciones"}) && data.gest_weeks > 0 && data.gest_weeks < 37);

    if (data.state == ObstetricState::POSTPARTO) {
        data.rpm_suspect = false;
        data.preterm_labor_suspect = false;
        data.antepartum_bleeding = false;
    }

    if (data.severe_bleeding) {
        data.red_flags.push_back("sangrado severo");
    }
    if (data.seizure) {
        data.red_flags.push_back("convulsiones");
    }
    if (data.hypertension_symptoms) {
        data.red_flags.push_back("sintomas de hipertension");
    }
    if (data.sepsis_suspect) {
        data.red_flags.push_back("posible sepsis obstetrica");
    }

    return data;
}

ClinicalPlan ClinicalEngine::plan_postpartum_hemorrhage(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = ObstetricState::POSTPARTO;
    plan.severity = "EMERGENCIA";
    plan.diagnoses.push_back("Hemorragia posparto probable");

    plan.immediate_steps = {
        "Activar protocolo de hemorragia posparto y pedir ayuda inmediata",
        "ABC: oxigeno, monitorizacion, signos vitales y estado hemodinamico",
        "Canalizar 2 vias IV de grueso calibre y comenzar reanimacion con cristaloides",
        "Masaje uterino bimanual y evaluar tono uterino",
        "Aplicar enfoque 4T: tono, trauma, tejido, trombina",
        "Revisar canal del parto y placenta; descartar restos y trauma",
        "Solicitar sangre y pruebas (Hb, coag, grupo y compatibilidad)"
    };

    plan.medications = {
        {"Oxitocina", "segun protocolo local", "IV/IM", "Uterotonico de primera linea"},
        {"Acido tranexamico", "segun protocolo local", "IV", "Terapia adyuvante en HPP"}
    };

    plan.monitoring = {
        "Control de sangrado, tono uterino y signos vitales cada 5-15 min",
        "Balance de liquidos y diuresis horaria"
    };

    plan.escalation = {
        "Si no responde: uterotonicos adicionales segun guia local",
        "Considerar balon uterino, tamponamiento o suturas compresivas",
        "Escalar a equipo quirurgico y activar transfusion masiva si shock"
    };

    plan.questions = {
        "Cantidad aproximada de sangrado?",
        "Hay dolor intenso, fiebre o retencion de placenta?",
        "Signos vitales actuales?"
    };

    if (!rag_context.empty()) {
        plan.warnings.push_back("Plan basado en guias recuperadas por RAG. Verificar con protocolo local.");
    } else {
        plan.warnings.push_back("No se encontro guia especifica; seguir protocolo local de hemorragia posparto.");
    }

    if (data.severe_bleeding) {
        plan.warnings.push_back("Sangrado severo: requiere traslado/atencion inmediata.");
    }

    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"abc", "masaje uterino", "2 vias", "4t"};
    plan.banned_terms = {"feto", "fcf", "doppler", "viabilidad", "ecografia fetal", "cordon", "rciu"};
    return plan;
}

ClinicalPlan ClinicalEngine::plan_eclampsia(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = "EMERGENCIA";
    plan.diagnoses.push_back("Eclampsia probable");

    plan.immediate_steps = {
        "Activar emergencia obstetrica y asegurar ABC",
        "Proteger via aerea, posicion lateral y control de convulsiones",
        "Control estricto de presion arterial",
        "Evaluar signos neurologicos y estado materno",
        "Solicitar laboratorios (plaquetas, creatinina, enzimas hepaticas) y proteinuria"
    };
    if (plan.state != ObstetricState::POSTPARTO) {
        plan.immediate_steps.push_back("Monitoreo materno-fetal segun protocolo local");
    }

    plan.medications = {
        {"Sulfato de magnesio", "segun protocolo local", "IV/IM", "Anticonvulsivante de eleccion"},
        {"Antihipertensivo", "segun protocolo local", "IV/VO", "Control de presion arterial"}
    };

    plan.monitoring = {
        "Signos vitales frecuentes y reflejos",
        "Diuresis horaria y estado neurologico"
    };

    plan.escalation = {
        "Si convulsiones persisten: escalar manejo con equipo de cuidados criticos",
        "Considerar interrupcion del embarazo segun estabilidad materna y edad gestacional"
    };

    plan.questions = {
        "Edad gestacional o semanas?",
        "Presion arterial actual?",
        "Numero y duracion de convulsiones?"
    };

    plan.warnings.push_back("Emergencia vital materna: requiere atencion inmediata.");
    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"abc", "convulsiones", "sulfato de magnesio"};
    if (plan.state == ObstetricState::POSTPARTO) {
        plan.banned_terms = {"feto", "fcf", "doppler", "viabilidad", "cordon"};
    }
    return plan;
}

ClinicalPlan ClinicalEngine::plan_preeclampsia(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = data.hypertension_symptoms ? "URGENTE" : "NO_URGENTE";
    plan.diagnoses.push_back("Preeclampsia probable");

    plan.immediate_steps = {
        "Medir presion arterial y evaluar sintomas de severidad",
        "Solicitar laboratorios (plaquetas, creatinina, enzimas hepaticas) y proteinuria",
        "Evaluar signos de alarma (cefalea intensa, vision borrosa, dolor epigastrico)"
    };
    if (plan.state != ObstetricState::POSTPARTO) {
        plan.immediate_steps.push_back("Monitoreo materno-fetal segun protocolo");
    }

    plan.medications = {
        {"Antihipertensivo", "segun protocolo local", "IV/VO", "Control de presion arterial si severa"}
    };

    plan.monitoring = {
        "Control de presion arterial seriado",
        "Seguimiento de laboratorios segun evolucion"
    };

    plan.escalation = {
        "Si criterios de severidad: manejo hospitalario y posible interrupcion del embarazo",
        "Escalar a especialista en medicina materno-fetal"
    };

    plan.questions = {
        "Presion arterial actual?",
        "Semanas de gestacion?",
        "Hay cefalea intensa, vision borrosa o dolor epigastrico?"
    };

    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"presion arterial", "proteinuria", "laboratorios"};
    if (plan.state == ObstetricState::POSTPARTO) {
        plan.banned_terms = {"feto", "fcf", "doppler", "viabilidad", "cordon"};
    }
    return plan;
}

ClinicalPlan ClinicalEngine::plan_sepsis_obstetrica(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = "EMERGENCIA";
    plan.diagnoses.push_back("Sepsis obstetrica probable");

    plan.immediate_steps = {
        "Activar manejo de sepsis y evaluar ABC",
        "Tomar signos vitales y evaluar estado hemodinamico",
        "Buscar foco infeccioso obstetrico",
        "Solicitar laboratorios y cultivos segun protocolo",
        "Iniciar antibioticos IV empiricos segun guia local"
    };
    if (plan.state != ObstetricState::POSTPARTO) {
        plan.immediate_steps.push_back("Monitoreo materno-fetal segun protocolo");
    }

    plan.medications = {
        {"Antibioticos de amplio espectro", "segun protocolo local", "IV", "Inicio temprano"},
        {"Fluidos IV", "segun protocolo local", "IV", "Resucitacion hemodinamica"}
    };

    plan.monitoring = {
        "Signos vitales frecuentes y diuresis",
        "Control de temperatura y estado mental"
    };

    plan.escalation = {
        "Escalar a cuidados criticos si inestabilidad",
        "Considerar intervencion quirurgica si foco no controlado"
    };

    plan.questions = {
        "Fiebre y tiempo de evolucion?",
        "Hay mal olor o dolor uterino?",
        "Signos vitales actuales?"
    };

    plan.warnings.push_back("Posible sepsis: requiere manejo inmediato.");
    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"antibioticos", "abc", "fluidos"};
    return plan;
}

ClinicalPlan ClinicalEngine::plan_rpm(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = "URGENTE";
    plan.diagnoses.push_back("Ruptura prematura de membranas probable");

    plan.immediate_steps = {
        "Confirmar ruptura de membranas con evaluacion clinica",
        "Determinar edad gestacional y bienestar materno",
        "Vigilar signos de infeccion",
        "Definir conducta (expectante vs induccion) segun semanas y protocolo"
    };

    plan.monitoring = {
        "Temperatura y signos vitales",
        "Evaluacion obstetrica seriada"
    };

    plan.questions = {
        "Cuantas semanas de gestacion?",
        "Salida de liquido continuo o unico episodio?",
        "Fiebre o mal olor?"
    };

    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"ruptura", "edad gestacional", "infeccion"};
    return plan;
}

ClinicalPlan ClinicalEngine::plan_preterm_labor(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = "URGENTE";
    plan.diagnoses.push_back("Amenaza de parto prematuro");

    plan.immediate_steps = {
        "Evaluar contracciones y cambios cervicales",
        "Determinar edad gestacional",
        "Valorar traslado a centro con neonatologia si aplica",
        "Considerar manejo farmacologico segun protocolo local"
    };

    plan.monitoring = {
        "Seguimiento de frecuencia de contracciones",
        "Evaluacion obstetrica seriada"
    };

    plan.questions = {
        "Semanas de gestacion?",
        "Frecuencia e intensidad de contracciones?",
        "Hay sangrado o salida de liquido?"
    };

    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"contracciones", "edad gestacional", "cervical"};
    return plan;
}

ClinicalPlan ClinicalEngine::plan_antepartum_bleeding(const ExtractedData& data, const std::string& rag_context) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = data.severe_bleeding ? "EMERGENCIA" : "URGENTE";
    plan.diagnoses.push_back("Sangrado en embarazo (placenta previa o desprendimiento)");

    plan.immediate_steps = {
        "Evaluar ABC y signos vitales",
        "No tacto vaginal si sospecha placenta previa",
        "Solicitar ecografia para localizacion placentaria",
        "Canalizar vias IV y preparar hemoderivados si sangrado importante"
    };
    if (plan.state != ObstetricState::POSTPARTO) {
        plan.immediate_steps.push_back("Monitoreo materno-fetal segun protocolo");
    }

    plan.monitoring = {
        "Control de sangrado y signos vitales",
        "Evaluacion obstetrica continua"
    };

    plan.escalation = {
        "Escalar a equipo quirurgico si inestabilidad",
        "Considerar interrupcion del embarazo segun estabilidad materna"
    };

    plan.questions = {
        "Cantidad de sangrado?",
        "Dolor abdominal asociado?",
        "Semanas de gestacion?"
    };

    plan.evidence = extract_evidence(rag_context, plan.state);
    plan.required_keywords = {"abc", "ecografia", "tacto vaginal"};
    return plan;
}

ClinicalPlan ClinicalEngine::plan_fallback(const ExtractedData& data) {
    ClinicalPlan plan;
    plan.state = data.state;
    plan.severity = data.red_flags.empty() ? "NO_URGENTE" : "URGENTE";
    plan.diagnoses.push_back("Requiere evaluacion clinica");
    plan.immediate_steps = {
        "Evaluar clinicamente y tomar signos vitales",
        "Confirmar contexto obstetrico (semanas, parto reciente, antecedentes)"
    };
    plan.questions = {
        "Edad gestacional o semanas si aplica?",
        "Sintomas principales y tiempo de inicio?",
        "Signos vitales disponibles?"
    };
    if (!data.red_flags.empty()) {
        plan.warnings.push_back("Hay signos de alarma: " + data.red_flags.front());
    }
    if (plan.state == ObstetricState::POSTPARTO) {
        plan.banned_terms = {"feto", "fcf", "doppler", "viabilidad", "cordon"};
    }
    return plan;
}

ClinicalPlan ClinicalEngine::build_plan(const std::string& user_input, const std::string& rag_context) {
    ExtractedData data = extract(user_input, rag_context);
    bool removed_incompatible = false;
    std::string filtered_rag = filter_rag_context(rag_context, data.state, &removed_incompatible);

    ClinicalPlan plan;
    if (data.eclampsia_suspect) {
        plan = plan_eclampsia(data, filtered_rag);
    } else if (data.postpartum_bleeding) {
        plan = plan_postpartum_hemorrhage(data, filtered_rag);
    } else if (data.sepsis_suspect) {
        plan = plan_sepsis_obstetrica(data, filtered_rag);
    } else if (data.antepartum_bleeding) {
        plan = plan_antepartum_bleeding(data, filtered_rag);
    } else if (data.preeclampsia_suspect) {
        plan = plan_preeclampsia(data, filtered_rag);
    } else if (data.rpm_suspect) {
        plan = plan_rpm(data, filtered_rag);
    } else if (data.preterm_labor_suspect) {
        plan = plan_preterm_labor(data, filtered_rag);
    } else {
        plan = plan_fallback(data);
        plan.evidence = extract_evidence(filtered_rag, data.state);
    }

    if (removed_incompatible) {
        plan.warnings.push_back("Se excluyo contenido incompatible con el estado obstetrico.");
    }
    for (const auto& c : data.contradictions) {
        plan.warnings.push_back(c);
    }

    if (plan.state == ObstetricState::POSTPARTO && plan.banned_terms.empty()) {
        plan.banned_terms = {"feto", "fcf", "doppler", "viabilidad", "cordon", "embarazo en curso"};
    }

    // Normalizar keywords para validacion
    normalize_list(plan.required_keywords);
    normalize_list(plan.banned_terms);
    return plan;
}
