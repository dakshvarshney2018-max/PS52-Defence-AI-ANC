# PS52_DANC — Candidate Report Documentation Correction Log

**Timestamp:** 2026-09-05 10:35:00 UTC  
**Audit Reference:** `results/candidate_evaluation/FINAL_CANDIDATE_ACCEPTANCE_AUDIT.md`  
**Decision:** ACCEPT WITH CORRECTIONS  
**Status:** COMPLETED

---

## 1. Context & Purpose

Following the read-only acceptance audit (`FINAL_CANDIDATE_ACCEPTANCE_AUDIT.md`), minor documentation inconsistencies were identified in `results/candidate_evaluation/CANDIDATE_EVALUATION_REPORT.md`. 
Specifically:
1. The baseline comparison for the 110 Defence Threat benchmark inadvertently used fallback uninitialized values (+1.52 dB, 0.8967 STOI, 0.1630x RTF) from an evaluation script indexing bug instead of the authoritative frozen values (+3.14 dB, 0.8840 STOI, 0.1664x RTF).
2. The operational RTF claim ambiguously stated that the entire pipeline operated "well within the 0.10x operational budget", whereas 0.0121x RTF applies to the neural component alone, while the end-to-end pipeline operates at 0.1601x / 0.1542x RTF.

This correction log documents the exact textual adjustments applied to align the candidate documentation with the authoritative evidence.

---

## 2. Detailed Summary of Corrections

### A. Defence 110-Scenario Threat Benchmark Baseline Alignment
- **File Edited:** `results/candidate_evaluation/CANDIDATE_EVALUATION_REPORT.md` (Table 2)
- **Authoritative Frozen Baseline Used:** `results/final_evaluation/final_summary.json` (`overall_110_defence_scenarios`) and `results/final_evaluation/pesq/pesq_summary.json`
- **Candidate Numbers Used (Unchanged):** `results/candidate_evaluation/candidate_summary.json` and `results/candidate_evaluation/candidate_defence_results.csv`
| Metric | Erroneous Draft Baseline | Authoritative Frozen Baseline | Candidate Value | Corrected Absolute Delta | Corrected Relative Change | Corrected Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean RTF** | 0.1630x | **0.1664x** | **0.1542x** | **-0.0122x** | **-7.3% (Faster)** | **IMPROVED** |
| **Mean $\Delta$SNR** | +1.52 dB | **+3.14 dB** | **+3.37 dB** | **+0.23 dB** | **+7.3%** | **IMPROVED** |
| **Mean STOI** | 0.8967 | **0.8840** | **0.8870** | **+0.0030** | **+0.34%** | **IMPROVED** |
| **Mean PESQ** | 1.6555 | **1.6555** | **1.6632** | **+0.0077** | **+0.47%** | **IMPROVED** |

*Key finding:* When compared against the genuine frozen defence baseline (0.8840 STOI), the candidate STOI is actually an improvement (+0.0030) rather than a trade-off (-0.0097). All 4 metrics (RTF, $\Delta$SNR, STOI, PESQ) show simultaneous gains across both controlled and defence benchmarks.

### B. RTF Budget & Operational Scope Clarification
- **File Edited:** `results/candidate_evaluation/CANDIDATE_EVALUATION_REPORT.md` (Section 5)
- **Correction:** Clarified that the isolated neural enhancement component operates at **0.0121x RTF** (well within the 0.10x component budget, delivering a 3.18x speedup over the 0.0330x baseline).
- **End-to-End Pipeline RTF:** Retracted the ambiguous statement that the full hybrid pipeline operates within 0.10x. Accurately documented that the integrated end-to-end hybrid pipeline operates at **0.1601x RTF** (controlled, 5.1% faster than 0.1686x) and **0.1542x RTF** (defence, 7.3% faster than 0.1664x), comfortably within real-time streaming constraints (RTF < 1.0x).

---

## 3. Immutability & Safety Verification

1. **Frozen Baseline Verification (`results/final_evaluation/`):**
   - `results/final_evaluation/` and all its subdirectories (`pesq/`) were completely **UNTOUCHED**.
   - No baseline files were modified, overwritten, or regenerated.
2. **Candidate Benchmark Evidence Verification (`results/candidate_evaluation/`):**
   - `candidate_300_results.csv`: Untouched.
   - `candidate_defence_results.csv`: Untouched.
   - `candidate_summary.json`: Untouched.
   - Diagnostic spectrograms and plots: Untouched.
3. **No Algorithmic Changes**:
   - No models were retrained, no controller parameters altered, no dataset definitions touched.
4. **Promotion Status**:
   - Candidate artifacts remain strictly inside `results/candidate_evaluation/`. No files have been promoted to production or copied to `results/final_evaluation/`.
