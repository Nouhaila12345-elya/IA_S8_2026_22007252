# Analyse des Portefeuilles — Question 1.1
## Statistiques Descriptives

**Contexte :** Banque d'investissement — Analyse risques  
**Capital client :** €500,000 | **Perte maximale tolérée :** €50,000 (10%) | **Horizon :** 1 an | **Confiance :** 95%

---

## PORTEFEUILLE A — Conservative
> Actions blue-chip européennes (CAC 40, DAX)

| Indicateur | Formule | Résultat |
|---|---|---|
| **a) Rendement mensuel moyen** | `mean(rendements_A)` | **0.94%** |
| **b) Écart-type mensuel** | `std(rendements_A, ddof=1)` | **0.48%** |
| **c) Médiane** | `median(rendements_A)` | **1.00%** |
| **d) Rendement annualisé** | `(1 + 0.0094)^12 - 1` | **11.85%** |
| **e) Volatilité annualisée** | `0.48 × √12` | **1.65%** |

### Interprétation
- Distribution **quasi-symétrique** : médiane (1.00%) ≈ moyenne (0.94%)
- Volatilité **très faible** : rendements stables et prévisibles
- Ratio Rendement/Risque : **7.17** → excellent

---

## PORTEFEUILLE B — Agressif
> Actions small-cap tech émergentes

| Indicateur | Formule | Résultat |
|---|---|---|
| **a) Rendement mensuel moyen** | `mean(rendements_B)` | **2.89%** |
| **b) Écart-type mensuel** | `std(rendements_B, ddof=1)` | **4.45%** |
| **c) Médiane** | `median(rendements_B)` | **4.70%** |
| **d) Rendement annualisé** | `(1 + 0.0289)^12 - 1` | **40.79%** |
| **e) Volatilité annualisée** | `4.45 × √12` | **15.41%** |

### Interprétation
- Médiane (4.70%) **supérieure** à la moyenne (2.89%) → mois négatifs rares mais très pénalisants
- Volatilité **9x plus élevée** que le Portefeuille A
- Ratio Rendement/Risque : **2.65** → rendement élevé mais risque disproportionné

---

## Comparaison Synthétique

| Indicateur | Portefeuille A | Portefeuille B |
|---|---|---|
| Rendement mensuel moyen | 0.94% | 2.89% |
| Écart-type mensuel | 0.48% | 4.45% |
| Médiane | 1.00% | 4.70% |
| Rendement annualisé | 11.85% | 40.79% |
| Volatilité annualisée | 1.65% | 15.41% |
| **Ratio Rendement/Risque** | **7.17** | **2.65** |

---

## Conclusion Analytique

Le **Portefeuille A** offre une performance régulière et prévisible avec un ratio rendement/risque nettement supérieur (7.17 vs 2.65). Le **Portefeuille B** génère un rendement annualisé de 40.79% mais avec une volatilité annualisée de 15.41%, soit **9 fois plus risqué** que le Portefeuille A.
