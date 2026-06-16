# EFA-Bench v1 — Ground-Truth Source Register

Every ground-truth outside-frame concept is backed by a specific, human-authored
source linking the concept to the problem type. Verification status:

- **VERIFIED (web)** — citation existence, authorship, venue, and year confirmed via
  web search on 2026-06-13. Minor page-number refinements applied where the search
  returned exact pagination.
- **CANONICAL** — a foundational, widely-cited reference (named author + standard work)
  that I assert with high confidence from established literature but did not individually
  web-check. These are textbook-level landmarks (e.g., Arrow 1951, Kahneman & Tversky
  1979, Janis 1972). Anyone reproducing the benchmark can confirm them trivially.

No citation here is model-summarized or invented; unverifiable candidates were not
included. Per the integrity protocol, cross-model relevance confirmation is a secondary
check only — a source-backed concept that models deny is **kept and flagged as pro-EFA
evidence**, not discarded.

## Spot-checked (VERIFIED via web, 2026-06-13)

| Problem | Concept | Source | Status |
|---|---|---|---|
| antibiotic_resistance | Selection pressure / fitness | zur Wiesch et al. (2011), *Lancet Infect. Dis.* 11(3):236–247 | VERIFIED |
| committee_choice | Agenda manipulation | McKelvey (1976), *J. Economic Theory* 12(3):472–482 | VERIFIED |
| bridge_collapse | Pedestrian lateral lock-in | Dallard et al. (2001), *The Structural Engineer* 79:17–33 | VERIFIED (pages corrected) |
| startup_growth | Complex contagion | Centola & Macy (2007), *Am. J. Sociology* 113(3):702–734 | VERIFIED |
| traffic_deaths | Safe System / Vision Zero | Belin, Tillgren & Vedung (2012), *Int. J. Injury Control & Safety Promotion* 19(2):171–179 | VERIFIED |
| building_energy | Rebound effect (Jevons) | Sorrell (2009), *Energy Policy* 37(4):1456–1469 | VERIFIED |
| neighborhood_crime | Violence as contagion | Slutkin (2013), in *Contagion of Violence*, National Academies Press | VERIFIED |
| crop_yield | Soil microbiome | van der Heijden, Bardgett & van Straalen (2008), *Ecology Letters* 11(3):296–310 | VERIFIED |
| traffic_deaths | Energy-damage model | Gibson (1961), *Behavioral Approaches to Accident Research* 1:77–89 | VERIFIED |
| forest_fire | Suppression paradox | Arno & Brown (1991), *Western Wildlands* 17:40–46 | VERIFIED (pages corrected) |
| building_energy | Principal-agent / split incentives | Jaffe & Stavins (1994), *Energy Policy* 22(10):804–810 | VERIFIED |

## Canonical references (high confidence, not individually web-checked)

ml_overfit: Grünwald (2007) *The MDL Principle*, MIT Press; Sober (2015) *Ockham's
Razors*, CUP; Wolpert & Macready (1997) *IEEE TEC* 1(1); Engel & Van den Broeck (2001)
*Statistical Mechanics of Learning*, CUP.
traffic_congestion: Duranton & Turner (2011) *AER* 101(6); Braess/Nagurney/Wakolbinger
(2005) *Transportation Science* 39(4); Vickrey (1963) *AER* 53(2).
hospital_readmission: Marmot (2005) *The Lancet* 365; Naylor et al. (2011) *Health
Affairs* 30(4); Green (2006) *Patient Flow*, Springer.
employee_turnover: Mitchell et al. (2001) *AMJ* 44(6); Akerlof & Yellen (1986) CUP;
Granovetter (1973) *AJS* 78(6).
antibiotic_resistance: Hardin (1968) *Science* 162; Ochman, Lawrence & Groisman (2000)
*Nature* 405.
software_outages: Perrow (1984) *Normal Accidents*; Weick & Sutcliffe (2001) *Managing
the Unexpected*; Hollnagel (2014) *Safety-I and Safety-II*.
crop_yield: Lal (2004) *Science* 304; Mundlak (2001) *Handbook of Ag. Economics* Vol.1.
portfolio_risk: Mandelbrot (1963) *J. Business* 36(4); Soros (1987) *The Alchemy of
Finance*; Kahneman & Tversky (1979) *Econometrica* 47(2).
neighborhood_crime: Jeffery (1971) *CPTED*; Cohen & Felson (1979) *ASR* 44(4).
bridge_collapse: Billah & Scanlan (1991) *Am. J. Physics* 59(2); Simiu & Scanlan (1996)
*Wind Effects on Structures*, Wiley.
startup_growth: Rogers (2003) *Diffusion of Innovations* 5th ed.; Katz & Shapiro (1985)
*AER* 75(3).
chronic_back_pain: Woolf (2011) *Pain* 152(3 Suppl); Vlaeyen & Linton (2000) *Pain*
85(3); Engel (1977) *Science* 196.
building_energy: Schultz et al. (2007) *Psychological Science* 18(5).
team_decisions: Janis (1972) *Victims of Groupthink*; Klein (2007) *HBR* 85(9);
Surowiecki (2004) *The Wisdom of Crowds*.
forest_fire: Bond & Keeley (2005) *TREE* 20(7); Bak (1996) *How Nature Works*.
exam_fairness: Hambleton, Swaminathan & Rogers (1991) *Fundamentals of IRT*; Holland &
Wainer (1993) *Differential Item Functioning*; Cronbach & Meehl (1955) *Psych. Bulletin*
52(4).
committee_choice: Arrow (1951) *Social Choice and Individual Values*; Condorcet (1785)
*Essai*.
food_spoilage: Labuza (1980) *Food Technology* 34(4); McMeekin et al. (1993) *Predictive
Microbiology*; Labuza & Saltmarch (1981) in *Water Activity*, Academic Press.
coral_decline: Hoegh-Guldberg et al. (2007) *Science* 318; Hoegh-Guldberg (1999) *Marine
& Freshwater Research* 50(8); Scheffer et al. (2001) *Nature* 413.
warehouse_throughput: Goldratt (1984) *The Goal*; Little (1961) *Operations Research*
9(3); Kingman (1961) *Math. Proc. Cambridge Phil. Soc.* 57(4).
language_learning: Lenneberg (1967) *Biological Foundations of Language*; Krashen (1985)
*The Input Hypothesis*; DeKeyser (2007) *Skill Acquisition Theory*.
