import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

DTG_PATH = Path(__file__).parent.parent / "data" / "dtg_300.json"

# Thresholds
FRAME_MAP_THRESHOLD = 0.40      # min cosine sim for frame → DTG node activation
FRAME_MAP_TOP_K = 5             # max activated nodes
DTG_K_HOPS = 1                  # BFS radius for activation footprint (k=2 covers all 61 nodes)
K_GAPS = 3                      # coverage gaps to explore per request
CONTAMINATION_THRESHOLD_SP = 0.70   # S(P) contamination: sim(S(P), F) < this
CONTAMINATION_THRESHOLD_RESP = 0.50  # response contamination: sim(R, F) > this
DELTA_EXCLUSION_THRESHOLD = 0.70     # concept in C₀ if sim > this
CONCEPT_MATCH_THRESHOLD = 0.75       # recall match threshold
CAUSAL_PROBE_TOP_N = 10              # top delta concepts to probe
