"""Усі налаштування проєкту в одному місці."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PLOTS_DIR = RESULTS_DIR / "plots"

SEED = 42

# ---- дані ----
IMG_SIZE = 320          # довша сторона зображення після зменшення
PER_CLASS = 160         # скільки зображень кожного класу кешуємо
TEST_PER_CLASS = 14     # з них у спільному тесті (його не бачить жоден воркер)

# ---- федеративне навчання ----
N_TRAIN = 400           # скільки train-зображень у сумі на двох воркерів
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 8
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0

FEDPROX_MU = 0.1
FEDADAM = {"server_lr": 0.005, "beta1": 0.5, "beta2": 0.9, "tau": 1e-3}
IMBALANCE_RATIO = 0.6   # режим C: частка класу k ~ ratio**k

STRATEGIES = ["FedAvg", "FedProx", "FedAdam"]
MODES = [
    "A_equal_balanced",
    "B_unequal_balanced",
    "C_unequal_imbalanced",
    "D_missing_classes",
]
MODE_TITLES = {
    "A_equal_balanced": "A: однакові розміри, збалансовані класи",
    "B_unequal_balanced": "B: різні розміри, збалансовані класи",
    "C_unequal_imbalanced": "C: різні розміри, дисбаланс класів",
    "D_missing_classes": "D: у кожного воркера відсутні ≥2 класи",
}
MODE_SHORT = {
    "A_equal_balanced": "A\nоднакові,\nзбаланс.",
    "B_unequal_balanced": "B\nрізні,\nзбаланс.",
    "C_unequal_imbalanced": "C\nрізні,\nдисбаланс",
    "D_missing_classes": "D\nвідсутні\nкласи",
}
