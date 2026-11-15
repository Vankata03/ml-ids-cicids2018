"""Конфигурация на експеримента.

Всички параметри на изследването са събрани тук, за да може експериментът
да бъде възпроизведен точно (фиксиран seed, фиксирани решетки от хиперпараметри).
"""
from pathlib import Path

# --- Пътища ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # оригиналните CSV файлове на CSE-CIC-IDS2018
RAW2017_DIR = DATA_DIR / "raw_cicids2017"  # CSV файловете на коригирания CIC-IDS2017 (за PortScan)
PROCESSED_DIR = DATA_DIR / "processed"     # междинни (почистени) данни
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# --- Възпроизводимост ------------------------------------------------------
SEED = 42

# --- Набор от данни ----------------------------------------------------------
DATASET_2018 = "CSE-CIC-IDS2018"
DATASET_2017 = "CIC-IDS2017"
DATASET_NAME = f"{DATASET_2018} и {DATASET_2017}"
LABEL_COLUMN = "label"
BENIGN_LABEL = "Benign"                       # нормален трафик в CSE-CIC-IDS2018
BENIGN_LABEL_2017 = "Benign (CIC-IDS2017)"    # нормален трафик в CIC-IDS2017
BENIGN_LABELS = {DATASET_2018: BENIGN_LABEL, DATASET_2017: BENIGN_LABEL_2017}

# --- CIC-IDS2017 (коригирана версия на Liu et al., 2022) ----------------------
# CSE-CIC-IDS2018 няма етикет за сканиране на портове, затова атаката PortScan се взема от
# CIC-IDS2017 на същата лаборатория. Използва се коригираната версия на набора
# (https://intrusion-detection.distrinet-research.be/CNS2022/), в която са поправени
# грешките в етикетите и в CICFlowMeter. Тя съдържа същите 77 признака на потока, но с
# по-дълги имена на колоните; тук те се преименуват към имената от CSE-CIC-IDS2018.
CIC2017_FILES = ("friday.csv", "monday.csv", "wednesday.csv")   # 07.07 (PortScan), 03.07 и 05.07.2017
CIC2017_DAYS = {"friday.csv": "Friday-07-07-2017", "monday.csv": "Monday-03-07-2017",
                "wednesday.csv": "Wednesday-05-07-2017"}
CIC2017_LABELS = {"BENIGN": BENIGN_LABEL_2017, "Portscan": "PortScan"}
# Колони без съответствие в CSE-CIC-IDS2018: идентификатори на потока и признаци,
# добавени в по-новата версия на CICFlowMeter.
CIC2017_DROP = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Fwd RST Flags", "Bwd RST Flags",
                "ICMP Code", "ICMP Type", "Total TCP Flow Time", "Attempted Category"]
CIC2017_RENAME = {
    "Total Fwd Packet": "Tot Fwd Pkts", "Total Bwd packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packet": "TotLen Fwd Pkts", "Total Length of Bwd Packet": "TotLen Bwd Pkts",
    "Fwd Packet Length Max": "Fwd Pkt Len Max", "Fwd Packet Length Min": "Fwd Pkt Len Min",
    "Fwd Packet Length Mean": "Fwd Pkt Len Mean", "Fwd Packet Length Std": "Fwd Pkt Len Std",
    "Bwd Packet Length Max": "Bwd Pkt Len Max", "Bwd Packet Length Min": "Bwd Pkt Len Min",
    "Bwd Packet Length Mean": "Bwd Pkt Len Mean", "Bwd Packet Length Std": "Bwd Pkt Len Std",
    "Flow Bytes/s": "Flow Byts/s", "Flow Packets/s": "Flow Pkts/s",
    "Fwd IAT Total": "Fwd IAT Tot", "Bwd IAT Total": "Bwd IAT Tot",
    "Fwd Header Length": "Fwd Header Len", "Bwd Header Length": "Bwd Header Len",
    "Fwd Packets/s": "Fwd Pkts/s", "Bwd Packets/s": "Bwd Pkts/s",
    "Packet Length Min": "Pkt Len Min", "Packet Length Max": "Pkt Len Max",
    "Packet Length Mean": "Pkt Len Mean", "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var",
    "FIN Flag Count": "FIN Flag Cnt", "SYN Flag Count": "SYN Flag Cnt", "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt", "ACK Flag Count": "ACK Flag Cnt", "URG Flag Count": "URG Flag Cnt",
    "CWR Flag Count": "CWE Flag Count", "ECE Flag Count": "ECE Flag Cnt",
    "Average Packet Size": "Pkt Size Avg",
    "Fwd Segment Size Avg": "Fwd Seg Size Avg", "Bwd Segment Size Avg": "Bwd Seg Size Avg",
    "Fwd Bytes/Bulk Avg": "Fwd Byts/b Avg", "Fwd Packet/Bulk Avg": "Fwd Pkts/b Avg",
    "Fwd Bulk Rate Avg": "Fwd Blk Rate Avg", "Bwd Bytes/Bulk Avg": "Bwd Byts/b Avg",
    "Bwd Packet/Bulk Avg": "Bwd Pkts/b Avg", "Bwd Bulk Rate Avg": "Bwd Blk Rate Avg",
    "Subflow Fwd Packets": "Subflow Fwd Pkts", "Subflow Fwd Bytes": "Subflow Fwd Byts",
    "Subflow Bwd Packets": "Subflow Bwd Pkts", "Subflow Bwd Bytes": "Subflow Bwd Byts",
    "FWD Init Win Bytes": "Init Fwd Win Byts", "Bwd Init Win Bytes": "Init Bwd Win Byts",
}

# Официалната дистрибуция („Processed Traffic Data for ML Algorithms“) съдържа по един
# CSV файл за всеки ден от записа, с 80 признака на потоците (CICFlowMeter-V3) и колона Label.
# Файловете без колона за етикет (не се срещат в CSE-CIC-IDS2018) получават етикет по папката.
FOLDER_LABELS: dict = {}

# Колони, които не се използват като признаци:
#   Timestamp - момент на улавяне (идентификатор на сесията, а не свойство на трафика);
#   Dst Port  - номер на порта на жертвата: би позволил на модела да „запомни“ порта на
#               атакуваната услуга (напр. 21/22 при BruteForce) вместо поведението на потока.
DROP_COLUMNS = ["Timestamp", "Dst Port"]

# Самите атаки и алгоритми са описани декларативно в src/registry.py.

# Част от записите, която се зарежда при четене (по префикс на етикета).
# Изтеглен е по един официален CSV файл за всяка DDoS разновидност (~2 млн. записа общо),
# затова се зареждат изцяло (1.0). При работа с всички DDoS файлове (>30 млн. записа)
# стойността може да се намали, напр. 0.05, без промяна на останалия код.
LOAD_FRACTION: dict = {}

# --- Извадка и разделяне ------------------------------------------------------
MAX_PER_CLASS = 100_000   # максимален брой записи на клас в балансираната задача
TEST_SIZE = 0.30          # 70% обучение / 30% тест (оперативна фаза)
NATURAL_TEST_SIZE = 200_000  # размер на допълнителния тест с естествено съотношение (E2)

# --- Предварителна обработка ----------------------------------------------------
VARIANCE_THRESHOLD = 1e-8   # признаци с (почти) нулева дисперсия се премахват
CORR_THRESHOLD = 0.95       # от двойка с |r| > прага се премахва единият признак
PCA_VARIANCE = 0.95         # PCA запазва компонентите, обясняващи 95% от дисперсията

# Основен метод за редукция на размерността на вектора на признаците:
#   "select" - подбор на SELECT_K най-информативни оригинални признака (по обучаващите данни);
#   "pca"    - анализ на главните компоненти (PCA_VARIANCE от дисперсията).
# Другият метод се изчислява като контролен експеримент (E5) за сравнение.
REDUCTION = "select"
SELECT_K = 20               # брой запазени признака (~70 след филтъра -> 20)
SELECT_METHOD = "xgb_gain"  # критерий за важност: среден прираст (gain) в помощен XGBoost
REDUCTION_LABEL = {"select": "Редуциран (подбор на 20 признака)", "pca": "Редуциран (филтър + PCA)"}[REDUCTION]
SELECT_CORR = 1.01          # при подбора корелационният филтър е изключен (подборът сам отстранява излишните)

# --- Модели и подбор на хиперпараметри --------------------------------------------
CV_FOLDS = 5
CV_SCORING = "f1"
SVM_MAX_TRAIN = 50_000      # SVM (RBF) има ~O(n^2) сложност -> ограничена обучаваща извадка
SVM_CV_MAX = 20_000         # извадка за GridSearchCV на SVM (финалното обучение е върху SVM_MAX_TRAIN)
RF_CV_MAX = 50_000          # извадка за GridSearchCV на RF (финалното обучение е върху целия обучаващ набор)
N_JOBS = -1

# Решетките от хиперпараметри са в src/registry.py (AlgorithmSpec.param_grid).

# Оперативна фаза: тестовите записи се подават на пакети, както в IDS
OPERATIONAL_BATCH = 10_000

# Времеви признаци на потока (за контролния експеримент E4): продължителност,
# интервали между пакетите (IAT) и периоди на активност/бездействие.
TIME_LIKE_PATTERNS = ("Duration", "IAT", "Active", "Idle")

# --- Разширени анализи (src/analysis.py) ------------------------------------------
BOOTSTRAP_ROUNDS = 1000          # повторни извадки за 95% доверителни интервали
LEARNING_CURVE_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
PCA_SWEEP = [2, 5, 10, 15, 20, 30, 40]   # брой PCA компоненти / подбрани признаци за анализа на редукцията
SWEEP_TRAIN_MAX = 20_000         # обучаваща извадка за анализите с много модели
SWEEP_TEST_MAX = 10_000          # тестова извадка за анализите с много модели
PERMUTATION_SAMPLES = 2_000      # записи за пермутационната важност
PERMUTATION_REPEATS = 3
PROJECTION_SAMPLES = 3_000       # записи за 2D проекциите (PCA и t-SNE)
TARGET_PRECISIONS = [0.50, 0.80, 0.90, 0.95, 0.99]  # работни точки при естествено съотношение
TARGET_RECALLS = [0.90, 0.95]  # прагове, които гарантират откриване на поне 90% / 95% от атаките
