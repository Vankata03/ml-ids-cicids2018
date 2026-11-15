# Данни: CSE-CIC-IDS2018

Изследването използва **само оригиналния набор CSE-CIC-IDS2018** на Communications Security Establishment (CSE) и Canadian Institute for Cybersecurity (CIC), University of New Brunswick. Синтетични, генерирани или аугментирани данни не се използват никъде. Автоматичните тестове в `tests/` работят върху копия на първите до 4 000 реални записа от всеки етикет в трите файла.

Наборът е записан в облачна инфраструктура на Amazon Web Services (AWS). Атакуваната организация има 5 отдела, 420 машини и 30 сървъра, а атакуващата инфраструктура е от 50 машини. Записани са 10 дни мрежов трафик (14.02–02.03.2018). От записите (PCAP) с CICFlowMeter-V3 са извлечени по 80 признака на всеки двупосочен поток.

- Описание на набора: https://www.unb.ca/cic/datasets/ids-2018.html
- Публичен достъп (AWS Open Data, без регистрация): `s3://cse-cic-ids2018/` или https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/
- Публикация: Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018, pp. 108–116. https://doi.org/10.5220/0006639801080116

## Използвани файлове

От папката `Processed Traffic Data for ML Algorithms/` се използват три дневни файла:

| Файл | Етикети | Роля |
|---|---|---|
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | `Benign`, `DDOS attack-HOIC`, `DDOS attack-LOIC-UDP` | атака 1 (DDoS) |
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | `Benign`, `FTP-BruteForce`, `SSH-Bruteforce` | атака 2 (BruteForce) |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | `Benign`, `Bot` | атака 3 (ботнет Ares) |

Нормалният трафик (`Benign`) от трите дни се обединява.

Не се използват колоните `Timestamp` (моментът на улавяне не е свойство на трафика) и `Dst Port`. Номерът на порта би позволил на модела да запомни порта на атакуваната услуга (21 за FTP, 22 за SSH), вместо да научи поведението на потока. Остават 77 признака.

## Подготовка

1. Изтеглете трите файла, например:
   ```
   B="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
   for d in Wednesday-21-02-2018 Wednesday-14-02-2018 Friday-02-03-2018; do
     curl -o data/raw/${d}_TrafficForML_CICFlowMeter.csv "$B/${d}_TrafficForML_CICFlowMeter.csv"
   done
   ```
2. Стартирайте `python run_all.py` от корена на хранилището.

Списъкът с прочетените файлове, броят записи по клас и статистиките на почистването се записват автоматично в `results/dataset_stats.json`. Оригиналните файлове съдържат повторени заглавни редове (етикет `Label`). Те не са записи и се пропускат при зареждането.
