# Данни: CSE-CIC-IDS2018 и CIC-IDS2017

Изследването използва **само реално уловен трафик** от два публични набора на Canadian Institute for Cybersecurity (CIC), University of New Brunswick. Синтетични, генерирани или аугментирани данни не се използват никъде. Автоматичните тестове в `tests/` работят върху копия на първите до 4 000 реални записа от всеки етикет във файловете.

| Атака | Набор | Защо |
|---|---|---|
| DDoS | CSE-CIC-IDS2018 | облачен център за данни в AWS |
| Brute Force (FTP, SSH) | CSE-CIC-IDS2018 | облачен център за данни в AWS |
| PortScan | CIC-IDS2017 (коригирана версия) | CSE-CIC-IDS2018 няма етикет за сканиране на портове |

Нормалният трафик на всяка задача е от същия набор като атаката. Записи от двата набора не се смесват в една задача.

## CSE-CIC-IDS2018

Наборът е създаден от Communications Security Establishment (CSE) и CIC и е записан в облачна инфраструктура на Amazon Web Services (AWS). Атакуваната организация има 5 отдела, 420 машини и 30 сървъра, а атакуващата инфраструктура е от 50 машини. Записани са 10 дни мрежов трафик (14.02–02.03.2018). От записите (PCAP) с CICFlowMeter-V3 са извлечени по 80 признака на всеки двупосочен поток.

- Описание на набора: https://www.unb.ca/cic/datasets/ids-2018.html
- Публичен достъп (AWS Open Data, без регистрация): `s3://cse-cic-ids2018/` или https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/
- Публикация: Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018, pp. 108–116. https://doi.org/10.5220/0006639801080116

От папката `Processed Traffic Data for ML Algorithms/` се използват три дневни файла:

| Файл | Използвани етикети | Роля |
|---|---|---|
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | `Benign`, `DDOS attack-HOIC`, `DDOS attack-LOIC-UDP` | атака DDoS |
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | `Benign`, `FTP-BruteForce`, `SSH-Bruteforce` | атака BruteForce |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | `Benign` | нормален трафик от трети ден |

Нормалният трафик (`Benign`) от трите дни се обединява. Записите с етикет `Bot` от третия файл не се използват.

## CIC-IDS2017 (коригирана версия)

CIC-IDS2017 е записан от същата лаборатория в мрежа с 12 машини на жертвата и атакуваща мрежа извън нея, в продължение на 5 дни (03–07.07.2017). Сканирането на портове е проведено с Nmap на 07.07.2017.

Използва се **коригираната версия** на набора, публикувана от Liu, Engelen, Lynar, Essam и Joosen. Авторите откриват грешки в етикетите и в инструмента CICFlowMeter на оригиналния набор, поправят ги и генерират файловете отново от оригиналните PCAP записи. Коригираните файлове съдържат същите 77 признака на потока като CSE-CIC-IDS2018, но с по-дълги имена на колоните. При зареждането те се преименуват към имената от CSE-CIC-IDS2018 (`CIC2017_RENAME` в `src/config.py`).

- Коригирана версия (директно изтегляне, без регистрация): https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip (343 MB)
- Описание на корекциите: https://intrusion-detection.distrinet-research.be/CNS2022/
- Публикация за корекциите: Liu, L., Engelen, G., Lynar, T., Essam, D., Joosen, W. *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018.* IEEE CNS 2022, pp. 254–262.
- Оригинален набор: https://www.unb.ca/cic/datasets/ids-2017.html

От архива се използват три дневни файла:

| Файл | Използвани етикети | Роля |
|---|---|---|
| `friday.csv` (07.07.2017) | `BENIGN`, `Portscan` | атака PortScan |
| `monday.csv` (03.07.2017) | `BENIGN` | нормален трафик от втори ден |
| `wednesday.csv` (05.07.2017) | `BENIGN` | нормален трафик от трети ден |

Останалите етикети в тези файлове (DDoS, Botnet, DoS Hulk и други) не се използват.

## Неизползвани колони

Не се използват колоните `Timestamp` (моментът на улавяне не е свойство на трафика) и `Dst Port`. Номерът на порта би позволил на модела да запомни порта на атакуваната услуга (21 за FTP, 22 за SSH), вместо да научи поведението на потока. От CIC-IDS2017 се премахват и идентификаторите на потока (адреси и портове) и петте признака, които липсват в CSE-CIC-IDS2018. Остават 77 признака.

## Подготовка

1. Изтеглете трите файла на CSE-CIC-IDS2018 в `data/raw/`:
   ```
   B="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
   for d in Wednesday-21-02-2018 Wednesday-14-02-2018 Friday-02-03-2018; do
     curl -o data/raw/${d}_TrafficForML_CICFlowMeter.csv "$B/${d}_TrafficForML_CICFlowMeter.csv"
   done
   ```
2. Изтеглете архива на коригирания CIC-IDS2017 и разархивирайте трите файла в `data/raw_cicids2017/`:
   ```
   mkdir -p data/raw_cicids2017
   curl -L -o data/raw_cicids2017/CICIDS2017_improved.zip "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
   unzip data/raw_cicids2017/CICIDS2017_improved.zip friday.csv monday.csv wednesday.csv -d data/raw_cicids2017
   ```
3. Стартирайте `python run_all.py` от корена на хранилището.

Списъкът с прочетените файлове, броят записи по клас и статистиките на почистването се записват автоматично в `results/dataset_stats.json`. Файловете на CSE-CIC-IDS2018 съдържат повторени заглавни редове (етикет `Label`). Те не са записи и се пропускат при зареждането.
