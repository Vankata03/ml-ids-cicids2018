# ML-IDS: откриване на PortScan, DDoS и груба сила с машинно обучение

Курсов проект по „Интелигентни системи за сигурност на екосистемите на големите данни и Интернет на нещата“, 2026/27.

Хранилище в GitHub: https://github.com/Vankata03/ml-ids-cicids2018

Девет модела за откриване на прониквания (3 атаки × 3 алгоритъма), обучени и оценени **само върху реални записи** от два публични набора на Canadian Institute for Cybersecurity. Синтетични или генерирани данни не се използват никъде.

| | Набор | Random Forest | SVM (RBF ядро) | XGBoost |
|---|---|---|---|---|
| **PortScan** (Nmap) срещу нормален трафик | CIC-IDS2017, коригирана версия | модел 1 | модел 2 | модел 3 |
| **DDoS** (HOIC, LOIC-UDP) срещу нормален трафик | CSE-CIC-IDS2018 | модел 4 | модел 5 | модел 6 |
| **BruteForce** (FTP, SSH) срещу нормален трафик | CSE-CIC-IDS2018 | модел 7 | модел 8 | модел 9 |

CSE-CIC-IDS2018 е записан в облачна инфраструктура на Amazon Web Services, но няма етикет за сканиране на портове. Затова PortScan е от CIC-IDS2017 на същата лаборатория, в коригираната версия на Liu и др. (2022). Двата набора имат едни и същи 77 признака на потока. Нормалният трафик на всяка задача е от набора на атаката.

## Основни резултати (оперативна фаза, 30% невиждани данни)

| Модел | TP | TN | FP | FN | Accuracy | Precision | Recall (Sensitivity) | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest – PortScan | 504 | 504 | 1 | 1 | 99,802% | 99,802% | 99,802% | 99,802% | 99,802% |
| SVM (RBF) – PortScan | 501 | 503 | 2 | 4 | 99,406% | 99,602% | 99,208% | 99,405% | 99,604% |
| XGBoost – PortScan | 504 | 503 | 2 | 1 | 99,703% | 99,605% | 99,802% | 99,703% | 99,604% |
| Random Forest – DDoS | 30 000 | 30 000 | 0 | 0 | 100,000% | 100,000% | 100,000% | 100,000% | 100,000% |
| SVM (RBF) – DDoS | 29 988 | 29 982 | 18 | 12 | 99,950% | 99,940% | 99,960% | 99,950% | 99,940% |
| XGBoost – DDoS | 30 000 | 30 000 | 0 | 0 | 100,000% | 100,000% | 100,000% | 100,000% | 100,000% |
| Random Forest – BruteForce | 28 227 | 28 229 | 0 | 1 | 99,998% | 100,000% | 99,996% | 99,998% | 100,000% |
| SVM (RBF) – BruteForce | 28 221 | 28 228 | 1 | 7 | 99,986% | 99,996% | 99,975% | 99,986% | 99,996% |
| XGBoost – BruteForce | 28 227 | 28 229 | 0 | 1 | 99,998% | 100,000% | 99,996% | 99,998% | 100,000% |

Накратко:
- **Данни:** 4 561 547 прочетени записа от 6 дневни файла, 2 662 013 след премахване на NaN/Inf и 1 327 997 дубликата; 77 признака (без `Timestamp` и `Dst Port`).
- **Всички девет модела** имат Accuracy над 99,4%. DDoS и BruteForce се откриват практически безгрешно.
- **PortScan е малка задача:** от 159 066 проби на скенера остават 1 682 уникални вектора, така че тестът е върху 1 010 записа. Разликите между алгоритмите там не са статистически значими (тест на McNemar).
- **Преобучение няма:** най-голямата разлика между точността при обучение и при тест е 0,255 пункта (`overfitting.csv`). Хиперпараметрите се избират с правилото „една стандартна грешка“.
- **Естествено съотношение** на класовете (14,0–27,7% атаки): Precision остава над 99% за DDoS и BruteForce и е 93,1%–98,2% за PortScan.
- **Хронологично разделяне:** DDoS и BruteForce остават над 99,5%. При PortScan моделите откриват около 86% от по-късните проби, защото скенерът минава към режими с пълна TCP връзка, които липсват в ранните данни.
- **Предварително изследване** върху CICIoT2023: сканирането на портове там не надхвърля около 85% (`results/preliminary_ciciot2023/`), защото записът описва прозорец от пакети, а не цял поток. Същата атака в CIC-IDS2017 се открива с над 99%.

Пълният анализ е в доклада (`report/Доклад_ML_IDS_CSE-CIC-IDS2018.docx`, Word, и PDF до него), а графиките са в `results/figures/`. Речникът на признаците е в `results/feature_dictionary.md`.

## Структура

```
src/
  registry.py     декларативни регистри на атаките (AttackSpec, с набора на всяка атака) и алгоритмите (AlgorithmSpec)
  config.py       параметри на експеримента (seed 42, k = 20 признака, размери на извадките)
  preprocess.py   зареждане, почистване, бинарни задачи, FeatureFilter, ImportanceSelector
  train.py        конвейер Filter → подбор → StandardScaler → RF/SVM/XGBoost, GridSearchCV (5-CV, правило 1-SE)
  evaluate.py     оперативна фаза (зареждане на модела, пакети по 10 000) и метрики
  stats.py        bootstrap доверителни интервали, тест на McNemar
  analysis.py     разширени анализи A1–A9
  robustness.py   проверки за приложимост A10–A14 (python -m src.robustness)
  experiment.py   клас Experiment: оркестрира целия експеримент, контролни E1–E5
  plots.py, plots_ext.py, diagram.py   фигурите на доклада
run_all.py        команден ред (избор на атаки, алгоритми, --no-extended)
tests/            10 автоматични теста върху реални записи (pytest)
models/           деветте обучени конвейера (.joblib)
report/           генератори на доклада: таблици и стойности (build_report.py), Word и PDF (build_docx.py), номерация без LaTeX (make_aux.py)
notebooks/        Colab ноутбук ML_IDS_CSE-CIC-IDS2018.ipynb: реалните данни, моделите и всички диаграми на презентацията
presentation/     генератор на презентацията (.pptx), формулите и скрипта за говорене (Word)
results/          метрики (CSV/JSON), лог и фигури
data/README.md    откъде и как се изтеглят данните
```

Добавянето на нова атака или нов алгоритъм е един запис в `src/registry.py`; останалият код не се променя.

## Възпроизвеждане

1. Изтеглете трите дневни CSV файла на CSE-CIC-IDS2018 от публичното хранилище в AWS S3 и трите на коригирания CIC-IDS2017 (подробности в `data/README.md`).
2. `pip install -r requirements.txt` (под Windows задайте `PYTHONUTF8=1`, за да се четат файловете с кирилица като UTF-8)
3. Експеримент:
   ```
   python run_all.py                                        # пълен (около 17 минути на 16 ядра)
   python run_all.py --attacks PortScan --algos RF XGB --no-extended
   python -m pytest tests
   ```
4. Доклад (pandoc или `pip install pypandoc-binary`, python-docx, pymupdf и LibreOffice или Microsoft Word; XeLaTeX не е задължителен):
   ```
   python report/build_report.py --no-pdf  # таблици и стойности
   python report/build_docx.py             # -> report/Доклад_ML_IDS_CSE-CIC-IDS2018.docx и .pdf
   ```
5. Ноутбук с диаграмите (локално или в Google Colab, вж. първата клетка):
   ```
   jupyter nbconvert --to notebook --execute --inplace notebooks/ML_IDS_CSE-CIC-IDS2018.ipynb   # -> presentation/figures/
   ```
   Ноутбукът се генерира от `notebooks/notebook_src.py` с `python notebooks/py2ipynb.py notebooks/notebook_src.py notebooks/ML_IDS_CSE-CIC-IDS2018.ipynb`.
6. Презентация и скрипт за говорене (Node.js с `pptxgenjs`):
   ```
   python presentation/render_formulas.py
   python presentation/speaker_script.py   # -> presentation/script.json и Скрипт_за_говорене_ML_IDS.docx
   node presentation/build_presentation.js presentation/Презентация_ML_IDS_CSE-CIC-IDS2018.pptx <път до apply_theme.js>
   ```

Всички случайни процеси използват seed 42, така че метриките се възпроизвеждат точно. Времената зависят от машината.

## Експерименти и анализи

- **E1**: същите модели с пълния вектор от 77 признака, без редукция (`metrics_full_features.csv`).
- **E2**: тест при естественото съотношение на класовете (`metrics_natural_ratio.csv`).
- **E3**: стабилност при 5-кратна кръстосана валидация (`cv_summary.json`, `cv_*.csv`).
- **E4**: модели без времевите признаци на потока: Duration, IAT, Active, Idle (`metrics_without_IAT.csv`).
- **E5**: PCA вместо подбор на признаци (`metrics_other_reduction.csv`).
- **A1**: профил на данните и разделимост на признаците (тест на Колмогоров–Смирнов).
- **A2**: двумерни проекции PCA и t-SNE.
- **A3**: криви на обучение.
- **A4**: зависимост на качеството от броя запазени признаци или PCA компоненти (`pca_sweep.csv`).
- **A5**: Precision–Recall криви и работни точки на прага (`threshold_operating_points.csv`, `threshold_recall_points.csv`).
- **A6**: откриване по подвидове (HOIC, LOIC-UDP, FTP, SSH).
- **A7**: пренос между атаките.
- **A8**: пермутационна важност на признаците.
- **A9**: bootstrap доверителни интервали (1000 повторения) и тест на McNemar за всяка двойка алгоритми.
- **A10**: без признаците-отпечатъци на TCP стека (`Init Fwd Win Byts`, `Init Bwd Win Byts`, `Fwd Seg Size Min`); точността спада най-много с 0,10 пункта.
- **A11**: хронологично разделяне 70/30 в деня на атаката; DDoS и груба сила остават над 99,5%, при PortScan моделите пропускат около 14% от по-късните проби.
- **A12**: пренос между подвидовете (HOIC ↔ LOIC-UDP, SSH ↔ FTP); моделите не се пренасят между инструменти. PortScan няма подвидове в етикетите.
- **A13**: фалшиви аларми върху нормален трафик от непознат ден на същия набор; нормален трафик от повече дни намалява алармите при DDoS от 15–23% до 0,12%, а при PortScan те остават 0,2–1,6%.
- **A14**: комбиниран детектор „XGBoost или SVM“; при хронологично разделяне за PortScan открива 89,5% от пробите, малко повече от XGBoost (85,9%).

Резултатите от A10–A14 са в `results/robustness/`.

## Източници на данните

- Sharafaldin I., Lashkari A. H., Ghorbani A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP, 108–116, 2018. Набори: https://www.unb.ca/cic/datasets/ids-2018.html и https://www.unb.ca/cic/datasets/ids-2017.html
- Liu L., Engelen G., Lynar T., Essam D., Joosen W. *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018.* IEEE CNS, 254–262, 2022. Коригирани набори: https://intrusion-detection.distrinet-research.be/CNS2022/
