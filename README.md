# ML-IDS върху CSE-CIC-IDS2018: откриване на DDoS, груба сила и ботнет

Курсов проект по „Интелигентни системи за сигурност на екосистемите на големите данни и Интернет на нещата“, 2026/27.

Хранилище в GitHub: https://github.com/Vankata03/ml-ids-cicids2018

Девет модела за откриване на прониквания (3 атаки × 3 алгоритъма), обучени и оценени **само върху реални записи** от публичния набор CSE-CIC-IDS2018 (Communications Security Establishment и Canadian Institute for Cybersecurity), уловен в инфраструктура на Amazon Web Services. Синтетични или генерирани данни не се използват никъде.

| | Random Forest | SVM (RBF ядро) | XGBoost |
|---|---|---|---|
| **DDoS** (HOIC, LOIC-UDP) срещу нормален трафик | модел 1 | модел 2 | модел 3 |
| **BruteForce** (FTP, SSH) срещу нормален трафик | модел 4 | модел 5 | модел 6 |
| **Bot** срещу нормален трафик | модел 7 | модел 8 | модел 9 |

## Основни резултати (оперативна фаза, 30% невиждани данни)

| Модел | TP | TN | FP | FN | Accuracy | Precision | Recall (Sensitivity) | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest – DDoS | 30 000 | 30 000 | 0 | 0 | 100,000% | 100,000% | 100,000% | 100,000% | 100,000% |
| SVM (RBF) – DDoS | 29 988 | 29 982 | 18 | 12 | 99,950% | 99,940% | 99,960% | 99,950% | 99,940% |
| XGBoost – DDoS | 30 000 | 30 000 | 0 | 0 | 100,000% | 100,000% | 100,000% | 100,000% | 100,000% |
| Random Forest – BruteForce | 28 227 | 28 229 | 0 | 1 | 99,998% | 100,000% | 99,996% | 99,998% | 100,000% |
| SVM (RBF) – BruteForce | 28 221 | 28 228 | 1 | 7 | 99,986% | 99,996% | 99,975% | 99,986% | 99,996% |
| XGBoost – BruteForce | 28 227 | 28 229 | 0 | 1 | 99,998% | 100,000% | 99,996% | 99,998% | 100,000% |
| Random Forest – Bot | 29 988 | 29 992 | 8 | 12 | 99,967% | 99,973% | 99,960% | 99,967% | 99,973% |
| SVM (RBF) – Bot | 29 786 | 29 973 | 27 | 214 | 99,598% | 99,909% | 99,287% | 99,597% | 99,910% |
| XGBoost – Bot | 29 994 | 29 995 | 5 | 6 | 99,982% | 99,983% | 99,980% | 99,982% | 99,983% |

Накратко:
- **Данни:** 3 145 725 изтеглени записа от 3 дни, 1 911 561 след премахване на NaN/Inf и 1 226 290 дубликата; 77 признака (без `Timestamp` и `Dst Port`).
- **Всички девет модела** имат Accuracy над 99,5%. DDoS и BruteForce се откриват практически безгрешно; ботнетът е най-трудният.
- **Преобучение няма:** най-голямата разлика между точността при обучение и при тест е 0.016 пункта (`overfitting.csv`). Хиперпараметрите се избират с правилото „една стандартна грешка“.
- **Сравнение:** RF и XGBoost превъзхождат SVM статистически значимо при ботнета (тест на McNemar). XGBoost е най-точен и най-бърз.
- **Естествено съотношение** на класовете (13,8–27,7% атаки): Precision остава над 99.4%.
- **Предварително изследване** върху CICIoT2023: сканирането на портове там не надхвърля около 85% (`results/preliminary_ciciot2023/`), защото признаците на прозорец от пакети не носят отпечатъка на сканирането. Затова е избран CSE-CIC-IDS2018.

Пълният анализ е в доклада (`report/Доклад_ML_IDS_CSE-CIC-IDS2018.docx`, Word), а графиките са в `results/figures/`. Речникът на признаците е в `results/feature_dictionary.md`.

## Структура

```
src/
  registry.py     декларативни регистри на атаките (AttackSpec) и алгоритмите (AlgorithmSpec)
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
tests/            8 автоматични теста върху реални записи (pytest)
models/           деветте обучени конвейера (.joblib)
report/           генератори на доклада: Word (build_docx.py) и LaTeX/PDF (build_report.py)
notebooks/        Colab ноутбук ML_IDS_CSE-CIC-IDS2018.ipynb: реалните данни, моделите и всички диаграми на презентацията
presentation/     генератор на презентацията (.pptx), формулите и скрипта за говорене (Word)
results/          метрики (CSV/JSON), лог и фигури
data/README.md    откъде и как се изтеглят данните
```

Добавянето на нова атака или нов алгоритъм е един запис в `src/registry.py`; останалият код не се променя.

## Възпроизвеждане

1. Изтеглете трите официални CSV файла на CSE-CIC-IDS2018 от публичното хранилище в AWS S3 (подробности в `data/README.md`).
2. `pip install -r requirements.txt`
3. Експеримент:
   ```
   python run_all.py                                        # пълен (около 25 минути)
   python run_all.py --attacks Bot --algos RF XGB --no-extended
   python -m pytest tests
   ```
4. Доклад (изисква XeLaTeX/latexmk и шрифтовете PT Serif/PT Sans):
   ```
   python report/build_report.py           # таблици, стойности и PDF версия
   python report/build_docx.py             # -> report/Доклад_ML_IDS_CSE-CIC-IDS2018.docx (pandoc, python-docx, LibreOffice)
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
- **A10**: без признаците-отпечатъци на TCP стека (`Init Fwd Win Byts`, `Init Bwd Win Byts`, `Fwd Seg Size Min`); точността спада най-много с 0,11 пункта.
- **A11**: хронологично разделяне 70/30 в деня на атаката; DDoS и груба сила остават над 99,5%, при ботнета RF и XGBoost пропускат около 10% от по-късните атаки.
- **A12**: пренос между подвидовете (HOIC ↔ LOIC-UDP, SSH ↔ FTP); моделите не се пренасят между инструменти.
- **A13**: фалшиви аларми върху нормален трафик от непознат ден; нормален трафик от повече дни намалява алармите при DDoS от 16–33% до под 0,13%.
- **A14**: комбиниран детектор „XGBoost или SVM“; при хронологично разделяне за ботнета открива 99,98% от атаките.

Резултатите от A10–A14 са в `results/robustness/`.

## Източник на данните

Sharafaldin I., Lashkari A. H., Ghorbani A. A. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP, 108–116, 2018. Набор: https://www.unb.ca/cic/datasets/ids-2018.html
