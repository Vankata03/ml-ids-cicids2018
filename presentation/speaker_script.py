"""Текст за говорене към презентацията.

Създава:
  presentation/script.json  - бележките към всеки слайд (чете ги build_presentation.js);
  presentation/Скрипт_за_говорене_ML_IDS.docx - пълният скрипт в Word.
Изпълнение: python presentation/speaker_script.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_JSON = ROOT / "presentation" / "script.json"
OUT_DOCX = ROOT / "presentation" / "Скрипт_за_говорене_ML_IDS.docx"

# ------------------------------------------------------------------ числа от резултатите
M = pd.read_csv(RES / "metrics.csv").set_index(["attack", "algo"])
MN = pd.read_csv(RES / "metrics_natural_ratio.csv").set_index(["attack", "algo"])
MF = pd.read_csv(RES / "metrics_full_features.csv").set_index(["attack", "algo"])
MT = pd.read_csv(RES / "metrics_without_IAT.csv").set_index(["attack", "algo"])
MP = pd.read_csv(RES / "metrics_other_reduction.csv").set_index(["attack", "algo"])
OF = pd.read_csv(RES / "overfitting.csv")
LC = pd.read_csv(RES / "learning_curves.csv")
PS = pd.read_csv(RES / "pca_sweep.csv")
SUB = pd.read_csv(RES / "ddos_subtypes.csv")
PRE = pd.read_csv(RES / "preliminary_ciciot2023" / "metrics.csv").set_index(["attack", "algo"])
DS = json.load(open(RES / "dataset_stats.json"))
TS = json.load(open(RES / "task_stats.json"))
MC = json.load(open(RES / "mcnemar.json"))
CV = json.load(open(RES / "cv_summary.json"))
RS = json.load(open(RES / "run_summary.json"))
V = json.load(open(ROOT / "presentation" / "figures" / "values.json"))


def i(n):
    return f"{int(round(n)):,}".replace(",", " ")


def p(x, k=1):
    return f"{x * 100:.{k}f}".replace(".", ",") + "%"


def d(x, k=1):
    return f"{x:.{k}f}".replace(".", ",")


ATT, ALG = ["DDoS", "BruteForce", "PortScan"], ["RF", "SVM", "XGB"]
H = "PortScan"   # атаката, показана подробно
acc = lambda a, g: M.loc[(a, g), "Accuracy"]
err = lambda a, g: int(M.loc[(a, g), "FP"] + M.loc[(a, g), "FN"])
speed = lambda a, g: 1e6 / M.loc[(a, g), "predict_time_per_record_us"]
L, CL = DS["load"], DS["clean"]
pre = lambda a, g: PRE.loc[(a, g), "Accuracy"]
min_acc = min(acc(a, g) for a in ATT for g in ALG)
import math
acc_floor = f"{math.floor(min_acc * 1000) / 10:.1f}".replace(".", ",") + "%"
gap_max = OF.gap_accuracy.max()
n_sig = sum(v["p_value"] < 0.05 for v in MC.values())
mc_rs, mc_sx, mc_rx = MC[f"{H}:RF-SVM"], MC[f"{H}:SVM-XGB"], MC[f"{H}:RF-XGB"]
nat_prec = MN.Precision.min()
nb = MN.loc[(H, "XGB")]; nbs = MN.loc[(H, "SVM")]; nbr = MN.loc[(H, "RF")]
nat_prec_2018 = MN.loc[["DDoS", "BruteForce"]].Precision.min()
loic = SUB[(SUB.label == "DDOS attack-LOIC-UDP") & (SUB.algo == "SVM")].iloc[0]
ftp = TS["BruteForce"]["attack_subtype_counts_in_sample"].get("FTP-BruteForce", 0)
lc_svm = LC[(LC.attack == H) & (LC.algo == "SVM")].sort_values("n_train")
lc_xgb = LC[(LC.attack == H) & (LC.algo == "XGB")].sort_values("n_train")
ps_sel5 = PS[(PS.attack == H) & (PS.algo == "XGB") & (PS.method == "select") & (PS.n_components == 5)].F1.iloc[0]
ps_pca5 = PS[(PS.attack == H) & (PS.algo == "XGB") & (PS.method == "pca") & (PS.n_components == 5)].F1.iloc[0]
cvp = lambda g, a: CV[f"{g}_{a}"]["best_params"]
ROB = RES / "robustness"
SF = pd.read_csv(ROB / "split_features.csv")
OD = pd.read_csv(ROB / "other_day_benign.csv")
LO = pd.read_csv(ROB / "held_out_day_benign.csv")
ST = pd.read_csv(ROB / "subtype_transfer.csv")
sf = lambda a, g, sp, fs="all", m="Accuracy": SF[(SF.attack == a) & (SF.algo == g) & (SF.split == sp) & (SF.features == fs)][m].iloc[0]
rnd = SF[(SF.split == "random") & (SF.algo != "XGB|SVM")].pivot_table(index=["attack", "algo"], columns="features", values="Accuracy")
fp_drop = (rnd["all"] - rnd["no_fp"]).max() * 100
one_day = OD[OD.features == "all"].groupby(["attack", "algo"]).FPR.max()
two_day = LO.groupby(["attack", "algo"]).FPR_held_out.max()

K, I, T3 = "Кристиян Януцев", "Иван Терзиев", "Тихомир Божков"

# ------------------------------------------------------------------ текст по слайдове
S = [
 ("Заглавна страница", K, "30 сек.", [
  f"Добър ден. Аз съм Кристиян Януцев, а колегите ми са Иван Терзиев и {T3}. Представяме експериментално изследване на откриването на проникване в мрежовия трафик на център за данни с машинно обучение.",
  "Изследвахме три атаки: сканиране на портове (PortScan), DDoS и Brute Force, с три алгоритъма: Random Forest, SVM и XGBoost. Това са девет модела, обучени само върху реални данни.",
 ]),
 ("Съдържание: път на изследването", K, "30 сек.", [
  f"Това е пътят на изследването в седем стъпки. Аз представям целта и данните. Иван продължава с обработката, моделите, плана и основните резултати. {T3} представя анализа на резултатите и изводите.",
 ]),
 ("Цел и изследователски въпроси", K, "45 сек.", [
  "Целта е да построим и сравним девет модела за откриване на проникване: три алгоритъма за всяка от трите атаки. Оценяваме ги върху записи, които не са виждали, по TP, TN, FP, FN, Accuracy, Precision и Sensitivity, допълнени с F1 и Specificity.",
  "Отговаряме на пет въпроса: колко точно се откриват атаките; как влияе редукцията на признаците; значими ли са разликите и има ли преобучение; какво е качеството при реалния дял на атаките; и от какво зависи постижимата точност.",
 ]),
 ("Избор на атаки и набори от данни", K, "1 мин.", [
  f"Започнахме с IoT набора CICIoT2023. Там Port Scan остана около {p(pre('PortScan', 'XGB'), 0)} при всички подобрения, защото записът описва прозорец от пакети и не носи отпечатъка на сканирането. Затова минахме към наборите на CICFlowMeter, в които един запис описва целия двупосочен поток.",
  "Трите атаки са три етапа на нападението: PortScan е разузнаването, Brute Force е опитът за достъп, а DDoS е отказът от услуга. DDoS и Brute Force са от CSE-CIC-IDS2018, записан в Amazon Web Services с 420 машини и 30 сървъра. На диаграмата вдясно се вижда, че в тези набори и трите атаки се откриват с над 99%.",
 ]),
 ("Данни и почистване", K, "1 мин.", [
  f"Прочетохме шест дневни файла, по три от всеки набор: общо {i(L['rows_read_total'])} записа с {L['n_features_raw']} признака. Махнахме {i(CL['rows_with_nan_or_inf'])} записа с невалидни стойности и {i(CL['duplicates_removed'])} дубликата. Остават {i(CL['rows_after'])} записа.",
  "Дубликатите са много, защото еднаквите потоци се повтарят. Без това почистване един и същ запис щеше да попадне и в обучението, и в теста, и резултатите щяха да са надценени. За всяка атака строим балансирана задача, атака срещу нормален трафик, и я делим 70 на 30.",
 ]),
 ("PortScan: защо втори набор", K, "1 мин.", [
  "CSE-CIC-IDS2018 няма етикет за сканиране на портове. Затова PortScan взехме от CIC-IDS2017, предходния набор на същата лаборатория, в коригираната му версия, в която са поправени грешките в етикетите. Признаците са същите 77, а нормалният трафик е от същия набор, за да не разпознава моделът набора вместо атаката.",
  f"На диаграмата се вижда защо задачата е малка. Пробата на скенера е най-краткият поток: един до три пакета без данни. Затова от {i(TS[H]['attack_rows_original'])} потока остават само {i(TS[H]['attack_rows_available'])} уникални, а тестът е върху {i(TS[H]['test_rows'])} записа. Това ще се види в по-широките интервали на оценките. С това предавам думата на Иван.",
 ]),
 ("Предварителна обработка и подбор на признаци", I, "1 мин.", [
  "Благодаря. Обработката е в два етапа. Първо махаме колоните Timestamp и Dst Port, защото описват кога и къде е бил експериментът, а не поведението на потока. Филтърът по дисперсия маха признаците, които винаги са нула.",
  f"След това обучаваме помощен XGBoost модел и запазваме 20-те признака с най-голяма важност. Така от {L['n_features_raw']} признака остават 20. Всички стъпки са в един Pipeline с класификатора и се научават само от обучаващите записи, така че няма изтичане на информация. PCA е контролен вариант.",
 ]),
 ("Три алгоритъма", I, "1 мин.", [
  "Сравняваме три алгоритъма от различни семейства. Random Forest е ансамбъл от дървета върху случайни извадки и случайни признаци. XGBoost строи дърветата последователно, като всяко ново поправя грешките на предишните, а регуляризацията го пази от преобучение.",
  "SVM с RBF ядро търси нелинейна граница с максимален марж. Обучението му расте поне квадратично с броя записи, затова го обучаваме върху най-много 50 000 записа. В долния ред са избраните хиперпараметри.",
 ]),
 ("Избор на хиперпараметри", I, "45 сек.", [
  "Хиперпараметрите избираме с 5-кратна кръстосана валидация по F1. Не взимаме конфигурацията с най-висока оценка, а прилагаме правилото „една стандартна грешка“: от всички конфигурации, които са статистически неразличими от най-добрата, избираме най-простата.",
  "На диаграмата кръгчето е най-добрата конфигурация, а запълнената точка е избраната. Тя е малко по-ниско, но е по-проста и обобщава по-добре.",
 ]),
 ("Метрики за оценка", I, "45 сек.", [
  "Метриките са от заданието. TP е открита атака, TN е нормален запис без аларма, FP е фалшива аларма, а FN е пропусната атака, най-скъпата грешка.",
  "Тълкуваме ги на 100 случая: Accuracy е колко от 100 решения са верни, Precision е колко от 100 аларми са истински атаки, а Sensitivity, наричана още Recall, е колко от 100 атаки са открити.",
 ]),
 ("Експериментална технологична рамка", I, "30 сек.", [
  f"Рамката е на Python със scikit-learn и XGBoost. Атаките и алгоритмите са записи в регистър, така че нова атака се добавя с един запис. Целият експеримент се пуска с една команда за около {round(RS['total_minutes'])} минути, а всички числа в доклада и в слайдовете се генерират от резултатите.",
 ]),
 ("План на експериментите", I, "45 сек.", [
  "Експериментът има две фази. Във фазата на машинно обучение върху 70% от записите избираме хиперпараметрите, обучаваме финалния модел и го записваме на диска.",
  "В оперативната фаза записаният модел се зарежда и класифицира останалите 30% на пакети по 10 000 записа, както би работила IDS върху поток. Около основния експеримент има контролни експерименти и проверки за приложимост.",
 ]),
 ("Основни резултати", I, "1 мин.", [
  f"Ето основните резултати в оперативната фаза. Всички девет модела имат Accuracy над {acc_floor}, като най-ниската е {p(min_acc, 2)}.",
  f"При PortScan тестът е {i(M.loc[(H, 'RF'), 'n_test'])} записа: Random Forest прави {err(H, 'RF')} грешки, XGBoost {err(H, 'XGB')}, а SVM {err(H, 'SVM')}. Всяка грешка там тежи около една десета от процента. При DDoS Random Forest и XGBoost не правят нито една грешка от {i(M.loc[('DDoS', 'RF'), 'n_test'])} записа, а при BruteForce по една.",
 ]),
 ("Основни резултати: матрици на объркване", I, "45 сек.", [
  "Същите резултати са показани графично като матрици на объркване. Редовете са истинският клас, колоните са решението на модела, а грешките са в червено.",
  f"Червените клетки са малко. Най-много пропуснати атаки има SVM: {int(M.loc[('DDoS', 'SVM'), 'FN'])} при DDoS, {int(M.loc[('BruteForce', 'SVM'), 'FN'])} при BruteForce и {int(M.loc[(H, 'SVM'), 'FN'])} при PortScan.",
 ]),
 ("Проверка за преобучение", I, "45 сек.", [
  "Тъй като резултатите са много високи, проверихме изрично за преобучение. За всеки модел сравняваме точността върху обучаващите данни, оценката от кръстосаната валидация и точността върху теста.",
  f"Трите съвпадат: най-голямата разлика е {d(gap_max * 100, 2)} процентни пункта, при PortScan, и отговаря на два-три тестови записа. Моделите обобщават, а не запомнят. С това предавам думата на {T3}.",
 ]),
 ("Статистическа значимост", T3, "1 мин.", [
  "Благодаря. Значими ли са разликите между алгоритмите? Вляво са 95% доверителни интервали на F1 при PortScan. Интервалите на трите алгоритъма се застъпват.",
  f"Вдясно е тестът на McNemar, който гледа само записите, на които единият модел е прав, а другият греши. При PortScan Random Forest е прав там, където SVM греши, в {mc_rs['only_A_correct']} случая срещу {mc_rs['only_B_correct']}, и това не е значимо: p е {d(mc_rs['p_value'], 2)}. Значими са {n_sig} от 9 сравнения: SVM срещу двата ансамбъла при DDoS и BruteForce, но разликата там е под пет стотни от процента.",
 ]),
 ("Контролни експерименти", T3, "45 сек.", [
  f"Контролните експерименти проверяват решенията в методологията. С всички {L['n_features_raw']} признака точността не се подобрява, значи 20 са достатъчни. С PCA вместо подбор резултатите са близки, но признаците вече не са обясними.",
  "Без времевите признаци DDoS и BruteForce не се променят. При PortScan спадът е до три десети от процента, защото пробата на скенера е много кратка и времето носи част от информацията.",
 ]),
 ("Естествено съотношение на класовете", T3, "1 мин.", [
  f"Досега тестът беше балансиран 50 на 50. В реалната мрежа атаките са по-малко и дори малко фалшиви аларми свалят Precision. Затова повторихме теста в естествения дял на атаките: {p(MN.loc[(H, 'RF'), 'attack_share'], 0)} за PortScan до {p(MN.loc[('DDoS', 'RF'), 'attack_share'], 0)} за DDoS.",
  f"При DDoS и BruteForce Precision остава над {p(nat_prec_2018, 1)}. При PortScan е {p(nbr.Precision, 1)} за Random Forest, {p(nb.Precision, 1)} за XGBoost и {p(nbs.Precision, 1)} за SVM. Тук се вижда проблемът на базовата честота, а SVM е на последно място.",
 ]),
 ("Отпечатъци и хронологично разделяне", T3, "1 мин. 15 сек.", [
  f"Проверихме и дали резултатите са приложими извън случайното разделяне. Първо махнахме признаците, които отразяват мрежовия стек на атакуващия инструмент. Точността пада най-много с {d(fp_drop, 2)} пункта, значи моделите не разчитат само на отпечатъка.",
  f"Второ, обучихме върху по-ранните 70% от потоците и тествахме върху по-късните 30%. DDoS и BruteForce остават над 99,5%. PortScan пада до около {p(sf(H, 'XGB', 'chronological'), 0)}: моделите откриват {p(sf(H, 'XGB', 'chronological', m='Sensitivity'), 0)} от по-късните проби. Причината е, че Nmap сменя режима: в началото пробата е един SYN и един RST, а по-късно скенерът установява пълна връзка и потокът прилича на нормална сесия. Комбинацията от XGBoost и SVM помага малко.",
 ]),
 ("Нормален трафик от непознат ден и пренос", T3, "1 мин.", [
  f"Трета проверка: фалшивите аларми върху нормален трафик от ден, който моделът не е виждал. Ако DDoS моделът е обучен с нормален трафик само от деня на атаката, той вдига аларма за {p(one_day[('DDoS', 'XGB')], 0)} до {p(one_day[('DDoS', 'SVM')], 0)} от нормалния трафик в другите дни. С трафик от два дни алармите падат до {p(two_day[('DDoS', 'SVM')], 2)}. Затова обучаваме с нормален трафик от всички дни.",
  "Проверихме и преноса. Модел за една атака не открива друга, а модел, обучен върху един инструмент, не открива друг инструмент. Обучаващите данни трябва да съдържат всеки инструмент и нормалния трафик на собствената мрежа.",
 ]),
 ("Сравнение на алгоритмите", T3, "45 сек.", [
  f"Обобщаваме. XGBoost е сред най-точните и е най-бърз: над {i(speed('DDoS', 'XGB') // 100000 * 100000)} записа в секунда при DDoS. Random Forest е равностоен по качество и има най-малко фалшиви аларми при PortScan.",
  "SVM изостава малко при всички атаки, вдига най-много фалшиви аларми и изисква подизвадка при обучението. Затова препоръчваме XGBoost или Random Forest.",
 ]),
 ("Ограничения и насоки", T3, "45 сек.", [
  "Ограниченията: данните са от контролирана среда през 2017 и 2018 г. PortScan е от друг набор и остава с малко уникални записи, затова оценките за него са с по-широки интервали. Моделите не се пренасят към нови инструменти и режими на атаката.",
  "Насоките следват от тях: признаци, агрегирани по източник, които биха открили и сканирането с пълна връзка, и многокласов детектор за всички атаки.",
 ]),
 ("Изводи", T3, "1 мин.", [
  f"Основните изводи. Първо, деветте модела постигат Accuracy над {acc_floor} върху реални данни, без преобучение. Второ, PortScan се открива с 99,4 до 99,8%, а DDoS и BruteForce практически без грешка; XGBoost и Random Forest са най-подходящи за практическа IDS.",
  "Трето, при реалния дял на атаките и при хронологично разделяне PortScan е по-трудната задача: обучаващите данни трябва да покриват всички режими на скенера.",
  f"И най-важното: една и съща атака, PortScan, се открива с около {p(pre('PortScan', 'XGB'), 0)} в CICIoT2023 и с над 99% в CIC-IDS2017, с едни и същи алгоритми. Качеството на IDS зависи преди всичко от това дали признаците носят отпечатъка на атаката.",
 ]),
 ("Информационни източници", T3, "15 сек.", [
  "Тук са източниците от доклада, със същата номерация: статиите и страниците на двата набора, статията за корекциите на CIC-IDS2017 и оригиналните статии за трите алгоритъма.",
 ]),
 ("Благодаря за вниманието!", T3, "15 сек.", [
  "С това приключваме. Благодарим ви за вниманието и ще се радваме да отговорим на въпросите ви.",
 ]),
]

# ------------------------------------------------------------------ JSON за бележките
OUT_JSON.write_text(json.dumps([{"title": t, "speaker": sp, "time": tm, "speech": "\n\n".join(ps_)} for t, sp, tm, ps_ in S],
                               ensure_ascii=False, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ Word документ
NAVY, BLUE, GREY = RGBColor(0x14, 0x21, 0x3D), RGBColor(0x1D, 0x5F, 0xA8), RGBColor(0x5B, 0x65, 0x77)
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.2
for name, size, color in [("Title", 24, NAVY), ("Heading 1", 15, NAVY)]:
    h = doc.styles[name]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True; h.font.italic = False
    h.font.color.rgb = color
    rpr = h.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), "Times New Roman")
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
doc.styles["Heading 1"].paragraph_format.space_before = Pt(14)
doc.styles["Heading 1"].paragraph_format.keep_with_next = True


def run(par, text, bold=False, color=None, size=None):
    r = par.add_run(text); r.bold = bold; r.italic = False
    if color: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    return r


doc.add_paragraph("Скрипт за говорене", style="Title")
pp = doc.add_paragraph()
run(pp, "Експериментално изследване на откриване на проникване в мрежов трафик на център за данни, базирано на машинно обучение", True, NAVY, 13)
pp = doc.add_paragraph()
run(pp, f"Иван Терзиев (Fak No. 961326004) · Кристиян Януцев (Fak No. 961326003) · {T3} (Fak No. 961326014)", False, GREY)
pp = doc.add_paragraph()
run(pp, "Разпределение: ", True)
rng_ = lambda who: (min(k for k, x in enumerate(S, 1) if x[1] == who), max(k for k, x in enumerate(S, 1) if x[1] == who))
(k1, k2), (i1_, i2_), (t1, t2) = rng_(K), rng_(I), rng_(T3)
run(pp, f"Кристиян Януцев представя слайдове {k1}–{k2} (цел, избор на атаки и данни). Иван Терзиев представя слайдове {i1_}–{i2_} (обработка, алгоритми, метрики, рамка, план и основни резултати). {T3} представя слайдове {t1}–{t2} (анализ на резултатите и изводи). Общата продължителност е около 20 минути: по около 5 минути за Кристиян, 7 за Иван и 8 за {T3}.")
pp = doc.add_paragraph()
run(pp, "Всички числа в текста са взети автоматично от results/ и съвпадат с доклада и със слайдовете.", False, GREY, 11)

for k, (t, sp, tm, paras) in enumerate(S, 1):
    doc.add_paragraph(f"Слайд {k}. {t}", style="Heading 1")
    meta = doc.add_paragraph()
    run(meta, sp, True, BLUE, 11); run(meta, f"  ·  {tm}", False, GREY, 11)
    for para in paras:
        doc.add_paragraph().add_run(para).italic = False

doc.save(OUT_DOCX)
print(len(S), "слайда ->", OUT_JSON.name, OUT_DOCX.name)
