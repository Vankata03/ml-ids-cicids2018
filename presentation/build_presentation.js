// Генерира презентацията (PowerPoint) от резултатите в ../results и диаграмите от ноутбука.
// Изпълнение (от корена на хранилището):
//   jupyter nbconvert --to notebook --execute --inplace notebooks/ML_IDS_CSE-CIC-IDS2018.ipynb   # диаграмите
//   python presentation/render_formulas.py                                                         # формулите
//   python presentation/speaker_script.py                                                          # текстът за говорене
//   node presentation/build_presentation.js [изход.pptx] [apply_theme.js]
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const RES = path.join(ROOT, 'results');
const FIG = path.join(__dirname, 'figures');
const RFIG = path.join(RES, 'figures');
const FORM = path.join(__dirname, 'formulas');
const OUT = process.argv[2] || path.join(__dirname, 'Презентация_ML_IDS_CSE-CIC-IDS2018.pptx');
const APPLY = process.argv[3];
const THIRD = 'Тихомир Божков';
const THIRD_FN = 'Fak No. 961326014';

// ---------------------------------------------------------------- данни
const parseCsvLine = line => { const out = []; let cur = '', q = false; for (const ch of line) { if (ch === '"') q = !q; else if (ch === ',' && !q) { out.push(cur); cur = ''; } else cur += ch; } out.push(cur); return out; };
const readCsv = f => {
  const [h, ...rows] = fs.readFileSync(path.join(RES, f), 'utf8').trim().split(/\r?\n/);
  const H = parseCsvLine(h);
  return rows.map(r => { const v = parseCsvLine(r); const o = {}; H.forEach((k, i) => { o[k] = isNaN(+v[i]) || v[i] === '' ? v[i] : +v[i]; }); return o; });
};
const J = f => JSON.parse(fs.readFileSync(path.join(RES, f), 'utf8'));
const M = readCsv('metrics.csv'), MN = readCsv('metrics_natural_ratio.csv'), OF = readCsv('overfitting.csv');
const PRE = readCsv('preliminary_ciciot2023/metrics.csv');
const SF = readCsv('robustness/split_features.csv'), LO = readCsv('robustness/held_out_day_benign.csv'), OD = readCsv('robustness/other_day_benign.csv');
const sfv = (a, g, sp, m = 'Accuracy', fs = 'all') => SF.find(r => r.attack === a && r.algo === g && r.split === sp && r.features === fs)[m];
const odMax = (a, g) => Math.max(...OD.filter(r => r.attack === a && r.algo === g && r.features === 'all').map(r => r.FPR));
const loMax = (a, g) => Math.max(...LO.filter(r => r.attack === a && r.algo === g).map(r => r.FPR_held_out));
const DS = J('dataset_stats.json'), TS = J('task_stats.json'), MC = J('mcnemar.json'), CV = J('cv_summary.json'), RS = J('run_summary.json');
const ENV = J('environment.json');
const V = JSON.parse(fs.readFileSync(path.join(FIG, 'values.json'), 'utf8'));
const NOTES = JSON.parse(fs.readFileSync(path.join(__dirname, 'script.json'), 'utf8'));
const get = (T, a, g) => T.find(r => r.attack === a && r.algo === g);
const int = n => Math.round(n).toLocaleString('en-US').replace(/,/g, ' ');
const pct = (x, k = 1) => (x * 100).toFixed(k).replace('.', ',') + '%';
const dec = (x, k = 1) => (+x).toFixed(k).replace('.', ',');
const ATT = ['DDoS', 'BruteForce', 'PortScan'], ALG = ['RF', 'SVM', 'XGB'];
const ATT_N = { DDoS: 'DDoS', BruteForce: 'BruteForce', PortScan: 'PortScan' };
const HARD = 'PortScan';   // атаката, показана подробно в диаграмите за един модел
const ALG_N = { RF: 'Random Forest', SVM: 'SVM (RBF)', XGB: 'XGBoost' };
const L = DS.load, CL = DS.clean;
const speed = x => 1e6 / x.predict_time_per_record_us;
const minAcc = Math.min(...M.map(r => r.Accuracy));
const accFloor = (Math.floor(minAcc * 1000) / 10).toFixed(1).replace('.', ',') + '%';
const natMin = a => Math.min(...MN.filter(r => r.attack === a).map(r => r.Precision));
const natMax = a => Math.max(...MN.filter(r => r.attack === a).map(r => r.Precision));
const errs = (a, g) => get(M, a, g).FP + get(M, a, g).FN;
const gapMax = Math.max(...OF.map(r => r.gap_accuracy));
const nSig = Object.values(MC).filter(v => v.p_value < 0.05).length;
const natPrec = Math.min(...MN.map(r => r.Precision));
const preAcc = (a, g) => PRE.find(r => r.attack === a && r.algo === g).Accuracy;
const bp = (g, a) => CV[`${g}_${a}`].best_params;

// ---------------------------------------------------------------- тема и оформление
// Академично оформление: бял фон, заглавие вляво с тънка линия под него, номерирани фигури и
// таблици с надписи, таблици в стил booktabs (само хоризонтални линии), без цветни блокове.
const THEME = {
  name: 'IDS Academic', headFontFace: 'Times New Roman', bodyFontFace: 'Times New Roman',
  colors: { dk1: '1F2430', lt1: 'FFFFFF', dk2: '14213D', lt2: 'F4F5F7', accent1: '1D5FA8', accent2: '14A38B', accent3: 'B5282E',
    accent4: 'C98A12', accent5: '5F6673', accent6: 'C9CED6', hlink: '1D5FA8', folHlink: '5F6673' },
};
const HEX = THEME.colors;
const INK = HEX.dk2, GREY = '5F6673', RULE = '14213D', LIGHT = 'C9CED6';
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13,333 x 7,5
pres.title = 'Откриване на проникване с машинно обучение (CSE-CIC-IDS2018 и CIC-IDS2017)';
pres.author = 'Иван Терзиев, Кристиян Януцев, ' + THIRD;
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, X0 = 0.65, XW = 12.03;

pres.defineSlideMaster({
  title: 'CONTENT', background: { color: 'FFFFFF' },
  objects: [
    { line: { x: X0, y: 1.22, w: XW, h: 0, line: { color: RULE, width: 1 } } },
    { placeholder: { options: { name: 'title', type: 'title', x: X0, y: 0.42, w: 10.2, h: 0.72, fontSize: 26, bold: true, color: C.text2, valign: 'bottom', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'sec', type: 'body', x: 9.4, y: 0.42, w: 3.28, h: 0.72, fontSize: 13, color: GREY, valign: 'bottom', align: 'right', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({ title: 'BLANK', background: { color: 'FFFFFF' }, objects: [] });
pres.defineSlideMaster({
  title: 'COVER', background: { color: '0A1424' },
  objects: [
    { image: { path: path.join(FIG, 'p00_title.png'), x: 0, y: 0, w: W, h: 7.5 } },
    { placeholder: { options: { name: 'title', type: 'title', x: 0.75, y: 1.55, w: 7.4, h: 2.6, fontSize: 32, bold: true, color: C.background1, valign: 'bottom', align: 'left', margin: 0, lineSpacingMultiple: 1.05 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.75, y: 4.35, w: 7.0, h: 0.9, fontSize: 18, color: 'C9D3E1', valign: 'top', align: 'left', margin: 0 }, text: '' } },
  ],
});

const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: THEME.bodyFontFace, color: HEX.dk1, fontSize: 18, valign: 'top', margin: 0, ...o });
const imgSize = p => { const b = fs.readFileSync(p); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };
let figNo = 0, tabNo = 0;
// фигура, вписана в кутия, с надпис „Фиг. N.“ непосредствено под нея
const fig = (s, file, x, y, w, h, caption, dir = FIG, align = 'c') => {
  const p = path.join(dir, file); const [iw, ih] = imgSize(p);
  let W2 = w, H2 = w * ih / iw;
  if (H2 > h) { H2 = h; W2 = h * iw / ih; }
  const X = align === 'l' ? x : x + (w - W2) / 2;
  s.addImage({ path: p, x: X, y, w: W2, h: H2, altText: caption || file });
  if (caption) {
    figNo += 1;
    s.addText([{ text: `Фиг. ${figNo}. `, options: { bold: true, color: INK } }, { text: caption, options: { color: GREY } }],
      { isTextBox: true, x: X, y: y + H2 + 0.08, w: Math.max(W2, 4), h: 0.5, fontSize: 13, valign: 'top', margin: 0 });
  }
  return { x: X, y, w: W2, h: H2, bottom: y + H2 + (caption ? 0.5 : 0) };
};
const formula = (s, file, x, y, w, h) => fig(s, file, x, y, w, h, null, FORM, 'l');
const bul = (s, items, x, y, w, h, o = {}) => s.addText(items.map((t, i) => {
  const runs = Array.isArray(t) ? t : [{ text: t }];
  return runs.map((r, j) => ({ text: r.text, options: { ...(r.b ? { bold: true, color: INK } : {}), ...(j === 0 ? { bullet: { indent: 18 } } : {}), ...(j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}), paraSpaceAfter: 10 } }));
}).flat(), { isTextBox: true, x, y, w, h, fontSize: o.fontSize || 18, color: HEX.dk1, valign: 'top', margin: 0, lineSpacingMultiple: 1.07, ...o });
const sub = (s, text, x, y, w) => {
  T(s, text, { x, y, w, h: 0.4, fontSize: 18.5, bold: true, color: INK });
  s.addShape(pres.shapes.LINE, { x, y: y + 0.42, w, h: 0, line: { color: LIGHT, width: 0.75 } });
};
// таблица в стил booktabs
const table = (s, rows, x, y, w, colW, caption, o = {}) => {
  tabNo += 1;
  s.addText([{ text: `Таблица ${tabNo}. `, options: { bold: true, color: INK } }, { text: caption, options: { color: GREY } }],
    { isTextBox: true, x, y, w, h: 0.35, fontSize: 13, margin: 0, valign: 'bottom' });
  const n = rows.length;
  const data = rows.map((row, i) => row.map((c, j) => {
    const cell = typeof c === 'object' ? c : { text: String(c) };
    const top = i === 0 ? { type: 'solid', pt: 1.5, color: RULE } : (i === 1 ? { type: 'solid', pt: 0.75, color: RULE } : { type: 'none' });
    const bottom = i === n - 1 ? { type: 'solid', pt: 1.5, color: RULE } : (i === 0 ? { type: 'solid', pt: 0.75, color: RULE } : { type: 'none' });
    return { text: cell.text, options: { bold: i === 0 || cell.b, color: i === 0 ? INK : HEX.dk1, align: j && o.right !== false ? 'right' : 'left',
      border: [top, { type: 'none' }, bottom, { type: 'none' }], fill: cell.fill ? { color: cell.fill } : undefined } };
  }));
  s.addTable(data, { x, y: y + 0.42, w, colW, fontSize: o.fontSize || 16.5, fontFace: THEME.bodyFontFace, rowH: o.rowH || 0.44, valign: 'middle', margin: [0, 0.08, 0, 0.08] });
};

let idx = 0, secName = '';
const slide = title => {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: secName });
  s.addText(title, { placeholder: 'title' });
  s.addText(secName, { placeholder: 'sec' });
  const note = NOTES[idx++]; if (note) s.addNotes(note.speech);
  return s;
};
const section = t => { secName = t; pres.addSection({ title: t }); };
// стандартни зони
const A = { fx: X0, fy: 1.55, fw: 7.6, fh: 4.95, tx: 8.65, ty: 1.6, tw: 4.03, th: 5.5 };   // фигура вляво, текст вдясно
const WIDE = { fx: X0, fy: 1.5, fw: XW, fh: 3.95 };                                          // широка фигура, текст отдолу


const numbered = (s, items, x, y, w, h, o = {}) => s.addText(items.map((q, i, a) => ({ text: q, options: { bullet: { type: 'number', indent: 22 }, breakLine: i < a.length - 1, paraSpaceAfter: o.gap || 10 } })),
  { isTextBox: true, x, y, w, h, fontSize: o.fontSize || 18, color: HEX.dk1, valign: 'top', margin: 0, lineSpacingMultiple: 1.05 });

// ================================================================ 1. ВЪВЕДЕНИЕ
section('Въведение');
let s = pres.addSlide({ masterName: 'COVER', sectionTitle: secName });
s.addNotes(NOTES[idx++].speech);
s.addText('Експериментално изследване на откриване на проникване в мрежов трафик на център за данни, базирано на машинно обучение', { placeholder: 'title' });
s.addText('PortScan, DDoS и BruteForce · Random Forest, SVM и XGBoost · реални данни CSE-CIC-IDS2018 и CIC-IDS2017', { placeholder: 'body' });
s.addShape(pres.shapes.LINE, { x: 0.75, y: 5.45, w: 1.2, h: 0, line: { color: HEX.accent4, width: 2.5 } });
[['Иван Терзиев', 'Fak No. 961326004'], ['Кристиян Януцев', 'Fak No. 961326003'], [THIRD, THIRD_FN]].forEach(([n, f], k) =>
  T(s, [{ text: n, options: { bold: true, color: 'FFFFFF', breakLine: true } }, { text: f, options: { color: 'AEBBCD' } }],
    { x: 0.75 + k * 2.75, y: 5.65, w: 2.6, h: 0.75, fontSize: 16 }));
T(s, 'Интелигентни системи за сигурност на екосистемите на големите данни и Интернет на нещата', { x: 0.75, y: 6.55, w: 8.0, h: 0.5, fontSize: 14, color: 'AEBBCD' });

s = slide('Съдържание: път на изследването');
{
  const stops = [
    ['Цел', 'изследователски въпроси'],
    ['Данни', 'избор на набор и атаки, почистване'],
    ['Обработка', 'филтър, подбор на признаци'],
    ['Модели и метрики', 'RF, XGBoost, SVM, правило 1-SE'],
    ['Рамка и план', 'инструменти, две фази, контроли'],
    ['Резултати', 'метрики, преобучение, значимост'],
    ['Анализ и изводи', 'приложимост, заключение'],
  ];
  const lx = X0 + 0.95, rx = X0 + XW - 0.95, ly = 4.15, step = (rx - lx) / (stops.length - 1), r = 0.32;
  s.addShape(pres.shapes.LINE, { x: lx, y: ly, w: rx - lx, h: 0, line: { color: LIGHT, width: 2.25 } });
  s.addShape(pres.shapes.LINE, { x: rx, y: ly, w: 0.42, h: 0, line: { color: LIGHT, width: 2.25, endArrowType: 'triangle' } });
  stops.forEach(([t, d], i) => {
    const cx = lx + i * step, up = i % 2 === 0, last = i === stops.length - 1;
    s.addShape(pres.shapes.OVAL, { x: cx - r, y: ly - r, w: 2 * r, h: 2 * r, fill: { color: last ? HEX.accent1 : INK }, line: { color: 'FFFFFF', width: 2 } });
    T(s, String(i + 1), { x: cx - r, y: ly - r, w: 2 * r, h: 2 * r, fontSize: 19, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle' });
    const ty = up ? ly - 0.62 - 1.35 : ly + 0.62;
    s.addShape(pres.shapes.LINE, { x: cx, y: up ? ly - 0.62 : ly + r, w: 0, h: 0.62 - r, line: { color: LIGHT, width: 0.75 } });
    T(s, [{ text: t, options: { bold: true, color: INK, breakLine: true, paraSpaceAfter: 3 } }, { text: d, options: { color: GREY, fontSize: 15.5 } }],
      { x: cx - 1.05, y: ty, w: 2.1, h: 1.35, fontSize: 19, align: 'center', valign: up ? 'bottom' : 'top' });
  });
  T(s, `Представят: Кристиян Януцев (1–2), Иван Терзиев (3–5), ${THIRD} (6–7).`, { x: X0, y: 6.45, w: XW, h: 0.4, fontSize: 15, color: GREY, align: 'center' });
}

s = slide('Цел и изследователски въпроси');
sub(s, 'Цел', X0, 1.55, 5.6);
T(s, [
  { text: 'Да се построят и сравнят девет модела за откриване на проникване: Random Forest, SVM и XGBoost, всеки поотделно за сканиране на портове (PortScan), за Distributed Denial of Service (DDoS) и за Brute Force срещу FTP/SSH (BruteForce). Моделите се оценяват върху невиждани записи по TP, TN, FP, FN, Accuracy, Precision, Recall, F1 и Specificity.', options: { breakLine: true, paraSpaceAfter: 12 } },
  { text: 'Използват се само реални записи от два набора на Canadian Institute for Cybersecurity. Синтетични данни няма.', options: {} },
], { x: X0, y: 2.15, w: 5.6, h: 4.5, fontSize: 18, lineSpacingMultiple: 1.08 });
sub(s, 'Въпроси', 6.85, 1.55, 5.83);
numbered(s, ['Колко точно трите алгоритъма откриват трите атаки?', 'Как редукцията на признаците влияе на качеството и скоростта?', 'Значими ли са разликите, стабилни ли са метриките, има ли преобучение?',
  'Какво е качеството при реалния дял на атаките?', 'От какво зависи постижимата точност?'], 6.85, 2.15, 5.83, 4.8, { gap: 12 });

// ================================================================ 2. ДАННИ
section('Данни');
s = slide('Избор на атаки и набори от данни');
let F = fig(s, 'p02_datasets.png', WIDE.fx, WIDE.fy, WIDE.fw, 3.45, 'Точност върху тестовите записи. Вляво: предварителното изследване върху CICIoT2023; вдясно: избраните набори.');
bul(s, [
  [{ text: 'PortScan: ', b: true }, { text: `в CICIoT2023 (запис = прозорец от пакети) остава около ${pct(preAcc('PortScan', 'XGB'), 0)}. CSE-CIC-IDS2018 няма етикет за сканиране, затова атаката е от коригирания CIC-IDS2017 (запис = двупосочен поток, Nmap).` }],
  [{ text: 'DDoS (HOIC, LOIC-UDP) и BruteForce (FTP, SSH): ', b: true }, { text: 'от CSE-CIC-IDS2018, записан в AWS (420 машини, 30 сървъра).' }],
  [{ text: 'Едни и същи 77 признака в двата набора. ', b: true }, { text: 'Нормалният трафик на всяка задача е от набора на атаката.' }],
], X0, F.bottom + 0.1, XW, 7.2 - F.bottom - 0.1, { fontSize: 16 });

s = slide('Данни и почистване');
F = fig(s, 'p01_class_counts.png', X0, 1.4, XW, 3.55, 'Записи по етикет преди и след почистването и дял на премахнатите.');
bul(s, [
  `${L.files} дневни CSV файла (3 + 3): ${int(L.rows_read_total)} записа, ${L.n_features_raw} признака.`,
  `Премахнати ${int(CL.rows_with_nan_or_inf)} записа с NaN/Inf и ${int(CL.duplicates_removed)} дубликата; остават ${int(CL.rows_after)}.`,
  `FTP: от 193 хиляди записа остават 53 уникални; PortScan: от 159 хиляди остават ${int(TS.PortScan.attack_rows_available)}.`,
], X0, F.bottom + 0.12, 5.6, 7.3 - F.bottom - 0.12, { fontSize: 15.5 });
table(s, [['Задача', 'На клас', 'Обучение', 'Тест'],
  ...ATT.map(a => [`${ATT_N[a]} срещу нормален`, int(TS[a].per_class_sample), int(TS[a].train_rows), int(TS[a].test_rows)])],
  6.55, F.bottom + 0.02, 6.13, [2.65, 1.1, 1.2, 1.18], 'Балансирани бинарни задачи, разделяне 70/30.', { fontSize: 14, rowH: 0.27 });

// ================================================================ 3. МЕТОДОЛОГИЯ
section('Методология');
s = slide('Предварителна обработка и подбор на признаци');
formula(s, 'select.png', X0, 1.6, 5.9, 1.75);
numbered(s, ['Махане на Timestamp и Dst Port: 79 → 77 признака', 'Филтър: признаци с нулева дисперсия, 77 → 67 (71 при PortScan)', 'Подбор по важност (gain) в помощен XGBoost: остават 20', 'Стандартизация (z-оценка)'],
  X0, 3.55, 5.9, 2.2, { fontSize: 17, gap: 6 });
T(s, 'Стъпки 2–4 са част от sklearn Pipeline и се научават само от обучаващите записи, във всеки фолд на кръстосаната валидация. PCA е контролен вариант (E5).',
  { x: X0, y: 5.85, w: 5.9, h: 1.3, fontSize: 16, color: GREY, lineSpacingMultiple: 1.05 });
fig(s, 'p08_selection.png', 6.95, 1.5, 5.73, 4.95, 'Важност на 12-те най-важни признака за PortScan.');

s = slide('Три алгоритъма: Random Forest, XGBoost и SVM');
{
  const cols = [
    ['Random Forest', 'rf.png', ['Ансамбъл от дървета върху bootstrap извадки и случайни признаци.', 'Средното на слабо корелирани дървета има малка дисперсия.'],
      ATT.map(a => { const p = bp('RF', a); return `${ATT_N[a]}: ${p.clf__n_estimators} дървета, дълб. ${p.clf__max_depth ?? '∞'}, листо ${p.clf__min_samples_leaf}`; })],
    ['XGBoost', 'xgb.png', ['Дърветата се строят последователно; всяко поправя грешките на предишните.', 'Регуляризацията γT + ½λΣw² пази от преобучение.'],
      ATT.map(a => { const p = bp('XGB', a); return `${ATT_N[a]}: ${p.clf__n_estimators} дървета, дълб. ${p.clf__max_depth}, ν ${dec(p.clf__learning_rate, 2)}`; })],
    ['SVM (RBF)', 'svm.png', ['Нелинейна граница с максимален марж; C наказва грешките, γ задава обхвата на ядрото.', 'Сложност O(n²)–O(n³): обучение върху най-много 50 000 записа.'],
      ATT.map(a => { const p = bp('SVM', a); return `${ATT_N[a]}: C = ${p.clf__C}, γ = ${p.clf__gamma === 'scale' ? 'scale' : dec(p.clf__gamma, 2)}`; })],
  ];
  const cw = (XW - 0.5) / 3;
  cols.forEach(([name, f, pts, hp], k) => {
    const x = X0 + k * (cw + 0.25);
    sub(s, name, x, 1.5, cw);
    formula(s, f, x, 2.1, cw, 1.25);
    bul(s, pts, x, 3.5, cw, 1.9, { fontSize: 15.5 });
    T(s, [{ text: 'Избрани параметри (1-SE)', options: { bold: true, color: INK, breakLine: true } }, ...hp.map((h, i, a) => ({ text: h, options: { breakLine: i < a.length - 1 } }))],
      { x, y: 5.5, w: cw, h: 1.5, fontSize: 13.5, color: GREY, lineSpacingMultiple: 1.05 });
  });
}

s = slide('Избор на хиперпараметри: правило „една стандартна грешка“');
formula(s, 'onese.png', X0, 1.55, 6.9, 1.5);
T(s, '5-кратна кръстосана валидация с критерий F1. От конфигурациите в рамките на една стандартна грешка от най-добрата се избира най-простата: по-плитки дървета, повече записи в листо, по-малки C и γ.',
  { x: 7.85, y: 1.6, w: 4.83, h: 1.6, fontSize: 16.5, lineSpacingMultiple: 1.05 });
fig(s, 'p09_one_se.png', X0, 3.3, XW, 3.3, 'Среден F1 ± стандартна грешка за всяка конфигурация при PortScan; пунктирът е прагът 1-SE.');

s = slide('Метрики за оценка');
formula(s, 'metrics.png', X0, 1.65, 5.6, 3.9);
table(s, [['Метрика', 'Какво показва (на 100 случая)'],
  ['Accuracy', 'колко от 100 решения са верни'], ['Precision', 'колко от 100 аларми са истински атаки'], ['Recall (Sensitivity)', 'колко от 100 атаки са открити'],
  ['Specificity', 'колко от 100 нормални записа минават без аларма'], ['F1', 'хармонично средно на Precision и Recall']],
  6.45, 1.5, 6.23, [2.25, 3.98], 'Тълкуване на метриките по материала „Метрика“.', { right: false, fontSize: 16, rowH: 0.58 });
bul(s, ['TP: открита атака; TN: нормален запис без аларма; FP: фалшива аларма; FN: пропусната атака, най-скъпата грешка за една система за сигурност.'], X0, 5.75, 5.6, 1.4, { fontSize: 16.5 });

// ================================================================ 4. РАМКА И ПЛАН
section('Рамка и план');
s = slide('Експериментална технологична рамка');
F = fig(s, '00_pipeline.png', X0, 1.45, 7.3, 4.6, 'Експериментален конвейер: от суровите CSV файлове до метриките.', RFIG);
table(s, [['Компонент', 'Съдържание'],
  ['Среда', `Python ${ENV.python.split('.').slice(0, 2).join('.')}, scikit-learn ${ENV['scikit-learn'].split('.').slice(0, 2).join('.')}, XGBoost 3.4`],
  ['registry.py', 'атаки и алгоритми като записи'],
  ['run_all.py', `целият експеримент (${Math.round(RS.total_minutes)} мин.)`],
  ['notebooks/', 'Colab ноутбук и диаграми'],
  ['models/', '9 обучени модела (.joblib)'],
  ['tests/', '10 теста върху реални записи']],
  8.25, 1.45, 4.43, [1.5, 2.93], 'Хранилище ml-ids-cicids2018.', { right: false, fontSize: 14.5, rowH: 0.48 });
T(s, 'Всички числа в доклада и в презентацията се генерират от results/. При фиксиран seed 42 повторният пуск дава идентични метрики.', { x: 8.25, y: 5.35, w: 4.43, h: 1.4, fontSize: 15.5, color: GREY });

s = slide('План на експериментите');
sub(s, 'Фаза на машинно обучение (70% от записите)', X0, 1.55, 5.8);
numbered(s, ['Балансирана бинарна задача и разделяне 70/30', 'GridSearchCV: 5 фолда, F1, правило 1-SE', 'Обучение на финалния конвейер', 'Диагностика на преобучението', 'Запис на модела във файл'], X0, 2.15, 5.8, 2.9);
sub(s, 'Оперативна фаза (30% невиждани записа)', 6.88, 1.55, 5.8);
numbered(s, ['Зареждане на записания модел', 'Класификация на пакети по 10 000 записа', 'TP, TN, FP, FN и производните метрики', 'Bootstrap интервали и тест на McNemar'], 6.88, 2.15, 5.8, 2.9);
s.addShape(pres.shapes.LINE, { x: X0, y: 5.25, w: XW, h: 0, line: { color: LIGHT, width: 0.75 } });
T(s, [{ text: 'Тестовите записи не участват нито в избора на хиперпараметри, нито в научаването на филтъра, подбора и стандартизацията.', options: { breakLine: true, paraSpaceAfter: 8 } }, { text: 'Допълнително: контролни експерименти E1–E5 (без редукция, естествен дял на атаките, стабилност, без времеви признаци, PCA), анализи A1–A9 и проверки за приложимост A10–A14.', options: { color: GREY } }], { x: X0, y: 5.4, w: XW, h: 1.6, fontSize: 16.5 });

// ================================================================ 5. РЕЗУЛТАТИ
section('Резултати');
s = slide('Основни резултати в оперативната фаза');
{
  const row = r => [`${ALG_N[r.algo]}, ${ATT_N[r.attack]}`, int(r.TP), int(r.TN), int(r.FP), int(r.FN), pct(r.Accuracy, 3), pct(r.Precision, 3), pct(r.Sensitivity, 3), pct(r.F1, 3), pct(r.Specificity, 3)];
  table(s, [['Модел', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity'],
    ...ATT.flatMap(a => ALG.map(g => { const r = get(M, a, g); const c = row(r); return (a === HARD && g === 'SVM') ? c.map(t => ({ text: t, fill: 'FBEFE3' })) : c; }))],
    X0, 1.42, XW, [2.95, 0.95, 0.95, 0.6, 0.6, 1.2, 1.2, 1.15, 1.13, 1.3], `Метрики върху ${int(get(M, 'DDoS', 'RF').n_test)} (DDoS), ${int(get(M, 'BruteForce', 'RF').n_test)} (BruteForce) и ${int(get(M, HARD, 'RF').n_test)} (PortScan) тестови записа.`, { fontSize: 14.5, rowH: 0.42 });
}
T(s, `Всички девет модела имат Accuracy над ${accFloor} (минимум ${pct(minAcc, 2)}). При PortScan тестът е малък (${int(get(M, HARD, 'RF').n_test)} уникални записа): грешките са ${errs(HARD, 'RF')} (RF), ${errs(HARD, 'XGB')} (XGBoost) и ${errs(HARD, 'SVM')} (SVM).`,
  { x: X0, y: 6.3, w: XW, h: 0.8, fontSize: 17.5 });

s = slide('Проверка за преобучение');
F = fig(s, 'p13_overfitting.png', WIDE.fx, WIDE.fy, WIDE.fw, 4.0, 'Точност при обучение, оценка от кръстосаната валидация и точност при тест за всеки модел.');
bul(s, [`Най-голямата разлика между обучение и тест е ${dec(gapMax * 100, 3)} процентни пункта: моделите не са преобучени.`,
  'Причини: регуляризация (дълбочина, листа, Ω в XGBoost, C и γ в SVM), правило 1-SE и почистване на дубликатите. По-голямата разлика при PortScan отговаря на 2–3 тестови записа.'],
  X0, F.bottom + 0.15, XW, 7.2 - F.bottom - 0.15, { fontSize: 17 });

s = slide('Статистическа значимост: bootstrap и McNemar');
formula(s, 'stats.png', X0, 1.45, 5.3, 1.75);
bul(s, [
  `${nSig} от 9 двойки се различават значимо (p < 0,05): SVM срещу двата ансамбъла при DDoS и BruteForce.`,
  `PortScan, RF срещу SVM: ${MC['PortScan:RF-SVM'].only_A_correct} срещу ${MC['PortScan:RF-SVM'].only_B_correct} записа, p = ${dec(MC['PortScan:RF-SVM'].p_value, 2)}: разликата не е значима.`,
  'RF и XGBoost дават еднакви решения при DDoS и BruteForce.',
], 6.3, 1.55, 6.38, 1.9, { fontSize: 16.5 });
fig(s, 'p16_mcnemar.png', X0, 3.4, XW, 3.3, 'Вляво: F1 и 95% bootstrap интервал при PortScan. Вдясно: записи, на които е прав само единият модел, и p от теста на McNemar.');

s = slide('Контролни експерименти E1, E4 и E5');
fig(s, 'p10_controls.png', X0, 1.45, 7.7, 5.2, 'Accuracy в четири варианта на конвейера; цветът е разликата спрямо основния вариант.');
bul(s, [
  [{ text: 'E1: ', b: true }, { text: 'с всички 77 признака качеството не се подобрява; 20 признака са достатъчни.' }],
  [{ text: 'E4: ', b: true }, { text: 'без времевите признаци DDoS и BruteForce не се променят; при PortScan спадът е до 0,3 пункта (SVM).' }],
  [{ text: 'E5: ', b: true }, { text: 'PCA дава близко качество, но губи обяснимостта на признаците.' }],
  'При DDoS и BruteForce разликите са под 0,03 пункта; при PortScan един тестов запис е 0,1 пункта.',
], 8.75, 1.7, 3.93, 5.3, { fontSize: 16.5 });

s = slide('Оценка при естествения дял на атаките (E2)');
formula(s, 'baserate.png', X0, 1.45, 5.2, 0.9);
F = fig(s, 'p19_natural.png', X0, 2.5, XW, 3.75, `Вляво: Precision–Recall за PortScan (${int(get(MN, HARD, 'RF').n_test)} неизползвани записа); вдясно: грешки на 10 000 записа за всяка атака.`);
T(s, `Делът на атаките е ${pct(get(MN, HARD, 'RF').attack_share, 1)} (PortScan) до ${pct(get(MN, 'DDoS', 'RF').attack_share, 1)} (DDoS). Precision остава над ${pct(Math.min(natMin('DDoS'), natMin('BruteForce')), 1)} при DDoS и BruteForce; при PortScan е ${pct(natMin(HARD), 1)}–${pct(natMax(HARD), 1)}.`,
  { x: 6.2, y: 1.5, w: 6.48, h: 0.9, fontSize: 17 });

// ================================================================ 6. АНАЛИЗ
section('Анализ');
s = slide('Какво учат моделите: пермутационна важност (A8)');
F = fig(s, 'p21_importance.png', WIDE.fx, WIDE.fy, WIDE.fw, 3.75, 'Спад на F1 при разбъркване на всеки признак за трите алгоритъма (до 2000 тестови записа, 3 повторения).');
bul(s, [
  [{ text: 'DDoS: ', b: true }, { text: 'началният размер на TCP прозореца, зададен от атакуващата програма.' }],
  [{ text: 'BruteForce: ', b: true }, { text: 'минималният размер на сегмента от клиента.' }],
  [{ text: 'PortScan: ', b: true }, { text: 'SVM разчита на броя флагове SYN и на протокола; при дърветата признаците са взаимозаменяеми и разбъркването на един не променя резултата.' }],
], X0, F.bottom + 0.05, XW, 7.15 - F.bottom - 0.05, { fontSize: 15.5 });

s = slide('Проверка за приложимост: отпечатъци и време (A10, A11, A14)');
fig(s, 'p25_robustness.png', X0, 1.45, 6.9, 5.3, 'Accuracy и дял открити атаки при четири постановки на теста.', FIG, 'l');
bul(s, [
  [{ text: 'A10: ', b: true }, { text: 'без признаците на TCP стека на инструмента точността почти не се променя.' }],
  [{ text: 'A11: ', b: true }, { text: `обучение върху ранните 70% от потоците, тест върху късните 30%. DDoS и BruteForce остават над 99,5%.` }],
  [{ text: 'PortScan: ', b: true }, { text: `моделите откриват само ${pct(sfv(HARD, 'XGB', 'chronological', 'Sensitivity'), 0)} от късните проби: скенерът минава към режими с пълна TCP връзка, които липсват в ранните данни.` }],
  [{ text: 'A14: ', b: true }, { text: `комбинацията „XGBoost или SVM“ помага малко: ${pct(sfv(HARD, 'XGB|SVM', 'chronological', 'Sensitivity'), 1)}.` }],
], 7.75, 1.6, 4.93, 5.4, { fontSize: 16.5 });

s = slide('Нормален трафик от непознат ден (A13)');
F = fig(s, 'p26_benign_days.png', WIDE.fx, WIDE.fy, WIDE.fw, 3.85, 'Фалшиви аларми върху нормален трафик от ден, който не е участвал в обучението (по-лошият от двата дни).');
bul(s, [
  `DDoS с нормален трафик само от 21.02: ${pct(odMax('DDoS', 'XGB'), 0)}–${pct(odMax('DDoS', 'SVM'), 0)} фалшиви аларми в другите дни; с трафик от два дни до ${pct(loMax('DDoS', 'SVM'), 2)}.`,
  `PortScan: ${pct(loMax(HARD, 'RF'), 2)} (RF) до ${pct(loMax(HARD, 'SVM'), 2)} (SVM) независимо от броя дни: с порядък повече от другите две атаки. Нормалният трафик при обучението трябва да покрива цялата мрежа.`,
], X0, F.bottom + 0.1, XW, 7.2 - F.bottom - 0.1, { fontSize: 16 });

s = slide('Пренос между атаките (A7) и подвидовете (A12)');
F = fig(s, 'p27_subtypes.png', WIDE.fx, WIDE.fy, WIDE.fw, 3.5, 'Дял открити атаки от подвид, който не е участвал в обучението.');
bul(s, [
  'A7: в рамките на един набор модел за една атака открива 0% от другата; между наборите има до 18% случайни съвпадения и 4–8% фалшиви аларми.',
  'HOIC (HTTP) и LOIC-UDP (UDP) нямат общи признаци: моделите не се пренасят между тях.',
  'RF открива FTP, обучен върху SSH, само заради общия TCP прозорец на инструмента; без него откриването пада до 11%.',
  'Обучаващите данни трябва да съдържат всеки използван инструмент, както е в основния експеримент.',
], X0, F.bottom + 0.1, XW, 7.2 - F.bottom - 0.1, { fontSize: 16.5 });

s = slide('Сравнение на алгоритмите');
table(s, [['Критерий', 'Random Forest', 'SVM (RBF)', 'XGBoost'],
  ['DDoS, BruteForce', 'практически без грешка', 'до 30 грешки', 'практически без грешка'],
  ['PortScan (Accuracy)', pct(get(M, HARD, 'RF').Accuracy, 2), pct(get(M, HARD, 'SVM').Accuracy, 2), pct(get(M, HARD, 'XGB').Accuracy, 2)],
  ['PortScan, Precision при реален дял', pct(get(MN, HARD, 'RF').Precision, 1), pct(get(MN, HARD, 'SVM').Precision, 1), pct(get(MN, HARD, 'XGB').Precision, 1)],
  ['Обучение', 'всички записи', 'до 50 000', 'всички записи'],
  ['Записи/s (DDoS)', int(speed(get(M, 'DDoS', 'RF'))), int(speed(get(M, 'DDoS', 'SVM'))), int(speed(get(M, 'DDoS', 'XGB')))],
  ['Обяснимост', 'вградена важност', 'само пермутационна', 'вградена важност']],
  X0, 1.4, XW, [3.6, 2.8, 2.8, 2.83], 'Обобщено сравнение на алгоритмите.', { right: false, fontSize: 14, rowH: 0.34 });
fig(s, 'p24_speed.png', 4.75, 4.4, 7.93, 2.75, null, FIG, 'r');
T(s, 'Препоръка: XGBoost (сред най-точните и най-бърз) или Random Forest (най-малко фалшиви аларми при PortScan). SVM е най-слаб и по трите критерия.',
  { x: X0, y: 4.55, w: 4.4, h: 2.5, fontSize: 17 });

s = slide('Ограничения и насоки за развитие');
sub(s, 'Ограничения', X0, 1.55, 5.8);
bul(s, ['Данни от контролирана среда (2017 и 2018 г.); реалният трафик днес е различен.', 'PortScan е от друг набор и остава с 1 682 уникални записа: малък тест, широки интервали.', 'Моделите не се пренасят към нов инструмент или режим на атаката (A11, A12).', 'Бинарен модел за всяка атака поотделно.'], X0, 2.2, 5.8, 4.8, { fontSize: 18 });
sub(s, 'Насоки', 6.88, 1.55, 5.8);
bul(s, ['Признаци, агрегирани по източник, за режимите на сканиране с пълна връзка.', 'DDoS и BruteForce върху коригираната версия на CSE-CIC-IDS2018.', 'Многокласов детектор за всички атаки в наборите.', 'Нови атаки и алгоритми: по един запис в регистъра.'], 6.88, 2.2, 5.8, 4.8, { fontSize: 18 });

section('Заключение');
s = slide('Изводи');
numbered(s, [
  `Деветте модела постигат Accuracy над ${accFloor} върху реални данни (минимум ${pct(minAcc, 2)}), без преобучение (разлика до ${dec(gapMax * 100, 2)} пункта).`,
  'DDoS и BruteForce се откриват практически без грешка. PortScan се открива с 99,4–99,8%, а разликите между алгоритмите там не са статистически значими.',
  `При реалния дял на атаките Precision остава над 99% за DDoS и BruteForce и е ${pct(natMin(HARD), 0)}–${pct(natMax(HARD), 0)} за PortScan.`,
  'Подборът на признаци е за предпочитане пред PCA: същото качество с обясними признаци.',
  `При хронологично разделяне DDoS и BruteForce остават над 99,5%; при PortScan моделите откриват ${pct(sfv(HARD, 'XGB', 'chronological', 'Sensitivity'), 0)} от по-късните проби, защото режимите на скенера се сменят.`,
  `Същата атака PortScan: около ${pct(preAcc('PortScan', 'XGB'), 0)} в CICIoT2023 и над 99% в CIC-IDS2017. Качеството на IDS зависи преди всичко от това дали признаците носят отпечатъка на атаката.`,
  'Кодът, моделите и ноутбукът са възпроизводими и използват само реални данни.',
], X0, 1.6, 11.6, 5.5, { fontSize: 17.5, gap: 9 });

s = slide('Информационни източници');
{
  // същият номериран списък като в доклада (report/build_docx.py -> references.json)
  const REFS = JSON.parse(fs.readFileSync(path.join(__dirname, 'references.json'), 'utf8')).map(t => t.replace(/--/g, '–').replace(/[<>]/g, ''));
  const half = Math.ceil(REFS.length / 2) + 1, cw = (XW - 0.35) / 2;
  [REFS.slice(0, half), REFS.slice(half)].forEach((col, c) => s.addText(col.map((t, i, a) => ({
    text: `[${c * half + i + 1}] ${t}`, options: { breakLine: i < a.length - 1, paraSpaceAfter: 2.5 } })),
    { isTextBox: true, x: X0 + c * (cw + 0.35), y: 1.45, w: cw, h: 5.45, fontSize: 8, color: HEX.dk1, valign: 'top', margin: 0, lineSpacingMultiple: 0.95, objectName: `refs-col-${c + 1}` }));
  T(s, 'Номерацията съвпада с доклада. Код, модели и ноутбук: https://github.com/Vankata03/ml-ids-cicids2018', { x: X0, y: 6.95, w: XW, h: 0.3, fontSize: 11, color: GREY });
}

// ---------------------------------------------------------------- финален слайд
s = pres.addSlide({ masterName: 'BLANK', sectionTitle: secName });
s.background = { color: '0A1424' };
s.addNotes(NOTES[idx++].speech);
s.addImage({ path: path.join(FIG, 'p00_title.png'), x: 0, y: 0, w: W, h: 7.5, altText: 'Реални записи от двата набора в първите две главни компоненти' });
T(s, 'Благодаря за вниманието!', { x: 0.75, y: 2.6, w: 8.2, h: 1.3, fontSize: 48, bold: true, color: 'FFFFFF', valign: 'bottom' });
s.addShape(pres.shapes.LINE, { x: 0.75, y: 4.15, w: 1.2, h: 0, line: { color: HEX.accent4, width: 2.5 } });
T(s, [
  { text: 'Иван Терзиев', options: { bold: true, color: 'FFFFFF' } }, { text: '   Fak No. 961326004', options: { color: 'AEBBCD', breakLine: true } },
  { text: 'Кристиян Януцев', options: { bold: true, color: 'FFFFFF' } }, { text: '   Fak No. 961326003', options: { color: 'AEBBCD', breakLine: true } },
  { text: THIRD, options: { bold: true, color: 'FFFFFF' } }, { text: '   ' + THIRD_FN, options: { color: 'AEBBCD' } },
], { x: 0.75, y: 4.4, w: 6.5, h: 1.4, fontSize: 17, lineSpacingMultiple: 1.2 });

if (idx !== NOTES.length) console.warn(`ВНИМАНИЕ: ${idx} слайда, ${NOTES.length} бележки`);
(async () => {
  await pres.writeFile({ fileName: OUT });
  if (APPLY) { const { applyTheme } = require(APPLY); await applyTheme(OUT, THEME); }
  console.log('written', OUT, idx, 'slides,', figNo, 'figures,', tabNo, 'tables');
})();
