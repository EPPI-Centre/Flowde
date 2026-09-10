# CONSORT conference dataset

Six studies with their original PDFs, Paddle-extracted figures, reference CONSORT
images, and manually annotated parsing answers. The notebook runs live inference;
these reference files are used to evaluate the results.

## Files

- `pdfs/`: the six unchanged article PDFs.
- `paddle-extracted/`: all eleven stored Paddle image crops, before orientation
  correction.
- `consort-images/`: the six curated CONSORT reference images; some were cropped or
  rotated during annotation.
- `ground-truth/`: the reference nodes, labels, flow and additional text, in four
  same-stem JSON files per diagram.
- `manifest.json`: article metadata, reference image names, licence links, and hashes of
  all dataset files.

Each article has its own Creative Commons licence, listed below. Retain the article
attribution and licence when sharing its PDF, figures or annotations derived from
the diagram. The repository software licence does not replace the article licences.
The Paddle figures are crops of the articles; the reference images may additionally
be rotated or trimmed, and the JSON files transcribe and structure diagram content.

## Article attribution

### Aung_2019.pdf

Effectiveness of a new multi-component smoking cessation service package for patients
with hypertension and diabetes in northern Thailand: a randomized controlled trial
(ESCAPE study)

Myo Nyein Aung, Motoyuki Yuasa, Saiyud Moolphate, Thaworn Lorga, Hirohide Yokokawa,
Hiroshi Fukuda, Tsutomu Kitajima, Susumu Tanimura, Yoshimune Hiratsuka, Koichi Ono,
Payom Thinuan, Kazuo Minematsu, Jitladda Deerojanawong, Yaoyanee Suya, Eiji Marui
(2019).

[Article](https://doi.org/10.1186/s13011-019-0197-2) · [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

### Baskerville_2018.pdf

Effect of a Mobile Phone Intervention on Quitting Smoking in a Young Adult Population of
Smokers: Randomized Controlled Trial

Neill Bruce Baskerville, L C Struik, G. Emmanuel Guindon, Cameron D. Norman, Robyn
Whittaker, Catherine M. Burns, David Hammond, Darly Dash, Karen Brown (2018).

[Article](https://doi.org/10.2196/10893) · [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

### Stanczyk_2016.pdf

Comparison of Text and Video Computer-Tailored Interventions for Smoking Cessation:
Randomized Controlled Trial

Nicola Esther Stanczyk, Catherine Bolman, Mathieu van Adrichem, Math J. J. M. Candel,
Jean Muris, Hein de Vries (2014).

[Article](https://doi.org/10.2196/jmir.3016) · [CC-BY-2.0](https://creativecommons.org/licenses/by/2.0/)

### Vander_2016.pdf

An individually-tailored smoking cessation intervention for rural Veterans: a pilot
randomized trial

Mark W. Vander Weg, Ashley J. Cozad, M. Bryant Howren, Margaret Cretzmeyer, Melody
Scherubel, Carolyn Turvey, Kathleen M. Grant, Thad E. Abrams, David A. Katz (2016).

[Article](https://doi.org/10.1186/s12889-016-3493-z) · [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

### Young_2008.pdf

Acceptability and effectiveness of opportunistic referral of smokers to telephone
cessation advice from a nurse: a randomised trial in Australian general practice

Jane Young, Seham Girgis, Tracey Bruce, Melissa Hobbs, Jeanette Ward (2008).

[Article](https://doi.org/10.1186/1471-2296-9-16) · [CC-BY-2.0](https://creativecommons.org/licenses/by/2.0/)

### Yu_2017.pdf

mHealth Intervention is Effective in Creating Smoke-Free Homes for Newborns: A
Randomized Controlled Trial Study in China

Shaohua Yu, Zongshuan Duan, Pamela Redmon, Michael P. Eriksen, Jeffrey P. Koplan, Cheng
Huang (2017).

[Article](https://doi.org/10.1038/s41598-017-08922-x) · [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

The original filename `Stanczyk_2016.pdf` is retained to match the annotations.
The article itself was published in 2014.

The ground truths can include several accepted interpretations. The four files
for each diagram have aligned `options` lists. Reference annotations are inputs to
the benchmark; they are not model predictions or fallback inference outputs.
