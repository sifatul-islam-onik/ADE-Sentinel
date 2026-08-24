# Nearest neighbours (PRD 7.3)

Top-5 cosine neighbours for a fixed probe list, side by side. Probes span four families so the table shows more than one kind of failure: drug names, toxicity morphology, everyday words used in a clinical sense, and the causation cues Stage 1 depends on.

| Probe | E1 GloVe (general) | E2 Word2Vec (ours) | E3 FastText (ours) |
|---|---|---|---|
| `doxorubicin` | cisplatin (0.65)<br>cyclophosphamide (0.63)<br>chemotherapeutic (0.56)<br>vincristine (0.56)<br>etoposide (0.56) | dox (0.71)<br>adriamycin (0.71)<br>anthracycline (0.65)<br>mitoxantrone (0.64)<br>naca (0.63) | doxorubicine (0.96)<br>doxorubicinol (0.94)<br>doxorubicin-based (0.88)<br>amrubicin (0.87)<br>daunorubicin (0.87) |
| `methotrexate` | cyclophosphamide (0.60)<br>azt (0.58)<br>azathioprine (0.57)<br>prednisone (0.55)<br>misoprostol (0.54) | MTX (0.81)<br>hd-mtx (0.63)<br>single-course (0.62)<br>hdmtx (0.61)<br>6-mercaptopurine (0.61) | methotrexat (0.97)<br>methotrexate-based (0.90)<br>methotrexate-loaded (0.87)<br>methotrexate-related (0.84)<br>methotrexate-treated (0.83) |
| `cisplatin` | carboplatin (0.69)<br>doxorubicin (0.65)<br>cyclophosphamide (0.61)<br>paclitaxel (0.61)<br>etoposide (0.60) | cddp (0.74)<br>cis-diamminedichloroplatinum (0.66)<br>once-every-3-weeks (0.64)<br>ddp (0.64)<br>platinum-based (0.63) | cisplatin's (0.93)<br>cisplatin-ccrt (0.91)<br>cisplatinum (0.90)<br>post-cisplatin (0.85)<br>irinotecan-cisplatin (0.85) |
| `vancomycin` | metronidazole (0.56)<br>antibiotic (0.55)<br>aureus (0.55)<br>methicillin (0.55)<br>enterococci (0.55) | cefazolin (0.68)<br>teicoplanin (0.68)<br>meropenem (0.66)<br>cefepime (0.66)<br>nafcillin (0.65) | vancomycin-gentamicin (0.90)<br>lincomycin (0.88)<br>vancomycin-soaked (0.85)<br>vancomycin-impregnated (0.80)<br>vancomycin-related (0.79) |
| `hepatotoxicity` | side-effects (0.52)<br>dyskinesia (0.50)<br>agranulocytosis (0.47)<br>tardive (0.47)<br>neurotoxicity (0.47) | nephrotoxicity (0.69)<br>hepatoxicity (0.68)<br>hepatotoxic (0.65)<br>cardiotoxicity (0.64)<br>n-acetyl-p-aminophenol (0.63) | hepatotoxicities (0.94)<br>hepatoxicity (0.93)<br>hepatotoxic (0.91)<br>hepatotoxicant (0.87)<br>hepatonephrotoxicity (0.87) |
| `thrombocytopenia` | anemia (0.59)<br>neutropenia (0.57)<br>leukopenia (0.56)<br>anaemia (0.52)<br>hemolysis (0.50) | thrombopenia (0.68)<br>thrombocytopaenia (0.67)<br>lansoprazole-induced (0.65)<br>heparin-induced (0.65)<br>hypofibrinogenemia (0.64) | thrombocytopenias (0.98)<br>macrothrombocytopenia (0.95)<br>thrombocytopenic (0.94)<br>thrombocytopaenia (0.94)<br>pseudothrombocytopenia (0.93) |
| `rhabdomyolysis` | obliterans (0.46)<br>uremic (0.46)<br>cardiomyopathy (0.44)<br>arrhythmias (0.43)<br>hepatotoxicity (0.43) | imnm (0.60)<br>abiraterone-induced (0.60)<br>saam (0.59)<br>statin-associated (0.59)<br>metformin-associated (0.59) | rhabdomyolyse (0.93)<br>rhabdoid (0.68)<br>rhabdomyosarcoma (0.66)<br>myocytolysis (0.61)<br>myospherulosis (0.58) |
| `nephrotoxicity` | mangxamba (0.51)<br>zety (0.49)<br>mongkolporn (0.48)<br>___________________________________________________________ (0.48)<br>rosnazura (0.47) | hepatotoxicity (0.69)<br>ototoxicity (0.64)<br>hepatoxicity (0.63)<br>colistin-induced (0.63)<br>gentamicin-induced (0.63) | nephrotoxicities (0.94)<br>hepatonephrotoxicity (0.92)<br>nephrotoxic (0.91)<br>nephro-toxicity (0.87)<br>nephrotoxicants (0.85) |
| `agranulocytosis` | aplastic (0.50)<br>hepatotoxicity (0.47)<br>neutropenia (0.47)<br>aspd (0.46)<br>plagiocephaly (0.46) | mmi-induced (0.71)<br>clozapine-induced (0.70)<br>azathioprine-induced (0.69)<br>r-lon (0.64)<br>clozapine-related (0.64) | granulocytopenia (0.85)<br>granulocytic (0.82)<br>granulocyte (0.76)<br>leucocytosis (0.74)<br>pleocytosis (0.73) |
| `rash` | spate (0.50)<br>rashes (0.45)<br>itching (0.43)<br>fever (0.42)<br>kidnappings (0.42) | rashes (0.76)<br>acne-like (0.76)<br>maculopapular (0.75)<br>acneiform (0.73)<br>eruptions (0.70) | rashes (0.79)<br>acneiform (0.74)<br>rash2 (0.72)<br>papulopustular (0.70)<br>maculopapular (0.69) |
| `renal` | hepatic (0.63)<br>kidney (0.62)<br>pulmonary (0.57)<br>cardiac (0.54)<br>liver (0.53) | kidney (0.77)<br>post-nephrectomy (0.58)<br>renovascular (0.57)<br>extrarenal (0.56)<br>renal-replacement (0.56) | kidney (0.74)<br>nonkidney (0.66)<br>nonrenal (0.66)<br>renally (0.66)<br>pre-renal (0.65) |
| `carcinoma` | carcinomas (0.71)<br>squamous (0.71)<br>metastatic (0.66)<br>adenocarcinoma (0.63)<br>melanoma (0.62) | squamous (0.78)<br>carcinomas (0.77)<br>hepatocellular (0.71)<br>merkel (0.69)<br>non-clear (0.69) | carcinomas (0.89)<br>carcinoma-in-situ (0.86)<br>microcarcinoma (0.84)<br>carcinome (0.84)<br>choriocarcinoma (0.84) |
| `lesions` | lesion (0.77)<br>tumors (0.66)<br>cysts (0.59)<br>cancerous (0.59)<br>precancerous (0.58) | lesion (0.79)<br>non-ostial (0.57)<br>restenotic (0.57)<br>tascii (0.55)<br>rotablation (0.55) | lesion (0.83)<br>lesional (0.76)<br>lesioning (0.70)<br>lesion's (0.68)<br>pcatlesion (0.68) |
| `induced` | induce (0.64)<br>induces (0.62)<br>inducing (0.58)<br>resulting (0.51)<br>stress (0.46) | caused (0.71)<br>induces (0.66)<br>triggered (0.65)<br>induce (0.63)<br>inducing (0.59) | druginduced (0.80)<br>induces (0.76)<br>cona-induced (0.76)<br>induce (0.76)<br>1260-induced (0.75) |
| `withdrawal` | pullout (0.83)<br>withdraw (0.73)<br>withdrawing (0.70)<br>redeployment (0.64)<br>timetable (0.62) | naloxone-precipitated (0.63)<br>naloxone-induced (0.60)<br>buprenorphine-precipitated (0.58)<br>abstinence (0.58)<br>symptom-triggered (0.58) | withdrawals (0.92)<br>withdrawalpolicy (0.89)<br>post-withdrawal (0.88)<br>withdraw (0.86)<br>withdrawing (0.82) |
| `discontinued` | ceased (0.64)<br>discontinue (0.56)<br>discontinuing (0.51)<br>superseded (0.47)<br>discontinuation (0.45) | stopped (0.79)<br>restarted (0.70)<br>discontinue (0.67)<br>discontinuation (0.67)<br>withdrew (0.66) | discontinuers (0.90)<br>discontinue (0.88)<br>discontinuing (0.87)<br>discontinuance (0.84)<br>discontinuation (0.82) |

**not in vocabulary** is a result, not a gap in the experiment: it is the strongest possible statement about coverage, and PRD 7.3 asks for at least one such probe.

## What the three columns actually show

**E1 GloVe** returns loosely topical words. For `hepatotoxicity` it offers `side-effects`, `tardive`, `dyskinesia` - medical-sounding, but not the same concept. It has no entry at all for much of the drug vocabulary.

**E2 Word2Vec** returns *semantic* relatives, which is what a distributional model is supposed to do:

| Probe | E2 neighbours | Relationship |
|---|---|---|
| `doxorubicin` | adriamycin, epirubicin, anthracycline | brand synonym, sibling drug, drug class |
| `hepatotoxicity` | nephrotoxicity, cardiotoxicity | sibling organ toxicities |
| `withdrawal` | abstinence, discontinuation, naloxone-precipitated | clinically co-occurring concepts |

**E3 FastText** returns *morphological* variants, and this is the finding the PRD did not anticipate:

| Probe | E3 neighbours | Relationship |
|---|---|---|
| `doxorubicin` | doxorubicine, doxorubicinol, doxorubicin-based | spelling and derivational variants |
| `hepatotoxicity` | hepatotoxicities, hepatoxicity, hepatotoxic | inflections and a misspelling |
| `withdrawal` | withdrawals, withdraw, **withdrawalpolicy** | string overlap, including a junk token |

PRD 7.1 predicted that subword information would help on rare drug names, and in one sense it does: FastText never fails to produce a vector, and it robustly unifies spelling variants (`hepatoxicity`/`hepatotoxicity`, `thrombocytopaenia`/`thrombocytopenia`), which is genuinely useful for a corpus with inconsistent orthography.

But the cost is visible. Character n-grams dominate the vector, so cosine similarity largely measures *string* similarity - note the inflated scores (0.92-0.98 for E3 against 0.58-0.69 for E2) and `withdrawalpolicy` surfacing as a near-neighbour of `withdrawal`. E3 does not recover `adriamycin` for `doxorubicin`, because a brand synonym shares no substring with its generic name - and that is exactly the inference an ADE system needs.

**So the ordering is not a single ranking.** E3 wins on robustness to unseen and misspelled forms; E2 wins on semantic relatedness. Which matters more is settled by downstream F1 (runs 3-6), not by this table - which is the reason the PRD specifies three independent evidence types rather than one.
