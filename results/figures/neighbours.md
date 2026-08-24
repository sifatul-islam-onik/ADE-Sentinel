# Nearest neighbours (PRD 7.3)

Top-5 cosine neighbours for a fixed probe list, side by side. Probes span four families so the table shows more than one kind of failure: drug names, toxicity morphology, everyday words used in a clinical sense, and the causation cues Stage 1 depends on.

| Probe | E1 GloVe (general) | E2 Word2Vec (ours) |
|---|---|---|
| `doxorubicin` | cisplatin (0.65)<br>cyclophosphamide (0.63)<br>chemotherapeutic (0.56)<br>vincristine (0.56)<br>etoposide (0.56) | dox (0.69)<br>adriamycin (0.68)<br>epirubicin (0.66)<br>anthracycline (0.64)<br>pirarubicin (0.64) |
| `methotrexate` | cyclophosphamide (0.60)<br>azt (0.58)<br>azathioprine (0.57)<br>prednisone (0.55)<br>misoprostol (0.54) | MTX (0.81)<br>hd-mtx (0.63)<br>single-course (0.63)<br>salazosulfapyridine (0.62)<br>dactinomycin (0.61) |
| `cisplatin` | carboplatin (0.69)<br>doxorubicin (0.65)<br>cyclophosphamide (0.61)<br>paclitaxel (0.61)<br>etoposide (0.60) | cddp (0.78)<br>taxol (0.66)<br>ddp (0.66)<br>cis-diamminedichloroplatinum (0.66)<br>nedaplatin (0.65) |
| `vancomycin` | metronidazole (0.56)<br>antibiotic (0.55)<br>aureus (0.55)<br>methicillin (0.55)<br>enterococci (0.55) | teicoplanin (0.69)<br>cefazolin (0.66)<br>piperacillin-tazobactam (0.66)<br>cefepime (0.66)<br>vanc (0.65) |
| `hepatotoxicity` | side-effects (0.52)<br>dyskinesia (0.50)<br>agranulocytosis (0.47)<br>tardive (0.47)<br>neurotoxicity (0.47) | nephrotoxicity (0.66)<br>hepatoxicity (0.65)<br>cardiotoxicity (0.63)<br>hepatotoxic (0.62)<br>hepato (0.61) |
| `thrombocytopenia` | anemia (0.59)<br>neutropenia (0.57)<br>leukopenia (0.56)<br>anaemia (0.52)<br>hemolysis (0.50) | thrombocytopaenia (0.69)<br>thrombopenia (0.69)<br>heparin-induced (0.67)<br>lansoprazole-induced (0.65)<br>ahit (0.64) |
| `rhabdomyolysis` | obliterans (0.46)<br>uremic (0.46)<br>cardiomyopathy (0.44)<br>arrhythmias (0.43)<br>hepatotoxicity (0.43) | imnm (0.60)<br>abiraterone-induced (0.60)<br>simvastatin-induced (0.59)<br>metformin-associated (0.59)<br>eudka (0.59) |
| `nephrotoxicity` | mangxamba (0.51)<br>zety (0.49)<br>mongkolporn (0.48)<br>___________________________________________________________ (0.48)<br>rosnazura (0.47) | hepatotoxicity (0.66)<br>nephrotoxic (0.63)<br>colistin-induced (0.63)<br>hepatoxicity (0.63)<br>cddp-induced (0.62) |
| `agranulocytosis` | aplastic (0.50)<br>hepatotoxicity (0.47)<br>neutropenia (0.47)<br>aspd (0.46)<br>plagiocephaly (0.46) | clozapine-induced (0.73)<br>mmi-induced (0.69)<br>azathioprine-induced (0.66)<br>r-lon (0.66)<br>diiha (0.64) |
| `rash` | spate (0.50)<br>rashes (0.45)<br>itching (0.43)<br>fever (0.42)<br>kidnappings (0.42) | maculopapular (0.75)<br>acne-like (0.75)<br>rashes (0.75)<br>acneiform (0.74)<br>eruptions (0.70) |
| `renal` | hepatic (0.63)<br>kidney (0.62)<br>pulmonary (0.57)<br>cardiac (0.54)<br>liver (0.53) | kidney (0.78)<br>post-nephrectomy (0.60)<br>renal-replacement (0.56)<br>endstage (0.55)<br>ifta (0.55) |
| `carcinoma` | carcinomas (0.71)<br>squamous (0.71)<br>metastatic (0.66)<br>adenocarcinoma (0.63)<br>melanoma (0.62) | squamous (0.78)<br>carcinomas (0.76)<br>hepatocellular (0.72)<br>adenosquamous (0.71)<br>non-clear (0.69) |
| `lesions` | lesion (0.77)<br>tumors (0.66)<br>cysts (0.59)<br>cancerous (0.59)<br>precancerous (0.58) | lesion (0.78)<br>non-ostial (0.58)<br>restenotic (0.57)<br>fp-cto (0.56)<br>steno-occlusive (0.55) |
| `induced` | induce (0.64)<br>induces (0.62)<br>inducing (0.58)<br>resulting (0.51)<br>stress (0.46) | caused (0.72)<br>induces (0.68)<br>induce (0.67)<br>triggered (0.66)<br>toxin-induced (0.62) |
| `withdrawal` | pullout (0.83)<br>withdraw (0.73)<br>withdrawing (0.70)<br>redeployment (0.64)<br>timetable (0.62) | naloxone-precipitated (0.62)<br>abstinence (0.59)<br>buprenorphine-precipitated (0.59)<br>naloxone-induced (0.58)<br>discontinuation (0.58) |
| `discontinued` | ceased (0.64)<br>discontinue (0.56)<br>discontinuing (0.51)<br>superseded (0.47)<br>discontinuation (0.45) | stopped (0.81)<br>restarted (0.73)<br>discontinue (0.67)<br>discontinuation (0.67)<br>reinitiated (0.65) |

**not in vocabulary** is a result, not a gap in the experiment: it is the strongest possible statement about coverage, and PRD 7.3 asks for at least one such probe.

*Pending: E3 FastText (ours). Rerun once trained.*
