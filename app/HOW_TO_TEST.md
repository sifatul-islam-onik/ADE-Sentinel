# How to test the ADE-Sentinel demo

A guide for anyone, no technical or medical knowledge needed.

> **Please note:** this is a research demo, not a medical tool. The sentences you make up are
> only for testing. "Insulin caused hair loss" is a test sentence, not a fact about insulin.

## What the demo does

You type a sentence. The demo answers one question:

**Does this sentence say that a medicine caused a side effect?**

Researchers call that an *adverse drug event*, or **ADE**. If the answer is yes, the demo also
highlights the medicine and the side effect.

## Opening the demo

If the demo is not already open, whoever set up the project runs this in the project folder:

```
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

A browser tab opens by itself. If it does not, go to <http://localhost:8501>.

## Reading the result

Type into the **Your sentence** box, then press **Ctrl+Enter** or click outside the box.

| You see | It means |
|---|---|
| **"Yes - this reports a side effect"** in red | the demo thinks a medicine caused harm |
| **"No side effect reported here"** in grey | the demo thinks nothing like that happened |
| a **blue** highlight | the medicine it found |
| an **orange** highlight | the harm it found |
| **"Medicine:"** and **"Harm it caused:"** under the sentence | the same findings, listed plainly |
| "The tool is 97% sure" | how confident it is in its yes/no answer |

Each sentence gets its own box. The buttons under **Try an example** fill in real sentences
from medical papers.

## Make up your own sentence in 3 steps

1. Pick a **medicine** from the [medicine list](#medicine-list).
2. Pick a **side effect** from the [side-effect list](#side-effect-list).
3. Put them into one of the patterns below.

### Sentences that should say ADE

You should see **"Yes - this reports a side effect"**, the medicine in blue and the harm in orange.

| Pattern | Example to type |
|---|---|
| The patient developed *side effect* after taking *medicine*. | `The patient developed a rash after taking penicillin.` |
| I developed *side effect* after taking *medicine*. | `I developed a headache after taking ibuprofen.` |
| My mother developed *side effect* after taking *medicine*. | `My mother developed dizziness after taking lisinopril.` |
| She had *side effect* after taking *medicine*. | `She had nausea after taking morphine.` |
| *Medicine* caused *side effect*. | `Warfarin caused bleeding.` |

### Sentences that should say not ADE

You should see **"No side effect reported here"** and no highlights.

| Pattern | Example to type | Why it is not ADE |
|---|---|---|
| The doctor gave her *medicine* for *what it is taken for*. | `The doctor gave her amoxicillin for an ear infection.` | the medicine is treating something |
| She took *medicine* and did not get *side effect*. | `She took morphine and did not get nausea.` | nothing went wrong |
| She developed *side effect* after eating peanuts. | `She developed a rash after eating peanuts.` | no medicine involved |
| She takes *medicine* every day. | `She takes insulin every day.` | no side effect |
| Anything that is not about medicine | `The weather is nice today.` | nothing medical |

## Medicine list

For the "doctor gave her … for …" pattern, use the right-hand column.

| Medicine | Usually taken for |
|---|---|
| aspirin | pain |
| ibuprofen | pain, back pain |
| paracetamol | a fever, a headache |
| codeine | pain, a cough |
| morphine | severe pain |
| tramadol | pain |
| naproxen | joint pain |
| penicillin | an infection, a sore throat |
| amoxicillin | an infection, an ear infection |
| doxycycline | an infection, acne |
| insulin | diabetes |
| metformin | diabetes |
| warfarin | blood clots |
| heparin | blood clots |
| digoxin | heart failure |
| propranolol | high blood pressure |
| lisinopril | high blood pressure |
| lithium | bipolar disorder |
| diazepam | anxiety |
| fluoxetine | depression |
| sertraline | depression |
| omeprazole | heartburn |
| simvastatin | high cholesterol |
| prednisone | asthma |
| quinine | malaria |
| allopurinol | gout |

## Side-effect list

Keep the little word in front when there is one ("a rash", "a headache").

| Body area | Side effects |
|---|---|
| Skin and hair | a rash, itching, blisters, acne, sweating, hair loss |
| Stomach and mouth | nausea, vomiting, diarrhea, constipation, stomach pain, dry mouth, mouth ulcers |
| Head and mind | a headache, dizziness, drowsiness, confusion, insomnia, anxiety, depression, hallucinations, memory loss, psychosis |
| Heart, blood and lungs | chest pain, palpitations, a heart attack, a stroke, bleeding, anemia, a cough, shortness of breath |
| Muscles and whole body | a fever, fatigue, weight gain, muscle pain, joint pain, a tremor, a seizure |
| Liver, kidneys and pancreas | liver damage, kidney failure, hepatitis, jaundice, pancreatitis |
| Eyes and ears | blurred vision, hearing loss |

## Tips

- **Use a medicine's name from the list.** General words are not recognised:
  `The patient developed a rash after taking steroids.` says not ADE.
- **Put the medicine and the side effect in the same sentence.** The demo reads each sentence
  on its own, so this says not ADE twice:
  `She was given penicillin for a throat infection. The next day she developed a rash.`
- **Typing several sentences?** End each one with a full stop, then a space, then start the
  next with a capital letter. `Aspirin caused a headache. She takes insulin for diabetes.`
  gives two boxes, one ADE and one not ADE. Without the capital letter
  (`aspirin caused a headache. she takes insulin for diabetes.`) both are read as one
  sentence in one box.
- **A single sentence** can be all lowercase or miss the full stop. That is fine.

## Phrasings that fool the model

These are known weaknesses, not bugs. The model learned from medical case reports, which are
written in formal clinical language, so everyday phrasing throws it.

| Try | What it gets wrong |
|---|---|
| `I got diarrhea after taking aspirin.` | highlights "got diarrhea" instead of just "diarrhea" |
| `Aspirin gave me nausea.` | also marks "me" as a medicine |
| `Morphine did not cause nausea.` | wrongly says ADE |
| `She took aspirin and had no nausea.` | wrongly says ADE |
| `He takes aspirin for pain.` | wrongly says ADE |

The last three are all the same underlying problem: **negation**. The model sees a drug and a
symptom close together and says "adverse drug event", even when the sentence says the opposite.
Notebook 6 measures how much that costs.

To stay inside what it handles well, write the way a case report would: say "developed" or
"had" rather than "got" or "gave me", and "did not get" rather than "did not cause".

## If something looks wrong

If the demo gets a sentence wrong and it is not one of the phrasings above, please write down:

1. the sentence, exactly as you typed it,
2. which model was selected in the sidebar,
3. what the demo showed (the label and the highlights),
4. what you expected instead.

---

*How these lists were chosen:* every medicine on the list was tried with every side effect in
the patterns above, and with each of its own uses in the "doctor gave her" pattern. That came to
about 23,000 made-up sentences, each run on both models. Every medicine, side effect and
pattern kept here was handled correctly at least 95% of the time. Words that failed too often
in some patterns were left out, for example "steroids", "hives", "swelling", "fainting",
"nosebleeds" and "low blood pressure".
