# Word Embeddings for Azerbaijani Language

Four word embedding models trained on a ~76-million-token corpus of Azerbaijani literature, served through a Django web application that lets users explore semantic similarity, nearest neighbours, and word analogies interactively.

Azerbaijani is a low-resource, agglutinative Turkic language. Pretrained embeddings are scarce, and its morphology — where a single root takes long chains of suffixes to form dozens of surface words — makes it a genuinely interesting test case for subword-aware methods like FastText.

> **Corpus not included.** The source texts are confidential and are not distributed with this repository. Trained model files are also not committed, so the web application cannot be run without first training the models.

---

## What it does

The web interface exposes three operations against each of four models:

| Operation | Input | Output |
|---|---|---|
| **Similarity** | Two words | Cosine similarity score |
| **Most similar** | One word | Top 5 nearest neighbours with scores |
| **Analogy** | Two positive words, one negative | The single best analogy completion |

The analogy operation is the classic vector-arithmetic query — `king − man + woman ≈ queen` — applied to Azerbaijani.

---

## The corpus

Built from approximately **250 books written in Azerbaijani**, yielding **76,116,387 tokens**.

Preprocessing, per file:

1. Strip non-breaking spaces and collapse newlines
2. Repair mojibake from bad encoding round-trips (`Ģ` → `ş`, `Ġ` → `i`)
3. Replace commas and colons with spaces
4. Lowercase
5. Collapse ellipses to single periods, then pad periods with spaces so they tokenize separately
6. Strip everything outside the Azerbaijani alphabet with a regex character class covering all 32 letters, plus space and period

Cleaned texts are concatenated into a single `dataCorpus.txt` and fed to gensim via `LineSentence`.

---

## The models

All four are trained with gensim on the same corpus:

| Model | Algorithm | Architecture | Notes |
|---|---|---|---|
| `CBOWw2v` | Word2Vec | CBOW | gensim defaults |
| `SGw2v` | Word2Vec | Skip-gram | `sg=1` |
| `CBOWFT` | FastText | CBOW | character n-grams |
| `SGFT` | FastText | Skip-gram | `size=300`, `bucket=500000`, n-grams 2–4 |

**Why both families.** Word2Vec treats each surface form as an atomic token, so `kitab`, `kitablar`, and `kitablarımızda` get unrelated vectors and rare inflections get poor ones. FastText represents words as bags of character n-grams, so morphologically related forms share substructure and unseen words can still be embedded. On an agglutinative language this difference should be substantial — which is exactly what the comparison is set up to measure.

**Why both architectures.** CBOW predicts a word from its context and trains faster; skip-gram predicts context from a word and generally does better on infrequent words. With a heavily inflected vocabulary, rare forms are the majority.

Each model is saved in word2vec text format and freed from memory before the next is trained, keeping peak RAM manageable.

### Evaluation

`evaluate_word_analogies()` is called against three analogy question files for each model, giving **syntactic**, **semantic**, and **capital-country** scores. This is intrinsic evaluation — it measures whether the vector space encodes the right relationships, without needing a downstream task.

The notebook ships with placeholder paths and no stored outputs, so no scores are recorded in the repository. Filling in the paths and committing the resulting table would make the four-way comparison concrete; see *Next steps*.

---

## The web application

Django 3.1, with a Bootstrap 4 admin-dashboard front end.

```
wordembedded_project/     Django project (settings, root urls, wsgi/asgi)
wordembedded/             App: views, urls, models, migrations
templates/                8 templates — one per model, plus index, authors, SDP
Python Codes for Model Training/
    Word Embedding.py     Training pipeline
    Word Embedding.ipynb  Same, as a notebook
[vendor asset folders]    Bootstrap, jQuery, Font Awesome, Chart.js, etc.
```

### Request flow

A `Models` class loads all four embedding files into memory at import time and exposes `getModelSimilarity`, `getModelMostSimilarity`, and `getAnalogy`. Each view reads query parameters, dispatches on a hidden `modelTypeHidden` field identifying which model page issued the request, and re-renders that page with the result.

Three states are handled per operation: blank input, a word outside the vocabulary, and success — each producing a different message on the page.

Routes: `/` (home), `/CBOW`, `/SkipGram`, `/FastText`, `/SGFastText`, `/Authors`, `/SDP2`, plus `/findSim`, `/findMostSim`, `/findAnalogy` as handlers.

---

## Running it

Requires the trained model files (`CBOWw2v.txt`, `SGw2v.txt`, `CBOWFT.txt`, `SGFT.txt`) in the project root — train them first with the script in `Python Codes for Model Training/`.

```bash
git clone https://github.com/RovshanBayramRB/Word-Embedding-for-Azerbaijani-Language.git
cd Word-Embedding-for-Azerbaijani-Language
pip install django==3.1 gensim==3.8.3 nltk
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000/`.

Startup is slow — all four models load into RAM before the first request is served.

**Version note:** the training code uses `size=` for the embedding dimension, which gensim 4.0 renamed to `vector_size`. Pin gensim to 3.x, or update the parameter names.

---

## Repository structure

```
.
├── manage.py                          # Django entry point
│
├── wordembedded_project/              # Project config
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── wordembedded/                      # Application
│   ├── views.py                       # Similarity, most-similar, analogy handlers
│   ├── urls.py                        # Route definitions
│   ├── models.py
│   ├── exten.py                       # Unused corpus helper
│   ├── admin.py
│   ├── apps.py
│   ├── tests.py
│   └── migrations/
│       ├── 0001_initial.py
│       ├── 0002_cars.py
│       └── 0003_auto_20210205_1715.py
│
├── templates/
│   ├── base.html                      # Shared layout
│   ├── index.html                     # Landing page
│   ├── CBOWW2V.html                   # Word2Vec CBOW
│   ├── SGW2V.html                     # Word2Vec Skip-gram
│   ├── CBOWFT.html                    # FastText CBOW
│   ├── SGFT.html                      # FastText Skip-gram
│   ├── authors.html
│   └── SDP.html
│
├── Python Codes for Model Training/
│   ├── Word Embedding.py              # Corpus cleaning + training all four models
│   └── Word Embedding.ipynb           # Same, as a notebook
│
├── css/  js/  fonts/  images/         # Front-end assets
├── bootstrap-4.1/  font-awesome-4.7/  font-awesome-5/  mdi-font/
├── animsition/  wow/  slick/  lightbox2/  select2/
├── chartjs/  vector-map/  progressbar/  circle-progress/
├── bootstrap-progressbar/  counter-up/  countdown/
├── css-hamburgers/  perfect-scrollbar/
├── jquery-3.2.1.min.js
├── jquery-ui.min.js
├── base.css
└── README.md
```
