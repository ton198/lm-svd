# History/future samples

Every experiment in this repository evaluates GPT-2 on inputs of the form **history + future**: two token sequences concatenated with no separator. This document explains how each sample set is generated and traces concrete examples from the raw source data to the stored samples.

All examples below were extracted from the files currently in `data/` and checked against them: stored token IDs equal the corresponding token-stream slice, decoded IDs equal the stored `text`, and rendered task text re-encodes to the stored IDs.

## 1. Overview

| Sample set | File | Built by | Used by | Unit | Size |
|---|---|---|---|---|---|
| `wikitext` | `data/history_future_samples/wikitext.parquet` | `data-pipline build-windows` | exp1–exp3 | 64-token history / 32-token future windows | 512 × 512 |
| `wikitext-large` | `data/history_future_samples/wikitext-large.parquet` | `data-pipline build-wikitext-large` | exp4 | same windows | 10000 × 10000 |
| `wikitext-103-train` stream | `data/token_streams/wikitext-103-train.npy` + `.json` | `data-pipline build-token-stream` | exp6–exp9 | random windows drawn during training | 117,920,140 tokens (`uint16`) |
| `task-single-text` | `data/history_future_samples/task-single-text.parquet` | `data-pipline build-task-inputs` | exp1–exp3 | instruction history / C4 document future | 1309 × 1309 |

## 2. Shared conventions

- **Tokenizer.** All sets use the GPT-2 tokenizer in `models/gpt2-small` with `add_special_tokens=False`. No BOS or EOS token is added anywhere.
- **`token_ids` is authoritative.** Experiments read `token_ids` and never re-encode `text`.
  - WikiText `text` is a decode of `token_ids`, for display only.
  - Task `text` is the rendered template string that was encoded to produce `token_ids`.
- **Parquet columns.**
  - Both WikiText files have `sample_id, group, token_count, start_token, token_ids, text`.
  - The task file adds `source, source_id, category, format_class`.
  - `group` is `history` or `future`. Histories come first (`sample_id` 0 … H−1), then futures.
- **Build metadata.** Each Parquet file stores its build settings as JSON under the `lm_svd` schema-metadata key:

  ```python
  import json, pyarrow.parquet as pq
  meta = json.loads(pq.ParquetFile(path).schema_arrow.metadata[b"lm_svd"])
  ```

- **Model input.** The model input for a pair is always `history.token_ids + future.token_ids`.

## 3. WikiText

### 3.1 Source and tokenization

The source is [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-103-raw-v1`, split `train`, at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. It has two shards in `data/original/wikitext/wikitext-103-raw-v1/`, each with 900,675 rows (1,801,350 rows total).

All three WikiText sample sets share one tokenization, `tokenize_stream` in `src/data_pipline/windows.py`:

1. Read the shards in sorted filename order and their rows in file order.
2. Encode each row independently.
3. Concatenate all rows' token IDs into one stream. No separator or EOS is inserted. Raw WikiText rows already end in `\n`, empty rows contribute 0 tokens, and article titles appear as rows like ` = Title = \n`.

The result has 117,920,140 tokens. `build-token-stream` saves exactly this stream; `build-windows` and `build-wikitext-large` rebuild the same stream in memory and cut windows from it.

### 3.2 Fixed windows: `wikitext` and `wikitext-large`

`select_windows` in `src/data_pipline/windows.py`, with defaults of 64-token histories, 32-token futures and `seed=0`:

1. **Slots.** Split the stream into consecutive slots of `max(64, 32) = 64` tokens. That gives `117920140 // 64 = 1,842,502` slots; the last 12 tokens are unused.
2. **Order.** Visit slots in the order of `numpy.random.default_rng(0).permutation(1842502)`.
3. **Fill histories, then futures.** The first slots visited become histories until the history count is reached; the following slots become futures.
4. **Window = slot prefix.** A history takes all 64 tokens of its slot. A future takes the **first 32 tokens** of its slot, and the rest of that slot is unused. Since slots are disjoint, no two windows overlap.
5. **Exact dedup.** A slot is skipped if its token tuple equals a window already taken.
6. **Sort.** Histories are sorted by `start_token`, then futures are sorted by `start_token`. `start_token` is the window's offset in the stream and is always a multiple of 64.

Windows ignore row and article boundaries, so a window can start mid-word, in the middle of a sentence, or span two articles. Histories and futures come from unrelated places in the corpus, so the junction in a concatenated pair is usually not grammatical.

### 3.3 How `wikitext` and `wikitext-large` relate

Both files use the same stream, seed and permutation, and differ only in counts (512 + 512 vs 10000 + 10000). Because histories are filled first from the same permutation:

- every `wikitext` history is also a `wikitext-large` history;
- every `wikitext` **future** is a `wikitext-large` **history**, because those slots are among the first 10000 visited;
- the two files share **no** futures.

All three statements, and slot alignment of every `start_token` in both files, were checked against the stored files.

### 3.4 Token stream for exp6–exp9

Nothing is pre-sampled for the streaming experiments. `TokenStream.sample_starts` in `src/utils/stream_neural_svd.py` draws window starts **uniformly with replacement** from `[0, N − window]`. This differs from the fixed windows:

- starts are not slot-aligned;
- windows may overlap or repeat;
- histories and futures are drawn independently, so they are not adjacent in the corpus.

Two independent generators are used (CLI defaults shown):

| Purpose | Generator | Draws |
|---|---|---|
| Fixed probe set (evaluation) | `torch.Generator().manual_seed(1)` (`--probe-seed`) | 1024 history starts (64 tokens), then 1024 future starts (32 tokens) |
| Training batches | `torch.Generator().manual_seed(0)` (`--seed`) | per step: 64 history starts, then 64 future starts |

Each training step evaluates all 64 × 64 history/future pairs with GPT-2, so every step sees a fresh block. Probe starts are saved in `train/probe.npz`.

### 3.5 Worked examples

#### History `sample_id` 0 (`wikitext.parquet`)

Stored row:

| Field | Value |
|---|---|
| `group` | `history` |
| `start_token` | 72960 = slot 1140 × 64 |
| `token_count` | 64 |
| `token_ids[:12]` | `[82, 18598, 656, 1964, 27690, 837, 290, 262, 29355, 326, 5107, 262]` |

`text`:

```text
s descent into political disgrace , and the betrayal that forms the crux of both Urania 's storyline and that of Trujillo himself . 
 The second and third storylines are set in 1961 , in the weeks prior to and following Trujillo 's assassination on 30 May . Each assassin has his own background
```

Where it comes from: stream tokens 72960–73023 span two source rows of `train-00000-of-00002.parquet`.

| Row | Row starts at stream token | Row tokens | Window starts at row token | Window tokens taken from row |
|---|---|---|---|---|
| 1176 | 72759 | 232 | 201 | 31 (row end) |
| 1177 | 72991 | 190 | 0 | 33 |

Row 1176 (the window starts at its 201st token, inside `father 's descent`):

```text
 The Feast of the Goat begins with the return of Urania to her hometown of Santo Domingo , a city which had been renamed Ciudad Trujillo during Trujillo 's time in power . This storyline is largely introspective and deals with Urania 's memories and her inner turmoil over the events preceding her departure from the Dominican Republic thirty @-@ five years earlier . Urania escaped the crumbling Trujillo regime in 1961 by claiming she planned to study under the tutelage of nuns in Michigan . In the following decades , she becomes a prominent and successful New York lawyer . She finally returns to the Dominican Republic in 1996 , on a whim , and finds herself compelled to confront her father and elements of her past she has long ignored . As Urania speaks to her ailing father , Agustin Cabral , she recalls more and more of the anger and disgust that led to her thirty @-@ five years of silence . Urania retells her father 's descent into political disgrace , and the betrayal that forms the crux of both Urania 's storyline and that of Trujillo himself . 
```

Row 1177 (the window takes its first 33 tokens and ends inside `background story`):

```text
 The second and third storylines are set in 1961 , in the weeks prior to and following Trujillo 's assassination on 30 May . Each assassin has his own background story , explaining his motivation for his involvement in the assassination plot . Each has been wronged by Trujillo and his regime , by torture and brutality , or through assaults on their pride , their religious faith , their morality , or their loved ones . Vargas Llosa weaves the tale of the men as memories recalled on the night of Trujillo 's death , as the conspirators lie in wait for " The Goat " . Interconnected with these stories are the actions of other famous Trujillistas of the time : Joaquín Balaguer , the puppet president ; Johnny Abbes García , the merciless head of the Military Intelligence Service ( SIM ) ; and various others — some real , some composites of historical figures , and some purely fictional . 
```

The first tokens of the window show the mid-word start. The `'` before `s` belongs to the previous slot.

| Position | ID | Token |
|---|---|---|
| 0 | 82 | `'s'` |
| 1 | 18598 | `' descent'` |
| 2 | 656 | `' into'` |
| 3 | 1964 | `' political'` |
| 4 | 27690 | `' disgrace'` |
| 5 | 837 | `' ,'` |
| 6 | 290 | `' and'` |
| 7 | 262 | `' the'` |

The row boundary appears inside the window as the tokens `' .'`, `' '`, `'\n'` (IDs 764, 220, 198): row 1176 ends with ` . \n`, and row 1177 begins with ` The`.

#### Future `sample_id` 512 (`wikitext.parquet`)

| Field | Value |
|---|---|
| `group` | `future` |
| `start_token` | 219840 = slot 3435 × 64 |
| `token_count` | 32 (first half of the slot) |
| source | `train-00000-of-00002.parquet` row 3306 (189 tokens), row tokens 156–187 |

`token_ids`:

```text
[764, 383, 968, 2488, 12, 31, 8936, 1912, 616, 66, 7451, 11646, 2332, 36671, 34325, 11777, 11172, 262, 4693, 736, 284, 262, 34306, 2269, 459, 6582, 287, 16994, 837, 11170, 1058, 220]
```

`text`:

```text
 . The New @-@ Zealand based mycologist Gordon Herriot Cunningham explicitly transferred the species back to the genus Geastrum in 1944 , explaining : 
```

Source row 3306; the window is its last 32 tokens apart from the final 1:

```text
 Because this species resembles the earthstar fungi of Geastrum , it was placed in that genus by early authors , starting with Christian Hendrik Persoon in 1801 ( as Geaster , an alternate spelling of Geastrum ) . According to the American botanist Andrew P. Morgan , however , the species differed from those of Geastrum in not having open chambers in the young gleba , having larger and branched capillitium threads , not having a true hymenium , and having larger spores . Accordingly , Morgan set Persoon 's Geaster hygrometricum as the type species of his new genus Astraeus in 1889 . Despite Morgan 's publication , some authorities in the following decades continued to classify the species in Geastrum . The New @-@ Zealand based mycologist Gordon Herriot Cunningham explicitly transferred the species back to the genus Geastrum in 1944 , explaining : 
```

`@-@` is WikiText's escaped hyphen; it is kept as-is and tokenizes as `' @'`, `'-'`, `'@'` (IDs 2488, 12, 31).

#### The pair as GPT-2 sees it

exp1–exp3 feed history 0 + future 512 as one 96-token input. Decoded:

```text
s descent into political disgrace , and the betrayal that forms the crux of both Urania 's storyline and that of Trujillo himself . 
 The second and third storylines are set in 1961 , in the weeks prior to and following Trujillo 's assassination on 30 May . Each assassin has his own background . The New @-@ Zealand based mycologist Gordon Herriot Cunningham explicitly transferred the species back to the genus Geastrum in 1944 , explaining : 
```

The junction `background . The New` joins two unrelated articles.

#### Token-stream windows (exp6–exp9)

These four windows are what a fresh exp6–exp9 run with default flags draws first. They were produced with the exact `sample_starts` calls listed in §3.4.

| Window | Start | Slot-aligned | Source rows |
|---|---|---|---|
| probe history 0 (seed 1) | 22,294,690 | no | shard 00000, rows 342673–342678 |
| probe future 0 (seed 1) | 8,820,029 | no | shard 00000, row 136237, row tokens 62–93 |
| step-1 history 0 (seed 0) | 116,654,581 | no | shard 00001, row 881044, row tokens 66–129 |
| step-1 future 0 (seed 0) | 72,429,031 | no | shard 00001, row 206059, row tokens 229–260 |

Probe history 0 crosses an article boundary:

- it takes the last 47 tokens of row 342673;
- it passes two empty rows (0 tokens each);
- it takes the full 9-token title row 342676 (` = Angel ( Madonna song ) = `);
- it skips another empty row;
- it ends with the first 8 tokens of row 342678.

```text
 TV advertisements for Lyons Maid and Kellogg 's . Aspects of Thunderbirds have since been used in advertising for Swinton Insurance , Nestlé Kit Kat , Specsavers and the UK Driver and Vehicle Licensing Agency . 
 = Angel ( Madonna song ) = 
 " Angel " is a song by American
```

Probe future 0:

```text
 been perfect soap fare . What does EastEnders have ? Another murder investigation in a place that has a higher homicide rate than The Wire 's Baltimore . Andrew
```

Step-1 history 0:

```text
 the earliest sculptures discovered in Igboland . Here , the grave of a well @-@ established man of distinction and a ritual store , dating from the 9th century AD , contained both chased copper objects and elaborate castings of leaded bronze . Along with these bronzes were 165 @,@ 000 glass beads said
```

Step-1 future 0:

```text
 the day after the cancellation , hinting that the performance would be rescheduled for a later date by assuring fans that " your call will be answered "
```

## 4. Task single-text

### 4.1 Inputs

`build-task-inputs` (`src/data_pipline/task_inputs.py`) reads only two sources.

**Task catalog** `data/original/task_sources/single_text.jsonl`:

- 1309 JSON lines, each with `id`, `source`, `category`, `instruction`;
- by source: 213 `sni` (Super-NaturalInstructions), 88 `tasksource`, 8 `flan`, 1000 `synthetic`;
- the catalog is frozen and its SHA-256 is recorded in the output metadata. It was produced beforehand: source tasks were filtered with heuristics for "one arbitrary prose document" inputs, and synthetic instructions were added one at a time with the `synthesize-single-text-tasks` skill. The builder does not modify it.

**C4 documents** `data/original/c4/en/c4-validation.00000-of-00008.json.gz`:

- `allenai/c4` English validation shard 00000 at revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`;
- 45,576 records, each with `text`, `timestamp`, `url`.

Histories and futures are built independently; any history can be paired with any future.

### 4.2 Histories

1. **Render.** Each catalog instruction is rendered as `Instruction: {instruction}` with no trailing whitespace, then encoded.
2. **Check** (`finalize_tasks`):
   - duplicate instructions are rejected, compared after lowercasing and keeping only word characters;
   - rendered histories longer than 256 tokens are rejected.

   Any rejection **aborts the build**, so the catalog must already be clean.
3. **Order** (`order_by_bucket`, seed 0):
   - group tasks by `category`; there are 30 categories, because each `sni` task keeps its SNI category while all `tasksource`, `flan` and `synthetic` tasks share one category per source;
   - shuffle each category with seed `0 + i`, where `i` is the category's index in sorted order;
   - interleave the categories round-robin.

   Leading prefixes are therefore category-balanced. With exp1–exp3 `--history-count 30`, you get exactly one task per category.

### 4.3 Futures

`select_passages` reads C4 records in file order and builds one candidate passage per record:

1. Split `text` into lines, strip them, and drop blank lines.
2. Append lines one at a time until the joined passage (lines joined with `\n`) reaches at least 96 tokens. Later lines are discarded.
3. Reject the record with the **first** failing rule, in this order:

| Reason | Rule (passage = the selected lines) |
|---|---|
| `short` | fewer than 96 tokens even with all lines |
| `long` | more than 160 tokens (the line that crossed 96 overshot) |
| `fragment_start` | first character is not uppercase and not a digit |
| `incomplete_line` | some selected line does not end in `.`, `!` or `?` (optionally followed by `"`, `'` or `)`) |
| `short_line` | some selected line has fewer than 4 words |
| `forbidden_content` | contains `http`, `www.`, `@`, `©`, `cookie`, `{`, `}`, `\|`, `click here` or `subscribe` (case-insensitive) |
| `non_ascii` | more than 1% of characters are non-ASCII |
| `low_alphabetic` | fewer than 75% of characters are letters |
| `uppercase` | more than 15% of letters are uppercase |
| `repeated_punctuation` | contains a run of two or more `!`/`?` characters, such as `!!`, `?!` or `!?!` |
| `duplicate_text` | same whitespace-normalized text as an accepted passage |
| `duplicate_domain` | same URL domain (lowercased, `www.` removed) as an accepted passage |

4. Shuffle the accepted passages with `default_rng(0).permutation` and keep the first 1309.
5. Render each as `\n\nDocument: {passage}\n\nResponse:` and encode it.

Counts from the build metadata (`passage_reject_counts`):

| Reason | Rejected records |
|---|---|
| `short` | 8,488 |
| `long` | 7,741 |
| `duplicate_domain` | 1,513 |
| `fragment_start` | 1,466 |
| `non_ascii` | 1,411 |
| `forbidden_content` | 1,221 |
| `low_alphabetic` | 1,146 |
| `uppercase` | 318 |
| `repeated_punctuation` | 307 |
| `incomplete_line` | 264 |
| `short_line`, `duplicate_text` | 0 (not recorded) |
| **Total rejected** | **23,875** |

Derived from these counts: 45,576 − 23,875 = 21,701 passages were accepted, of which 1309 were sampled.

### 4.4 Build-time checks

- The longest history plus the longest future must fit in the tokenizer's `model_max_length` (1024 for GPT-2).
- **Concatenation safety:** for 2000 seeded random pairs, `encode(history) + encode(future)` must equal `encode(history + future)`. This is a sampled check that concatenating stored `token_ids` gives the same tokens as encoding the combined text; it is not run on every pair.

Variable lengths make this set unsuitable for exp4, which requires equal-length inputs within a block.

### 4.5 Worked examples

#### Histories: one per source

The table shows the lowest `sample_id` from each source. `sample_id` 27–29 are the last three entries of the first round-robin round.

| `sample_id` | Source | Category | Tokens |
|---|---|---|---|
| 0 | `sni` | Ethics Classification | 82 |
| 27 | `flan` | flan | 9 |
| 28 | `synthetic` | synthetic | 29 |
| 29 | `tasksource` | tasksource | 84 |

**`sample_id` 0, `sni`.** Catalog line:

```json
{"id": "task498_scruples_anecdotes_whoiswrong_classification", "source": "sni", "category": "Ethics Classification", "instruction": "Given a real-life anecdote of a complex ethical situation, identify who is wrong here. Classify each post into the following categories : AUTHOR (if the author of the post is wrong in the situation), OTHER (if any of the others mentioned in the paragraph is wrong), EVERYBODY (if everyone mentioned is wrong) and NOBODY (if no one is wrong in the situation)."}
```

Stored `text` (82 tokens; `token_ids[:12]` = `[6310, 2762, 25, 11259, 257, 1103, 12, 6042, 49533, 286, 257, 3716]`):

```text
Instruction: Given a real-life anecdote of a complex ethical situation, identify who is wrong here. Classify each post into the following categories : AUTHOR (if the author of the post is wrong in the situation), OTHER (if any of the others mentioned in the paragraph is wrong), EVERYBODY (if everyone mentioned is wrong) and NOBODY (if no one is wrong in the situation).
```

Every history begins with the same three tokens, `Instruction` + `:` = `[6310, 2762, 25]`.

**`sample_id` 27, `flan`.** Catalog line:

```json
{"id": "cnn_dailymail#0", "source": "flan", "category": "flan", "instruction": "Write highlights for this article:"}
```

Stored `text` (9 tokens; `token_ids` = `[6310, 2762, 25, 19430, 11330, 329, 428, 2708, 25]`):

```text
Instruction: Write highlights for this article:
```

**`sample_id` 28, `synthetic`.** Catalog line:

```json
{"id": "syn-0959", "source": "synthetic", "category": "synthetic", "instruction": "Count the adjacent pairs of alphabetic characters in the document that are identical ignoring case, and return the total as an integer."}
```

Stored `text` (29 tokens):

```text
Instruction: Count the adjacent pairs of alphabetic characters in the document that are identical ignoring case, and return the total as an integer.
```

**`sample_id` 29, `tasksource`.** Catalog line:

```json
{"id": "HelpSteer3/preference_strength", "source": "tasksource", "category": "tasksource", "instruction": "Which response is the better next assistant reply, and by how much? With no explanation, answer with either \"-1: Response 1 is slightly better\", \"-2: Response 1 is better\", \"-3: Response 1 is much better\", \"0: About the same\", \"1: Response 2 is slightly better\", \"2: Response 2 is better\" or \"3: Response 2 is much better\"."}
```

Stored `text` (84 tokens):

```text
Instruction: Which response is the better next assistant reply, and by how much? With no explanation, answer with either "-1: Response 1 is slightly better", "-2: Response 1 is better", "-3: Response 1 is much better", "0: About the same", "1: Response 2 is slightly better", "2: Response 2 is better" or "3: Response 2 is much better".
```

#### History order: the first round-robin round

`sample_id` 0–29 cover all 30 categories once, in sorted category order:

```text
Ethics Classification, Grammar Error Correction, Grammar Error Detection, Information Extraction, Irony Detection, Keyword Tagging, Linguistic Probing, Misc., Named Entity Recognition, Paraphrasing, Punctuation Error Detection, Question Generation, Sentence Composition, Sentence Compression, Sentence Expansion, Sentiment Analysis, Spelling Error Detection, Story Composition, Style Transfer, Summarization, Text Categorization, Text Completion, Text Quality Evaluation, Text Simplification, Title Generation, Toxic Language Detection, Translation, flan, synthetic, tasksource
```

Round 2 begins with `Ethics Classification, Grammar Error Detection, Information Extraction, Irony Detection`. `Grammar Error Correction` is missing because its bucket has only one task. As small categories run out, later rounds contain fewer buckets; from `sample_id` 397 onward, every history is `synthetic`.

#### Future `sample_id` 1309: from C4 record to stored future

| Field | Value |
|---|---|
| `source_id` | `http://miiu.org/fold-away-furniture/vision/64d4/slat-back-folding-fold-away-kitchen-bar-stools-breakfast-foldaway-argos-chairs-up-amusing-wooden.html` |
| `category` (domain) | `miiu.org` |
| C4 record | line 7149 (0-based) of the shard |
| Nonblank lines in record | 5 |
| Lines kept | 1 — line 0 alone already has 134 tokens (≥ 96 and ≤ 160) |
| Future `token_count` | 142 = 134 passage tokens + 8 template tokens |

Line 0 of the record, which becomes the passage. It ends in `.`, has more than 4 words, contains no forbidden content, and its only non-ASCII character (`’`) is under 1%:

```text
Vainness will place in spot that’s humid, high wet and traffic. The vanity material you choose should stand along with the atmosphere. Wooden veneer, thermo foils and laminate vanity is material is effective well in furniture. Durable high vanity ensures you stay need for easy clean vanity top material. The next is storage solution that you simply have to have in your own furniture. First of all, take list for what you will shop inside your vanity. Organize objects you need to reach and also everything will nearby to make you easier to slat back folding fold away kitchen bar stools breakfast foldaway argos chairs up amusing wooden.
```

The remaining 4 lines of the record are discarded. The rules check form, not quality: this SEO-style text passes them all.

#### A full task prompt

History `sample_id` 1 (`sni`, Grammar Error Correction, 28 tokens) + future `sample_id` 1309 (142 tokens) = 170 tokens. The pair is concatenation-safe: 28 + 142 tokens encode identically to the combined string.

```text
Instruction: In this task, you will be shown an incorrect English sentence. You need to generate a corrected form of the input sentence.

Document: Vainness will place in spot that’s humid, high wet and traffic. The vanity material you choose should stand along with the atmosphere. Wooden veneer, thermo foils and laminate vanity is material is effective well in furniture. Durable high vanity ensures you stay need for easy clean vanity top material. The next is storage solution that you simply have to have in your own furniture. First of all, take list for what you will shop inside your vanity. Organize objects you need to reach and also everything will nearby to make you easier to slat back folding fold away kitchen bar stools breakfast foldaway argos chairs up amusing wooden.

Response:
```

#### Rejected C4 documents: first example of each reason

How these examples were found: each C4 record was passed alone, in file order, to `select_passages`, and the first record rejected for each reason was kept. Per-document rules do not depend on other records, so the reasons match the real build. `duplicate_domain` depends on earlier records: the example is the first record that passes every per-document rule but whose domain was already used by an earlier accepted passage.

Each excerpt is the candidate passage (the selected lines), truncated to 300 characters. `short_line` and `duplicate_text` have no example: both rules exist in the code, but no record was rejected for either in this build.

**`short`.** Line 5, `https://www.barcelona-tourist-guide.com/en/hotels/reviews/citadines-hotel-barcelona.html`. The whole document is only 83 tokens.

```text
Stayed here about four years ago and they helped me out this weekend. Thankfully Citadines had a double apartment room for our stay. Usual facilities and a cooker, dishwasher, iron etc. We had a large view of the cathedral and port area and Montjuïc. Fabulous. Non fussy and perfectly functional. 3 s…
```

**`long`.** Line 2, `http://synergistic-technologies.com/biocmeisner.html`. The line that crossed 96 tokens brought the passage to 328 tokens.

```text
CHARLES H. MEISNER, one of the prosperous and highly-esteemed business men of Birnamwood, Shawano county, was born in Newton township, Manitowoc Co., Wis., February 14, 1860, and is a son of Frederick and Magarette (Steltzer) Meisner, both natives of Germany.
Frederick Meisner was born February 5, 1…
```

**`fragment_start`.** Line 1, `https://jmtd.net/log/geek/linux/ui/`. The passage starts with lowercase `m`.

```text
messaging system designed as a plugin-based system.
Doubtless the kopete people would disagree with me here, but I'd say that their project was utterly redundant. What advantages does kopete bring that could not have been submitted to gaim itself? Why go to the time and effort to independently devel…
```

**`incomplete_line`.** Line 101, `http://bugs.call-cc.org/timeline?from=2010-05-06T08%3A28%3A04%2B02%3A00&amp;precision=second`. The first offending selected line is `awful.setup (trunk): use "-O3 -lambda-lift" instead of "-O2"`; it ends in `"` without a preceding `.`, `!` or `?`.

```text
Yet another fix for <tags>.
Added <examples> tag (thanks to Jim for the tip).
spiffy-request-vars: moving code to trunk.
spiffy-request-vars: adding trunk & tags dirs.
​http://chicken.wiki.br/cross-compilation may be a start.
This would make it easier to conditionalize code for a particular version.…
```

**`forbidden_content`.** Line 72, `https://www.petrescue.com.au/listings/487192`. The first forbidden match is `@`, beyond the excerpt.

```text
Camilla is a very gentle cat and she gets on well with both other cats and with dogs. She does not like small children. She's arrived back at the shelter after being in foster care for some time. She does well as an inside cat provided she has access to an outside enclosure attached to the house so …
```

**`non_ascii`.** Line 6, `https://butterflyinsight.com/`. 5 of 468 characters (curly quotes) are non-ASCII, just over the 1% limit.

```text
It’s the end of the world,” said the caterpillar.
“It’s the beginning of the world,” said the butterfly.
In the life cycle of a butterfly, it starts out as an egg that turns into a caterpillar. Then it becomes a cocoon before it turns into a beautiful creature capable of fluttering free through the …
```

**`low_alphabetic`.** Line 15, `https://www.company-detail.com/company-dolphin-fashion-corp-2297344`. 282 of 402 characters are letters (70%).

```text
DOLPHIN FASHION CORP. is an entity registered at NEW YORK county with company number 2297344. DOLPHIN FASHION CORP. located at the address Suite #1007 110 West 40Th St New York, New York, 10018. Company is incorporated on September 14, 1998. Current status of the company is inactive - dissolution by…
```

**`uppercase`.** Line 523, `https://www.coloradomidwaymotors.com/2012-JEEP-LIBERTY-Wheat-Ridge-CO-1805?nonmobile=0&force_mobile=1`. 124 of 469 letters are uppercase (26%).

```text
VERY NICE JEEP! SPORT PACKAGE, AUTOMATIC, PRIVACY GLASS, PREMIUM WHEELS, NEW TIRES, JUST SERVICED, CLEAN CAR FAX, AWESOME VALUE IN THIS NICE 4 X 4 ! We're here to help!! Colorado's Midway motors, Inc. offers competitive financing nationwide. Even if you already have financing, in most cases we can m…
```

**`repeated_punctuation`.** Line 125, `https://www.b78coaching.com/2014/10/sabrina-macleod/`. Match: `!?!`.

```text
Whether it be about your nutrition, hydration, preparation or challenging physical boundaries….never stop being willing to learn.
What!?! There’s another season coming? Can I finish this one first?
I didn’t die. And since we are still in the season, I am fairly confident I will accomplish this…there…
```

**`duplicate_domain`.** Line 609, `https://www.nps.gov/noca/blogs/cascade-pass.htm`. This passes every per-document rule, but domain `nps.gov` was already taken by the record at line 162.

```text
Conditions for climbs in the Cascade Pass area including Sahale, Buckner, and Ptarmigan Traverse can be found below. For conditions on other climbs, return to Planning a Climb.
Condition reports come from climbing and wilderness rangers, the voluntary climbing register, and other climbers. To make a…
```

## 5. Reproducing an example

```python
import numpy as np
import pyarrow.parquet as pq

rows = pq.read_table("data/history_future_samples/wikitext.parquet").to_pylist()
row = rows[0]  # sample_id == list index
stream = np.load("data/token_streams/wikitext-103-train.npy", mmap_mode="r")
start, n = row["start_token"], row["token_count"]
assert stream[start:start + n].tolist() == row["token_ids"]
print(row["group"], start, row["text"])
```

For the task file, use `data/history_future_samples/task-single-text.parquet`. There `start_token` is always −1, and `source_id` points back to the catalog `id` (histories) or the C4 `url` (futures).
