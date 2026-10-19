# Senior Data Scientist: a private document agent

Resourcly reads industrial manufacturers' technical documents. This challenge is a small version of
that work. We care about how you solve it as much as about what you hand in.

## Time

4 hours of focused work. We are not timing you. If you run short, hand in less and say what you
dropped and why.

## Data

- `data/drawings/`: 11 technical drawings (PDF),
  [CTC-01](https://www.nist.gov/document/nist-cad-model-ctc-01),
  [CTC-02](https://www.nist.gov/document/nist-cad-model-ctc-02),
  [CTC-03](https://www.nist.gov/document/nist-cad-model-ctc-03),
  [CTC-04](https://www.nist.gov/document/nist-cad-model-ctc-04),
  [CTC-05](https://www.nist.gov/document/nist-cad-model-ctc-05),
  [FTC-06](https://www.nist.gov/document/nist-cad-model-ftc-06),
  [FTC-07](https://www.nist.gov/document/nist-cad-model-ftc-07),
  [FTC-08](https://www.nist.gov/document/nist-cad-model-ftc-08),
  [FTC-09](https://www.nist.gov/document/nist-cad-model-ftc-09),
  [FTC-10](https://www.nist.gov/document/nist-cad-model-ftc-10),
  [FTC-11](https://www.nist.gov/document/nist-cad-model-ftc-11).
- `data/reports/`: 3 PDF documents with more description, NTSB Materials Laboratory Factual Reports
  [16-006](https://data.ntsb.gov/Docket/Document/docBLOB?ID=40454212&FileExtension=.PDF&FileName=Materials+Laboratory+Factual+Report+16-006-Master.PDF),
  [18-061](https://piperowner.org/wp-content/uploads/2020/06/NTSBMaterialsTest.pdf),
  [24-013](https://data.ntsb.gov/Docket/Document/docBLOB?ID=17072191&FileExtension=pdf&FileName=Materials%20Laboratory%20Factual%20Report%2024-013-Rel.pdf).

## Task

Write an agent a user can chat with, plus the assessment and tracking around it.

1. **Input.** The user sends text, a picture, or text plus a picture (images or PDFs).
2. **A picture.** The agent says what is wrong and what to do, based on the documents in
   `data/reports/`.
3. **A technical drawing**, like those in `data/drawings/`. Instead of explaining, the agent returns a
   JSON file with every value on the drawing. For each value: what it belongs to on the drawing and
   what it means.
4. **Automatic.** The agent decides which case applies. The user never says what they want.
5. **Bulk upload.** The user can upload many drawings at once.
6. **Private.** All data stays in the system: no internet access, no data leakage.
7. **Scale.** There are only 14 documents here, for simplicity. Assume more than a million, all very
   alike. The system must still work.
8. **Tracking.** A technical person, not the user, can see how the agent performed: for every request,
   what it did and what it called. Stored, and easy to access.
9. **Validation.** Code that measures how well the agent works, for each important part (for example:
   choosing the case, reading a drawing, answering from the reports) and for the whole system. It is
   incremental: plug in a new version of the agent, or of one part, run it on a chosen number of test
   cases, and see whether it got better or worse. No interface needed.

## How you work

We want to see how you get there, not only where you end up.

- **Git.** Work in a git repository from the first hour. Commit as you go, with messages that say why.
  Do not squash.
- **`NOTES.md`.** Keep it as you work, not at the end: what you tried, what failed, what you measured,
  and why you chose what you did.
- **AI assistants** are allowed for building the system and its datasets, and they may see these
  documents (assume a corporate subscription). Include their session logs in `ai_logs/`. The system
  you hand in must still run fully in-house.
- **No GPU?** Colab and Kaggle notebooks give you one for free.

## What to send back

The git repository, containing:

- the code, with a README on how to set up and run each part: the chat, a bulk upload, looking up a
  request, and validating a new agent version;
- your evaluation data and results;
- the JSON output for the 11 drawings;
- `NOTES.md` and `ai_logs/`;
- one page on how your design handles privacy and scale.

## The review call

1 to 2 hours. You run it in front of us, then we go through your code, your notes and your numbers with
you. Be ready to explain every decision. "The assistant wrote it" is not an answer.
