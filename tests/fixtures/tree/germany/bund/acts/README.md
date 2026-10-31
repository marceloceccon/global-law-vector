# German law, change by change

German law as one git repository. Each commit is one law taking effect, credited to that law, so `git log` is the history of the law and `git blame` says which law wrote each line.

| Part | Folder | From | Text |
| --- | --- | --- | --- |
| The Grundgesetz | [`grundgesetz/`](grundgesetz) | 23 May 1949 | one file per article, every version, each amending law dated by the day it took effect |
| All other federal law (6147 acts) | [one folder per act (`bgb/`, `stgb/` …)](verzeichnis.json) | 10 June 2019 | one file per act, every change published on Gesetze im Internet since then |
| State law (Landesrecht) | `laender/` | | not yet included |

*Deutsch: [unten](#deutsch).*

## A lawyer's questions, as git commands

| Question | Command |
| --- | --- |
| What did Article 16 GG say on 1 June 1993? | `python tools/stand_am.py 1993-06-01 16` |
| What did the BGB say on 1 July 2021? | `python tools/stand_am.py 2021-07-01 --gesetz bgb` |
| Which law wrote each paragraph of Article 1 GG? | `git blame grundgesetz/artikel/001.md` |
| Which words did each change to the StGB replace? | `git log -p --word-diff stgb/` |
| What did one amending law change, across all acts? | `git log --author='G v. 22.12.2025' --stat` |
| When did each change take effect? | `git log --format='%(trailers:key=In-Kraft,valueonly) %s'` |
| Which Grundgesetz articles were amended most often? | `git log --format= --name-only -- grundgesetz/artikel \| sort \| uniq -c \| sort -rn \| head` |
| Where is the reader for a Grundgesetz amendment? | the `Reader:` trailer, e.g. `git log -1 --format='%(trailers:key=Reader,valueonly)' gg/36` |
| The whole Grundgesetz on a day | `git checkout "$(git tag --list 'in-kraft/*' \| awk -v d='in-kraft/1990-10-03' '$0 <= d' \| tail -1)"` |

In PowerShell, the last one is `git checkout (git tag --list 'in-kraft/*' | Where-Object { $_ -le 'in-kraft/1990-10-03' } | Select-Object -Last 1)`.

After a rebuild of this repository (see `REPRODUCE.md`), refresh an existing clone with `git fetch --force --tags`, or the old tags stay.

## The dates, and which one is used

A law has several dates: passed by the Bundestag (`Beschlossen`), signed (`Ausgefertigt`), promulgated in the Bundesgesetzblatt (`Verkuendet`) and in force (`In-Kraft`). The date in force decides which text applies, so commits are dated by it, at noon in Berlin. Where a law sets no date, Article 82(2) of the Grundgesetz puts it in force on the fourteenth day after the Gazette issue. The trailer keys are ASCII so that git can read them.

- **Grundgesetz commits** carry all four dates, from the sources below. Git cannot store a date before 1970: commits for changes that took effect before then are dated 1 January 1970 (UTC), a second apart in order, and their real dates are in the message and in the `in-kraft/YYYY-MM-DD` tags.
- **Federal-law commits** are dated, and carry `In-Kraft:`, by the day the change took effect in the consolidated text: the day Gesetze im Internet published it, which it does as changes take effect (within a week until May 2020, within a day since). That keeps the history in date order and lets it only grow. Where the act's own footnotes give an earlier in-force date (`mWv 21.5.2003`), for a late consolidation or a change with retroactive effect, the commit also carries `In-Kraft-laut-Fussnote:`. `Beobachtet:` is the exact time it was first seen. The tags `stand/YYYY-MM-DD` mark every act as published that day.

## What "author" means

The author of each commit is the law that made the change: `36. Änderung (1990)` for the 36th amending law to the Grundgesetz, or `G v. 22.3.2025 I Nr. 94` (Gesetz vom 22. März 2025, Bundesgesetzblatt 2025 Teil I Nr. 94) for federal law, as the act's Stand note names it. The addresses end in `.invalid`. Changes no law explains are never credited to a law:
- the first state of the federal acts is `Gesetze im Internet (Bestand)`;
- text that changed while the act's Stand note still names the same law is `Textänderung ohne neue Standangabe`;
- acts added or no longer published without a law named are `Neu aufgenommen` and `Nicht mehr veröffentlicht`.

## Where the text comes from, and how it is checked

- **The Grundgesetz to 2019** comes from [DocPatch](https://github.com/c3e/grundgesetz-dev), which records each amending law to the 63rd as a patch. Its 2019 text was compared with the official text on [Gesetze im Internet](https://www.gesetze-im-internet.de/gg/), article by article. They differed in 43 articles, some in wording (Article 3(3) lacked the 1994 sentence on disability). For each, the official text replaces DocPatch's from the article's last change on. Every correction is listed in [`grundgesetz/KORREKTUREN.md`](grundgesetz/KORREKTUREN.md).
- **The Grundgesetz from 2019, and all federal law,** come from the official consolidated text on Gesetze im Internet (Federal Ministry of Justice), which keeps only the current version, through [QuantLaw's archive](https://github.com/QuantLaw/gesetze-im-internet), which has stored it weekly since 10 June 2019 and daily since May 2020.
- **Checked:** the Grundgesetz's final text equals today's official text, article for article. Builds are deterministic: the same sources give the same commits.

## What it does not show

- Federal law other than the Grundgesetz before 10 June 2019. Rebuilding that from the Bundesgesetzblatt is planned.
- Versions of a Grundgesetz article older than its last change before 2019, which are DocPatch's transcription and were not checked against the Bundesgesetzblatt.
- The exact day a federal change took effect where the footnotes do not give it, and the stages of a law that takes effect in several steps under one Stand note.
- Footnotes and images. State law. EU law.

The consolidated texts are not the official promulgation. Only the Bundesgesetzblatt is authoritative. This is a research aid.

## Licence

German statutes are not protected by copyright (§ 5 Abs. 1 UrhG). DocPatch's own content is CC BY-SA 3.0 DE; QuantLaw's archive is BSD-3-Clause. This repository is released under CC0 1.0 ([`LICENSE`](LICENSE)). It is built by Lexifina Watch with the scripts in [`tools/`](tools). [`REPRODUCE.md`](REPRODUCE.md) records every source, command, checksum and decision, and what an AI model did and did not do.

---

## Deutsch

Deutsches Recht als ein Git-Repository. Jeder Commit ist ein Gesetz, das in Kraft tritt, mit diesem Gesetz als Autor.

- `grundgesetz/`: das Grundgesetz seit 1949, eine Datei je Artikel, jedes Änderungsgesetz auf den Tag seines Inkrafttretens datiert.
- one folder per act (`bgb/`, `stgb/` …): das übrige Bundesrecht, eine Datei je Gesetz, jede Änderung auf Gesetze im Internet seit dem 10. Juni 2019.
- `laender/`: Landesrecht folgt.

```sh
python tools/stand_am.py 1993-06-01 16             # Artikel 16 GG vor der Asylrechtsänderung 1993
python tools/stand_am.py 2021-07-01 --gesetz bgb   # das BGB am 1. Juli 2021
git blame grundgesetz/artikel/001.md                   # welches Gesetz welchen Absatz geschrieben hat
```

Commits tragen die Trailer `Beschlossen`, `Ausgefertigt`, `Verkuendet` und `In-Kraft`. Der Text des Grundgesetzes bis 2019 stammt aus DocPatch und wurde mit dem amtlichen Text abgeglichen; alles Übrige ist der amtliche konsolidierte Text von Gesetze im Internet. Maßgeblich ist allein das Bundesgesetzblatt.
