# Contributing to Nova-Violet Role

This applies to every repository in the organisation. Individual repositories
may add stricter rules; none may relax these.

---

## The one rule everything else follows from

> **A claim that has not been measured does not ship.**

Not "should be measured". Does not ship. Our documentation states counts,
timings and exit codes because readers are meant to re-run them. If you write a
number, you ran the command that produced it.

## The most valuable contribution you can make

**Prove one of our claims false.**

Every repository carries an issue form for exactly this, and a report that
falsifies a documented claim is treated as a defect in the documentation — not
as an inconvenience, and not as an attack. It is the fastest way to improve any
of these projects, and it is credited in the changelog.

If you are looking for a way in and do not know where to start: pick a number
in a README, try to reproduce it, and tell us what you got.

---

## Before you open a pull request

### 1. Read the exit code directly

```sh
bash your/test/command; echo "exit=$?"
```

Never through a pipe. `cmd | grep ...` reports **grep's** status, not `cmd`'s,
and that mistake has produced a false green in this organisation's own history.
If you must pipe, read `${PIPESTATUS[0]}`.

### 2. Watch your test fail

A test you have never seen fail is not evidence. Break the code on purpose,
watch the assertion fire, restore it, watch it pass. Say in the pull request
that you did this.

**A test that survives no mutation is vacuous.** This is not a stylistic
preference — a gate that passes whether or not the property holds is worse than
no gate, because it reports safety it never checked.

### 3. Prove the negative control fires

If your check has a negative control, assert the mutation is genuinely present
before running, so "the mutation did not apply" can never be mistaken for "the
property survived". Silence is not a pass.

### 4. Say what you could not verify

"I have no macOS machine; CI covers it" is a good answer. Silence is not.

An honest gap is a result. A fabricated green is a conduct breach — see the
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Encoding — non-negotiable across the organisation

Every text file is **UTF-8** with **LF** line endings, no BOM, no CR anywhere,
including the final line.

```sh
git ls-files -z | xargs -0 grep -lI $'\r'    # must print nothing
```

Beware of measuring this wrong: shell escape sequences do not always survive
being passed through tooling, and an empty pattern matches **every** line, which
looks like total corruption. Count bytes instead when in doubt:

```sh
tr -dc '\r' < FILE | wc -c                   # must print 0
```

## Commits

- One logical change per commit.
- The subject line names the **lesson**, not the file touched. Compare
  `fix: update checker` with `the oracle read the host's settings, so only the
  passing turns could fail`. The second tells a reader why the commit exists.
- The body states what was measured, what gap was found, what was fixed, and the
  before/after numbers.
- Do not bypass a failing gate to land work. If you are deliberately recording
  work in progress, say so in the message — but never write a message that
  claims verification while a gate is red.

## Licensing

Contributions are accepted under the licence of the repository you are
contributing to. Across this organisation that is **AGPL-3.0-or-later OR
EUPL-1.2** unless a repository's `LICENSE` says otherwise — these are
**copyleft** licences, and that is deliberate. Check the repository's
`LICENSE` before you start; do not assume.

New source files carry an `SPDX-License-Identifier` header matching the
repository's choice.

## Formal proofs (repositories that carry them)

Some of these projects specify behaviour in **Lean 4**. Where they do:

- `lake build` and read the exit code **directly**.
- Zero `sorry`. Never `native_decide`.
- `#print axioms` on what you proved — `sorryAx` means it is not proved, and no
  axioms at all is usually a sign the statement is vacuous.
- `lake env leanchecker <Module>` re-runs the **kernel** over the compiled proof
  terms, independently of the elaborator that produced them. Exit 0 with zero
  output is the pass; a module with no build artefacts exits 1, which is the
  control proving the instrument can fail.
- Then mutate the definition, rebuild, and confirm the theorems **die**.

Three instruments, then a mutation. A green build alone is elaboration, not
truth.

---

## Getting help

Open a discussion in the repository's **Q&A** category. Questions are not noise;
an unanswerable question usually means our documentation failed.

*Nova-Violet Role · Reality is the judge.*
