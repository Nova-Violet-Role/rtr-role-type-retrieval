<!-- Delete any section that genuinely does not apply, and say why it does not.
     A deleted section with no explanation reads as an unanswered question. -->

## What this changes, and what question it answers

<!-- One paragraph. State the adversarial question your change answers —
     "can a check pass while the thing it checks is broken?" — not just the
     files you touched. -->

## The evidence

```
<!-- The command you ran, its tail, and its exit code. -->
```

<!-- Read the exit code DIRECTLY, never through a pipe. `cmd | grep …` reports
     grep's status, not cmd's. When you must pipe, use ${PIPESTATUS[0]}. -->

## Checklist

- [ ] The repository's test or gate suite exits **0**, and I read the exit code
      directly rather than through a pipe.
- [ ] Any behavioural change ships with a test **I have watched fail** — I broke
      the code on purpose, saw the assertion fire, restored it, saw it pass.
- [ ] If my check has a negative control, it asserts the mutation is genuinely
      **present** before running, so "did not apply" cannot be misread as
      "survived".
- [ ] No carriage returns: `git ls-files -z | xargs -0 grep -lI $'\r'` prints
      nothing. (When in doubt, count bytes: `tr -dc '\r' < FILE | wc -c`.)
- [ ] New files carry the repository's `SPDX-License-Identifier` header.
- [ ] Executable scripts are executable **in the git index**:
      `git ls-files -s path/ | grep -v '^100755'` prints nothing.
- [ ] I did not edit a check's expected value merely to make it pass. Any
      changed expectation is justified from first principles in this PR.
- [ ] Documentation numbers I touched were re-measured, not carried over.

## What I could not verify

<!-- Say it plainly. "I have no macOS machine; CI covers it" is a fine answer
     and a far better one than silence. An honest gap is a result; a fabricated
     green is a conduct breach. -->
