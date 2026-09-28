# Security Policy

Applies to every repository in the Nova-Violet Role organisation unless that
repository ships its own `SECURITY.md`, which then takes precedence.

---

## Reporting a vulnerability

**Do not open a public issue.**

Use GitHub's private vulnerability reporting on the affected repository:

> Repository → **Security** → **Advisories** → **Report a vulnerability**

That channel is private between you and the maintainer, gives us a place to work
on a fix before it is public, and lets us credit you in the advisory.

If the repository has advisories disabled and you believe you have found
something real, contact [@Saimonokuma](https://github.com/Saimonokuma) on GitHub
and say only that you have a security report — no details in the open.

## What we will do

| | |
|:--|:--|
| **Acknowledge** | Within 7 days. If you have not heard back, assume it was missed and ping again — that is not rude. |
| **Assess** | We reproduce it, or tell you exactly where we failed to. |
| **Fix** | With a regression test that we have **watched fail** before it passed. |
| **Disclose** | Via a GitHub Security Advisory, crediting you unless you ask us not to. |

We will tell you honestly if we decide **not** to fix something, and why. A
disclosed limitation is preferable to a silent one.

## What counts as a vulnerability here

These projects are developer tooling. The interesting attack surface is not a
web server — it is the **trust boundary between a claim and its evidence**:

- **Gate evasion** — making a check report success while the property it checks
  is false.
- **Record or attestation forgery** — producing an artefact that a verifier
  accepts as evidence of work that never happened.
- **Fence escape** — content crossing a boundary that is documented as holding
  it, including untrusted text reaching a place where it is treated as
  instruction rather than data.
- **Injection through tool output** — data from a file, log, network response or
  another agent being executed, expanded, or obeyed.
- **Privilege or path escape** — a hook, checker or installer touching anything
  outside its declared footprint, or writing outside the paths it declares.
- **Credential exposure** — any path by which a token, key or secret could be
  written to a log, an artefact, a commit, or a published archive.
- **Supply chain** — a release archive whose contents do not match its tracked
  sources.

A proof-of-concept that makes a gate lie is the highest-value report we accept.

## What is not a vulnerability

- A documented limitation. Check the repository's review or limits document
  first; we publish what these tools do **not** claim.
- A finding that requires an attacker who already controls the machine the tool
  runs on. If they own the shell, they do not need our bug.
- Output from an automated scanner with no demonstrated impact. Show the path.

Reporting one of these in good faith is not a problem — you will get a real
answer explaining which category it fell into.

## Supported versions

The default branch, and the most recent tagged release of each repository.
Older tags are not patched.

## Safe harbour

We will not pursue or support action against anyone who, in good faith:

- researches vulnerabilities in these repositories,
- reports them privately through the channel above,
- gives us reasonable time to fix before public disclosure,
- and does not access, modify or exfiltrate anyone else's data in the process.

Testing against your own checkout is always fine. These are local developer
tools; there is no production service of ours to attack.

---

*Nova-Violet Role · Reality is the judge.*
