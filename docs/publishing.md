# Publish a clean source release

The public release should contain the application, tests, documentation,
licenses, and a notebook with no execution outputs. It should not contain
source videos, generated frames, output movies, model weights, local
environments, credentials, private notebook links, or personal account data.

## Build and check

From the project root, using your development environment:

```console
python -m pytest
python scripts/build_notebook.py
python scripts/check_release.py
python scripts/package_release.py
```

By default, the archive is created as `VideoColor-v0.2.0-source.zip` in the
parent directory. An existing archive is not overwritten; use `--output`
with a new destination to produce another build.

Inspect the script's reported archive path and file list. The packaging
script selects release source files rather than copying the entire working
directory. `.gitignore` is an additional safeguard; it is not a privacy audit
and does not affect files already committed to Git.

Rebuild the notebook after changes to application code, package metadata,
or embedded documentation. The release checker rejects a notebook containing
an older source bundle.

Extract the generated source ZIP into a new folder and test it there:

```console
python -m venv .venv
```

Use that environment's Python to install with `python -m pip install -e .`,
run `python -m videocolor doctor`, and make a short preview of a recording you
are entitled to process. Inspect the exported color, sound, duration, and
frame order. Validate the Colab notebook separately on the current runtime;
local CPU tests do not establish Colab GPU compatibility.

## Create the GitHub repository

Create an empty repository in your GitHub account. Upload the contents of
the clean, extracted source folder, or initialize Git there and connect it
to the new repository using GitHub's instructions. Do not initialize the
public repository around a directory containing private media or old
notebook history.

Before committing, inspect the staged filenames and content. Check notebook
outputs and metadata as well as ordinary text files. If you want to keep
your personal email private, configure the repository's Git author identity
with the no-reply address provided in your GitHub account settings before
making commits. Local commit metadata and the public GitHub profile are
separate privacy choices.

Enable private vulnerability reporting in the repository settings so
[SECURITY.md](../SECURITY.md) has a private reporting route. Run the included
CI workflow and investigate failures before tagging a release.

## Release description

Use a factual description such as:

> VideoColor adds a preview-first local and Colab workflow around DDColor for
> AI colorization of black-and-white video, with motion-based color smoothing,
> audio export, and resumable jobs.

State the devices and workflows actually tested. Do not describe synthetic
pipeline tests as visual AI validation, promise a processing speed you have
not measured, or advertise historical accuracy or guaranteed flicker removal.
Include [limitations](limitations.md) and the [disclaimer](../DISCLAIMER.md).

For a compelling example, use a short clip whose redistribution rights you
have checked. Publish the original and colorized interval together, with
the model/settings and an AI-colorized label. Obtain permission before
sharing personal recordings. No private source footage is supplied as a
public demonstration in this release.

## License and distribution boundary

Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, and every retained upstream license
and notice in source releases. DDColor weights and FFmpeg are separate
downloads. If you later package those files into a desktop installer,
container, or hosted service, review that exact distribution's licenses and
requirements. The source release's MIT license alone does not cover them.
