# Security

## Scope

VideoColor is a local video-processing tool and a Colab workflow. It is not
designed to be an unauthenticated public upload service. FFmpeg, image
decoders, PyTorch, model files, and notebook execution are part of its trust
boundary. Use maintained dependencies and process untrusted media in an
isolated environment.

Do not load checkpoint files from people or sites you do not trust. Model
weights and their download locations are distinct from the application's MIT
license. Read [Third-party notices](THIRD_PARTY_NOTICES.md) before changing or
redistributing models.

Processing outputs, cached frames, reports, and error messages may reveal
video contents or local paths. Review them before sharing. A `.gitignore`
file does not remove files already committed to Git or protect an archive
made by zipping an entire working folder.

## Report a vulnerability

When this project is hosted on GitHub and private vulnerability reporting is
enabled, use the repository's **Security > Report a vulnerability** action.
Include the affected version, impact, and a minimal reproducible example.
Do not include private footage, credentials, or a working exploit in a
public issue. If private reporting is unavailable, open an issue asking for
a private contact route without disclosing exploitation details.

This project has no guaranteed response time or long-term security support
commitment. Security fixes target the latest release.
