# Making Burmese / Myanmar Unicode Render & Type in a Terminal (Windows 11)

Research date: 2026-06-13. Every tool below was fetched and verified to exist with a working URL.

---

## (A) Verdict (one paragraph)

**No terminal on any OS renders Burmese 100% correctly today, and that includes Windows Terminal + PowerShell.** The blocker is architectural, not a font you forgot to install: terminals are a fixed character-cell grid ("one codepoint = one cell"), but Myanmar script needs glyph reordering, consonant stacking, and combining vowel marks that collapse several codepoints into one visual cluster of irregular width. Windows Terminal uses **DirectWrite**, which _does_ do OpenType complex shaping, so Burmese in Windows Terminal looks **much better than in most Linux terminals** (Padauk/Myanmar Text glyphs will shape, not show as boxes) — but the cell-width math (`wcwidth`) still can't measure a shaped cluster, so you get **cursor misalignment, overlapping marks, and selection/editing weirdness**, especially on longer Burmese lines. The realistic goal is: **kill the "boxes" (tofu) and get readable shaped Burmese in Windows Terminal for short strings, and read/edit any serious Burmese in VS Code's editor pane (not a terminal) where shaping is correct.** The "perfect terminal Burmese" fix does not exist yet; the Unicode **Terminal Complex Script Support Working Group** (2023–) is still standardizing it.

---

## (B) Ranked fixes (exact config)

### Fix 0 — Rule out Zawgyi first (the #1 cause of "garbled but not boxes")

If text shows recognizable Burmese letters but in the _wrong order / wrong shapes_ (not empty boxes), it's almost certainly **Zawgyi**, a non-standard legacy font encoding still common in Myanmar that reuses Unicode code points incorrectly. It is incompatible with real Unicode and no amount of font/terminal config fixes it — you must **convert the bytes to Unicode**. Use Google `myanmar-tools` (detect) + Rabbit/`myanmar-tools` (convert). See tool table.
Boxes/tofu = missing font (Fix 1). Garbled letters = encoding (Fix 0).

### Fix 1 — Install a real Myanmar Unicode font (kills the boxes)

Windows 11 ships **Myanmar Text** (`mmrtext.ttf`), which DirectWrite shapes correctly — so you may already have shaping. For best coverage install **Padauk (SIL)** and/or **Pyidaungsu**:

- Padauk: https://software.sil.org/padauk/download/ (also `winget install` is not available; download the `.ttf`, right-click → Install for all users)
- Pyidaungsu (official, Myanmar Computer Federation): https://mcf.org.mm/pyidaungsu-font
- Noto Sans Myanmar: https://fonts.google.com/noto/specimen/Noto+Sans+Myanmar

**Caveat:** none of these are true fixed-width _monospace_ fonts. There is **no good monospace Myanmar terminal font** — Myanmar is inherently proportional. You therefore run a Latin monospace as primary and let Myanmar fall back.

### Fix 2 — Windows Terminal `settings.json` font + fallback

Windows Terminal **does** support a comma-separated `font.face` fallback list (the old single-font limitation is gone in current builds). Put a monospace first for code, Myanmar second so Burmese code points fall back to it:

```jsonc
// settings.json  ->  profiles > defaults
{
  "profiles": {
    "defaults": {
      "font": {
        "face": "Cascadia Mono, Padauk, Myanmar Text, Noto Sans Myanmar",
        "size": 12,
      },
    },
  },
}
```

If your build rejects the comma list, fall back to a single face that has Myanmar glyphs:

```jsonc
"font": { "face": "Myanmar Text", "size": 12 }
```

(You lose nice monospace Latin, but Burmese will shape.)

### Fix 3 — Force UTF-8 everywhere in PowerShell (so bytes aren't mangled on the way in/out)

Add to your PowerShell profile (`notepad $PROFILE`):

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8
chcp 65001 > $null
```

Also enable Windows' "Beta: Use Unicode UTF-8 for worldwide language support" (Settings → Time & language → Language & region → Administrative language settings → Change system locale → tick the UTF-8 box → reboot). This stops the legacy code page from corrupting Burmese in non-UTF-8 tools.

### Fix 4 — Typing Burmese into the terminal

Install a Unicode keyboard so input is real Unicode, not Zawgyi: Windows 11 has a built-in **Myanmar (Visual order / Burmese) keyboard** (Settings → Language → Myanmar → add keyboard), or use **Keymagic** (open source, Burmese community). Note: even with correct input, the _cursor position_ while typing Burmese in the terminal will drift because of the width problem — short inserts are fine, long-line editing is painful.

### Fix 5 — When the terminal still isn't good enough: read/edit in VS Code's editor

For any non-trivial Burmese (paragraphs, your WikiHow-MY corpus rows, model outputs), **don't fight the terminal** — open the file in the **VS Code editor pane**, which uses the OS shaping engine correctly. The VS Code _integrated terminal_ shares the terminal-grid problem; the _editor_ does not. Practical pattern: have Claude Code write Burmese output to a `.md`/`.txt` and open it, rather than reading it inline in the terminal.

---

## (C) Verified open-source tools (Burmese community + relevant)

| Name                                         | What it does                                                                                            | URL                                                                                       | Language / binding                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Google **myanmar-tools**                     | ML-based **Zawgyi vs Unicode detection** + conversion; CLDR-based rules                                 | https://github.com/google/myanmar-tools                                                   | C++, Java, JS (Node+browser), PHP, Ruby, Dart, C#; Python via PyPI |
| myanmar-tools (Python)                       | Python detector wrapper                                                                                 | https://pypi.org/project/myanmartools/                                                    | Python                                                             |
| **Rabbit Converter** (org)                   | Zawgyi ⇄ Unicode converter, JSON rule set, many language ports                                          | https://github.com/Rabbit-Converter                                                       | JS, PHP, Node, Java, Python, Swift                                 |
| Rabbit (core)                                | Reference Zawgyi⇄Unicode implementation                                                                 | https://github.com/Rabbit-Converter/Rabbit                                                | JS                                                                 |
| Rabbit web tool                              | Paste-and-convert site                                                                                  | http://www.rabbit-converter.org/                                                          | Web                                                                |
| **Padauk** (SIL)                             | Comprehensive Myanmar Unicode font, Unicode 16 ranges, all OSes                                         | https://software.sil.org/padauk/                                                          | Font (OFL)                                                         |
| **Pyidaungsu** (Myanmar Computer Federation) | Government-standard Myanmar Unicode font                                                                | https://mcf.org.mm/pyidaungsu-font                                                        | Font (OFL)                                                         |
| **Noto Sans Myanmar** (Google)               | Pan-Unicode Myanmar font                                                                                | https://fonts.google.com/noto/specimen/Noto+Sans+Myanmar                                  | Font (OFL)                                                         |
| **awesome-myanmar-unicode**                  | Curated hub of MM fonts/converters/keyboards/NLP tools                                                  | https://github.com/khzaw/awesome-myanmar-unicode                                          | List                                                               |
| Myanmar-Unicode-Fonts                        | Packaged collection of MM Unicode fonts (incl. Pyidaungsu .ttf)                                         | https://github.com/AungMyoKyaw/Myanmar-Unicode-Fonts                                      | Fonts                                                              |
| **Parabaik**                                 | Zawgyi ⇄ Unicode text converter                                                                         | (listed in awesome-myanmar-unicode) https://github.com/khzaw/awesome-myanmar-unicode      | Lib                                                                |
| **osc66**                                    | Rust CLI: pipes text through HarfBuzz, emits OSC 66 so a _capable_ terminal sizes shaped clusters right | https://thottingal.in/blog/2026/03/22/complex-scripts-in-terminal/ (tool described there) | Rust CLI                                                           |

> Width-fixing tools like **osc66 / OSC 66** only help on terminals that implement OSC 66 — currently **kitty** and **foot**, **not** Windows Terminal/PowerShell.

---

## (D) Honest limitations

- **No terminal fully shapes Myanmar on a cell grid.** Confirmed by the Unicode complex-scripts blog (Thottingal) and the SIL Padauk issue #53, where a user reports Myanmar renders wrong in GNOME Terminal, kitty, Alacritty, and foot alike. Windows Terminal is _better_ (DirectWrite shapes the glyphs) but still suffers cursor/width misalignment.
- **Windows Terminal advantage is real but partial:** DirectWrite (successor to Uniscribe, Myanmar shaping since Windows 8, script tag `mym2`) will render Padauk/Myanmar Text shaped — you'll see correct glyphs, not boxes. The _positioning math_ is what stays broken.
- **No true monospace Myanmar font exists** — the script is proportional; you always run Latin-mono + Myanmar fallback.
- **Zawgyi can't be fixed with fonts** — it's a byte-level encoding problem; you must convert (myanmar-tools / Rabbit). Detect first, because the symptom (garbled letters) looks like a rendering bug but isn't.
- **VS Code:** editor pane shapes Myanmar correctly; its _integrated terminal_ does not (same grid problem). The reported VS Code issue (#92700) was an input/encoding case, closed info-needed.
- **Best practical posture for Claude Code:** treat the terminal as roughly readable for short Burmese strings, and route real Burmese content to a file opened in the VS Code editor.

---

## Sources

- Complex scripts in terminals / OSC 66 / TCSS WG: https://thottingal.in/blog/2026/03/22/complex-scripts-in-terminal/
- Burmese multilingual support (works out-of-box Win 8+, complex text must be enabled): https://en.wikipedia.org/wiki/Help:Multilingual_support_(Burmese)
- Microsoft — Myanmar OpenType / shaping (`mym2`, not `mymr`): https://learn.microsoft.com/en-us/typography/script-development/myanmar
- Uniscribe/DirectWrite shaping engines incl. Myanmar (Win 8): https://handwiki.org/wiki/Uniscribe
- Universal Shaping Engine: https://learn.microsoft.com/en-us/globalization/reference/universal-shaping-engine
- Windows Terminal appearance/font settings: https://learn.microsoft.com/en-us/windows/terminal/customize-settings/profile-appearance
- WT multiple fontFace request (history): https://github.com/microsoft/terminal/issues/5634
- Padauk in Linux terminals renders wrong (issue #53): https://github.com/silnrsi/font-padauk/issues/53
- VS Code Myanmar font/encoding issue: https://github.com/microsoft/vscode/issues/92700
- google/myanmar-tools: https://github.com/google/myanmar-tools | PyPI: https://pypi.org/project/myanmartools/
- Rabbit-Converter: https://github.com/Rabbit-Converter | http://www.rabbit-converter.org/
- awesome-myanmar-unicode: https://github.com/khzaw/awesome-myanmar-unicode
- Padauk (SIL): https://software.sil.org/padauk/download/
- Pyidaungsu (MCF): https://mcf.org.mm/pyidaungsu-font
- Noto Sans Myanmar: https://fonts.google.com/noto/specimen/Noto+Sans+Myanmar
- Myanmar-Unicode-Fonts: https://github.com/AungMyoKyaw/Myanmar-Unicode-Fonts
