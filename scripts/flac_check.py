#!/usr/bin/env python3
# Author: Jakub Cabal <jakubcabal@gmail.com>
# SPDX-License-Identifier: MIT

"""
flac_check.py - find weakly compressed, damaged, non-subset or dishonest FLACs.

Workflow: `analyze` scans a folder and stores its findings in a report file.
`reencode` and `repair` then act on that report, so the slow decoding pass
happens exactly once.

    flac_check.py analyze   FOLDER     full analysis -> report
    flac_check.py show      FOLDER     print the stored report again
    flac_check.py find-fake FOLDER     separate check: does the file lie?
    flac_check.py reencode  FOLDER     rewrite weakly compressed files
    flac_check.py repair    FOLDER     fix subset, headers, damaged audio

WHAT CAN BE DETECTED, AND HOW RELIABLY
The compression level (-0..-8) is not stored in a FLAC file, it can only be
inferred from traces the encoder left behind. Best clues first:

  block size    A hard encoder parameter, independent of the material,
                readable from the header. See BLOCKSIZE_TABLE.
  no LPC        Levels -0..-2 run with -l 0, so they emit no LPC subframe
                at all. A certain sign of a low level.
  stereo mode   Always-independent channels mean -0 or -3.
  LPC order     UNRELIABLE, the encoder picks it from the material. Shown
                for information only, never flags a file on its own.

Levels -0/-1/-2 are detected reliably. Telling -5 from -8 after the fact is
not possible, and the difference is about half a percent anyway.

DOES THE FILE TELL THE TRUTH (`find-fake`)
A FLAC is lossless with respect to what it was given, which says nothing about
what that was. Three lies are detectable, and a file can tell any combination
of them:

  lossy source    A codec's brick-wall filter leaves a cliff in the spectrum
                  that natural material never has. Catches MP3 at any bitrate
                  and AAC up to about 128 kbps; higher AAC is transparent here.
  fake hi-res     96 kHz upsampled from 44.1 or 48 kHz has arithmetic zero
                  above the original Nyquist, where a genuine capture always
                  has at least noise.
  padded depth    16 bit in a 24 bit container: the low byte of every sample
                  is zero. Not statistics - a proof.

None of this is repairable, which is why it is a command of its own and not
part of the analyze/reencode/repair chain. See the FakeCheck section for the
thresholds and the measurements they came from.

FLAC SUBSET
The subset is what hardware players restrict themselves to. A file outside it
is still valid FLAC, but a set-top box or car radio may refuse it. The limits
are looser above 48 kHz, see the SUBSET_ constants. What counts are the ACTUAL
values in the stream, not the encoder settings.

Two kinds of violation, only one of which re-encoding can fix:
  block size, LPC order, partition order   encoder parameters -> `repair`
                                           rewrites the file losslessly.
  bit depth, sample rate                   properties of the audio itself.
                                           Re-encoding cannot change them;
                                           only resampling or dithering could,
                                           and that would be lossy. Reported,
                                           never touched. 24 bit and 96 kHz
                                           are inside the subset, so ordinary
                                           hi-res material is never affected.

OVERWRITE SAFETY
Encoding goes to a temporary file in the same directory. The original is
replaced atomically (os.replace) only after the new file passes both an MD5
check of the decoded audio and `flac -t`. On any problem the temporary file
is removed and the original is left untouched.

`repair` handles three defects. Returning a file into the subset and
correcting a lying header are both verifiable, so the original is simply
replaced. Salvaging damaged audio is not - part of it is already gone - so
there the original is kept as <name>.orig.flac and tags and cover art are
carried over into the replacement.

Whether to rewrite a file is decided PER FILE, never from a folder average,
see meets_threshold().

Requires Python 3.8+ and the `flac` tool 1.3+ in PATH. No PyPI packages.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from dataclasses import dataclass, field, fields, is_dataclass
from typing import BinaryIO, Callable, Sequence

MIN_PYTHON = (3, 8)
MIN_FLAC = (1, 3, 0)
REPORT_VERSION = 2

# --------------------------------------------------------------------------
# Languages
#
# Both variants sit side by side so they cannot drift apart.
# --------------------------------------------------------------------------

LANGUAGES = ("cs", "en")
DEFAULT_LANGUAGE = "cs"
_lang = DEFAULT_LANGUAGE

#: key: (Czech, English). Named {placeholders} must match in both.
def _messages(table: str) -> dict:
    """key | Czech | English, one message per line.

    Both languages stay side by side so they cannot drift apart, without the
    quoting that costs two physical lines per message. Whitespace around the
    columns is padding, `\\n` is a line break, and the placeholders of the two
    languages have to match - a typo in either is caught here, at import.
    """
    out = {}
    for line in table.strip().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, cs, en = (part.strip().replace("\\n", "\n")
                       for part in line.split("|"))
        if set(re.findall(r"{(\w+)", cs)) != set(re.findall(r"{(\w+)", en)):
            raise ValueError(f"MESSAGES[{key}]: placeholders differ")
        out[key] = (cs, en)
    return out


MESSAGES = _messages(r"""
# --- dependencies -----------------------------------------------------
dep_python_old   | Python {have} je starý, potřeba {want}+ | Python {have} is too old, need {want}+
dep_flac_missing | nástroj 'flac' není v PATH              | the 'flac' tool is not in PATH
dep_flac_old     | flac {have} je starý, potřeba {want}+   | flac {have} is too old, need {want}+

# --- read and encode errors -------------------------------------------
err_eof              | neočekávaný konec souboru v metadatech   | unexpected end of file in the metadata
err_not_flac         | není to FLAC (chybí 'fLaC')              | not a FLAC file (no 'fLaC' marker)
err_first_block      | první blok není STREAMINFO               | the first block is not STREAMINFO
err_short_streaminfo | zkrácený STREAMINFO                      | truncated STREAMINFO
err_no_streaminfo    | chybí STREAMINFO                         | STREAMINFO is missing
err_no_output        | bez výpisu                               | no output
err_analyze_failed   | flac -a skončil s kódem {code}: {detail} | flac -a exited with {code}: {detail}
err_no_frames        | flac -a nevrátil žádné rámce             | flac -a returned no frames
err_encode_failed    | flac skončil s kódem {code}: {detail}    | flac exited with {code}: {detail}
err_md5_decode       | dekódování pro MD5 selhalo               | decoding for the MD5 failed

# --- losslessness check -----------------------------------------------
err_mismatch    | nesouhlasí {field}: {old} -> {new}            | {field} does not match: {old} -> {new}
err_md5_unset   | nový soubor nemá MD5, nelze ověřit            | the new file carries no MD5, cannot verify
err_md5_differs | MD5 se liší - překódování NEBYLO bezeztrátové | MD5 differs - the re-encode was NOT lossless
err_flac_t      | flac -t neprošel: {detail}                    | flac -t failed: {detail}

# --- severity and labels ----------------------------------------------
sev_ok   | v pořádku                    | fine
sev_warn | Slabší komprese              | Weaker compression
sev_bad  | Velmi slabá komprese (-0/-2) | Very weak compression (-0/-2)

# --- why a file was flagged -------------------------------------------
why_blocksize_low | blok {size} při {rate} Hz (nízké úrovně mají {expected})   | block size {size} at {rate} Hz (low levels use {expected})
why_blocksize_odd | nestandardní blok {size} při {rate} Hz                     | unusual block size {size} at {rate} Hz
why_no_lpc        | žádná LPC predikce (enkodér běžel s -l 0)                  | no LPC prediction at all (encoder ran with -l 0)
why_low_lpc       | jen {pct:.0f} % subrámců používá LPC                       | only {pct:.0f} % of subframes use LPC
why_low_order     | nejvyšší LPC řád jen {order} (odpovídá -3, ale závisí na obsahu) | highest LPC order is only {order} (suggests -3, but depends on the material)
why_no_stereo     | žádná stereo dekorelace (-0 nebo -3)                       | no stereo decorrelation (-0 or -3)

# --- FLAC subset ------------------------------------------------------
sub_blocksize | blok {size} > {limit} (limit pro {rate} Hz)                | block size {size} > {limit} (limit at {rate} Hz)
sub_bps       | bitová hloubka {bps} (subset dovoluje {allowed})           | bit depth {bps} (the subset allows {allowed})
sub_rate      | frekvence {rate} Hz nejde zapsat do hlavičky rámce         | sample rate {rate} Hz cannot be stored in the frame header
sub_lpc       | LPC řád {order} > {limit} (limit pro {rate} Hz)            | LPC order {order} > {limit} (limit at {rate} Hz)
sub_partition | Rice partition order {order} > {limit}                     | Rice partition order {order} > {limit}
sub_header    | MIMO FLAC SUBSET - hardwarový přehrávač může odmítnout:    | OUTSIDE THE FLAC SUBSET - a hardware player may refuse it:
sub_fixable   | (opraví repair, bezeztrátově)                              | (repair fixes this, losslessly)
sub_inherent  | (nelze opravit beze ztráty - vlastnost zvuku, ne enkodéru) | (cannot be fixed losslessly - a property of the audio, not of the encoder)

# --- per-file report --------------------------------------------------
rep_block       | blok                                                 | block
rep_subframes   | subrámce: {types}                                    | subframes: {types}
rep_lpc         | LPC řád: ⌀ {avg:.2f}, max {max}                      | LPC order: mean {avg:.2f}, max {max}
rep_stereo      | stereo: {mode}                                       | stereo: {mode}
rep_compression | komprese {pct:.1f} %                                 | compression {pct:.1f} %
rep_encoder     | enkodér {vendor}                                     | encoder {vendor}
rep_partial     | (po přepsání, bez hloubkové analýzy - spusť analyze) | (rewritten, no deep analysis yet - run analyze)

# --- damaged files ----------------------------------------------------
dmg_header           | 💥 POŠKOZENÉ SOUBORY ({count}) - dekodér na nich selhal:    | 💥 DAMAGED FILES ({count}) - the decoder failed on them:
dmg_locating         | Zjišťuji rozsah poškození ({count})                        | Locating the damage ({count})
dmg_at_start         | {time} = {pct:.1f} % stopy, na začátku                     | {time} = {pct:.1f} % in, near the start
dmg_at_end           | {time} = {pct:.1f} % stopy, na konci                       | {time} = {pct:.1f} % in, near the end
dmg_at_middle        | {time} = {pct:.1f} % stopy, uprostřed                      | {time} = {pct:.1f} % in, in the middle
dmg_error_at         | ✗ chyba v datech na {where}                                | ✗ data error at {where}
dmg_truncated        | ✂ useknuto na {where} - chybí {samples} vzorků ({seconds:.2f} s) | ✂ truncated at {where} - {samples} samples missing ({seconds:.2f} s)
dmg_encoder_bug      | → enkodér: {vendor}                                        | → encoder: {vendor}
dmg_harmless_header  | 🔧 VADNÁ HLAVIČKA ({count}) - enkodér zahodil poslední neúplný rámec,\n   ale do hlavičky zapsal plný počet vzorků. Zvuk je celý, chybu\n   hlásí jen `flac -t`. | 🔧 BAD HEADER ({count}) - the encoder dropped the last partial frame\n   but wrote the full sample count into the header. No audio is\n   missing, only `flac -t` complains.
dmg_unknown_position | (přesnou pozici se nepodařilo určit)                       | (the exact position could not be determined)
unreadable_header    | Nepodařilo se přečíst:                                     | Could not be read:

# --- repair -----------------------------------------------------------
fix_nothing       | Report nehlásí žádný poškozený soubor.                     | The report lists no damaged files.
fix_confirm       | Chystám se opravit {count}:                                | About to repair {count}:
fix_intro_subset  | {count} mimo subset - překóduji na místě, ověřím proti MD5 | {count} outside the subset - re-encoded in place, MD5 verified
fix_intro_header  | {count} s vadnou hlavičkou - opravím jen ji, zvuk zůstane  | {count} with a bad header - only it is corrected, audio stays
fix_intro_salvage | {count} s poškozeným zvukem - zachráním, co jde; originál → *{suffix} | {count} damaged - whatever is readable is salvaged; original → *{suffix}
fix_header_done   | hlavička opravena, {old} → {new} vzorků, zvuk nezměněn     | header corrected, {old} → {new} samples, audio untouched
fix_running       | Zachraňuji ({count})                                       | Salvaging ({count})
fix_recovered     | zachráněno {pct:.2f} %{lost}                               | recovered {pct:.2f} %{lost}
fix_lost          | , ztraceno {seconds:.2f} s                                 | , {seconds:.2f} s lost
fix_no_loss       | , beze ztráty                                              | , nothing lost
fix_backed_up     | originál → {name}                                          | original → {name}
fix_dry           | nanečisto, nic se nezapsalo                                | dry run, nothing written
fix_failed        | záchrana selhala: {error}                                  | salvage failed: {error}
fix_exists        | {name} už existuje - originál by se přepsal, přeskakuji    | {name} already exists - the original would be lost, skipping
fix_no_meta       | tagy se přenést nepodařilo                                 | tags could not be carried over

# --- reencode ---------------------------------------------------------
rc_nothing      | Report nehlásí nic k překódování.                          | The report lists nothing to re-encode.
rc_confirm      | Chystám se PŘEPSAT {count} na místě ({total}).             | About to OVERWRITE {count} in place ({total}).
rc_settings     | nastavení: flac {opts}                                     | settings: flac {opts}
rc_promise      | Zvuk zůstane bit po bitu stejný (ověřuje se MD5), tagy a obal také. | The audio stays bit-for-bit identical (MD5 checked), so do tags and cover art.
rc_not_tty      | Vstup není terminál - pro neinteraktivní běh použij --yes. | Input is not a terminal - use --yes for non-interactive runs.
rc_prompt       | Pokračovat? [ano/Ne]:                                      | Continue? [yes/No]:
rc_cancelled    | Zrušeno, nic se nezměnilo.                                 | Cancelled, nothing changed.
rc_running      | Překódovávám ({count})                                     | Re-encoding ({count})
rc_running_dry  | Zkouším nanečisto ({count})                                | Dry run over ({count})
rc_replaced     | {old} → {new} (ušetřeno {saved}, {pct:.1f} %)              | {old} → {new} (saved {saved}, {pct:.1f} %)
rc_would        | ušetřilo by se {saved} ({pct:.1f} %), soubor nezměněn      | would save {saved} ({pct:.1f} %), file unchanged
rc_would_subset | vrátil by se do subsetu ({change}), soubor nezměněn        | would return into the subset ({change}), file unchanged
rc_skipped      | úspora pod {min} % ({saved}), originál ponechán            | saving below {min} % ({saved}), original kept
rc_subset_done  | v subsetu, {old} → {new} ({change})                        | now in subset, {old} → {new} ({change})
rc_grew         | naroste o {pct:.2f} %                                      | grows by {pct:.2f} %
rc_saved_pct    | ušetřeno {pct:.2f} %                                       | saved {pct:.2f} %
rc_failed       | {error} - originál ponechán                                | {error} - original kept

# --- report file ------------------------------------------------------
rp_saved       | Report uložen: {path}                                      | Report saved: {path}
rp_missing     | Report pro '{root}' neexistuje - spusť nejdřív:\n  {cmd}   | No report for '{root}' - run this first:\n  {cmd}
rp_broken      | Report {path} nejde přečíst ({error}), spusť analyze znovu. | Report {path} cannot be read ({error}), run analyze again.
rp_from        | Report z {when} ({count})                                  | Report from {when} ({count})
rp_stale_file  | {name}: od analýzy se změnil, přeskakuji                   | {name}: changed since the analysis, skipping
rp_gone_file   | {name}: už neexistuje, přeskakuji                          | {name}: no longer exists, skipping
rp_stale_count | Přeskočeno {count} (změněno od analýzy) - spusť analyze znovu. | Skipped {count} (changed since the analysis) - run analyze again.
rp_reused      | beze změny od minule: {count}, znovu analyzuji {fresh}     | unchanged since last time: {count}, re-analysing {fresh}

# --- progress and run -------------------------------------------------
run_scanning    | Prohledávám {root}          | Scanning {root}
run_none        | Žádné .flac soubory.        | No .flac files found.
run_meta        | Čtu hlavičky ({count})      | Reading headers ({count})
run_deep        | Hloubková analýza ({count}) | Deep analysis ({count})
run_bad_path    | cesta '{root}' není složka  | '{root}' is not a directory
run_error       | Chyba: {problem}            | Error: {problem}
run_interrupted | Přerušeno.                  | Interrupted.
prog_eta        | zbývá {eta}                 | {eta} left
prog_done       | hotovo za {elapsed}         | done in {elapsed}

# --- summary ----------------------------------------------------------
sum_total       | FLAC souborů celkem           | FLAC files in total
sum_damaged     | POŠKOZENÉ                     | DAMAGED
sum_harmless    | Vadná hlavička (zvuk je celý) | Bad header (audio is complete)
sum_unreadable  | Nečitelná hlavička            | Unreadable header
sum_subset_fix  | Mimo subset (opravitelné)     | Outside subset (fixable)
sum_subset_keep | Mimo subset (neopravitelné)   | Outside subset (not fixable)
sum_fake        | Nesedí deklarovaná kvalita    | Quality is not what it claims
sum_grew        | Narostlo o                    | Grew by
sum_done        | Překódováno                   | Re-encoded
sum_saved       | Ušetřeno                      | Saved
sum_skipped     | Přeskočeno (malá úspora)      | Skipped (saving too small)
sum_failed      | Selhalo (originál zachován)   | Failed (original kept)
sum_repaired    | Zachráněno                    | Salvaged
sum_subset_done | Vráceno do subsetu            | Returned into the subset
sum_header_done | Opravena hlavička             | Header corrected

# --- next step --------------------------------------------------------
adv_next        | Další krok:                                                | Next step:
adv_reencode    | {cmd}  ({count}, ušetří místo, {eta})                      | {cmd}  ({count}, saves space, {eta})
adv_repair      | {cmd}  ({count} s vadou)                                   | {cmd}  ({count} with a defect)
adv_findfake    | {cmd}  (jestli soubory nelžou o kvalitě, {eta})            | {cmd}  (whether the files lie about their quality, {eta})
adv_clean       | Vše v pořádku, nic k opravě.                               | All clear, nothing to fix.
adv_fake        | {count} má horší kvalitu, než tvrdí; překódování nepomůže, jen lepší kopie ze zdroje. | {count} are worse than they claim; re-encoding will not help, only a better copy from the source.
adv_was_dry     | Bylo to nanečisto. Spusť totéž bez --dry-run.              | That was a dry run. Repeat it without --dry-run.
adv_failed      | U selhaných zůstaly originály; zkontroluj práva a volné místo. | Originals of the failed files are untouched; check permissions and free space.
adv_subset_keep | {count} mimo subset nelze opravit beze ztráty, viz výš.    | {count} outside the subset cannot be fixed losslessly, see above.
adv_reanalyze   | Report je aktualizovaný; pro plný obraz spusť analyze znovu. | The report is updated; run analyze again for the full picture.

# --- files that lie about their quality --------------------------------
fake_unusable  | stopa je příliš krátká nebo tichá                          | the track is too short or too quiet
fake_running   | Ověřuji deklarovanou kvalitu ({count})                     | Verifying the declared quality ({count})
fake_header    | 🎭 KVALITA NEODPOVÍDÁ DEKLARACI ({count}):                  | 🎭 QUALITY IS NOT WHAT IS CLAIMED ({count}):
fake_lossy     | ztrátový zdroj: ostrý ořez na {khz:.1f} kHz, sráz {db:.0f} dB - odpovídá {hint} | lossy source: sharp cutoff at {khz:.1f} kHz, {db:.0f} dB cliff - consistent with {hint}
fake_upsampled | falešné hi-res: {claimed} kHz převzorkováno z {source} kHz, nad {edge:.1f} kHz je ticho ({db:.0f} dB pod hudbou) | fake hi-res: {claimed} kHz upsampled from {source} kHz, silence above {edge:.1f} kHz ({db:.0f} dB below the music)
fake_padded    | falešná hloubka: hlavička říká {claimed} bitů, vzorky využívají {real} | padded depth: the header says {claimed} bits, the samples use {real}
fake_caveat    | Ořez může mít i nahrávka z analogového pásu; ověř spektrogramem. AAC nad ~192 kbps se takhle chytit nedá. | An analogue tape source can be cut off too; check a spectrogram. AAC above ~192 kbps cannot be caught this way.
fake_none      | Žádný soubor nelže o své kvalitě.                          | No file lies about its quality.
fake_skipped   | Nešlo změřit ({count}): {reason}                           | Could not be measured ({count}): {reason}

# --- command line help ------------------------------------------------
cli_description  | Kontrola FLAC knihovny: komprese, subset, poškození.       | Checks a FLAC library: compression, subset, damage.
cli_epilog       | Nejdřív analyze, pak podle nálezu reencode nebo repair.    | Run analyze first, then reencode or repair as needed.
cli_command      | příkaz                                                     | command
cli_folder       | složka s hudbou (rekurzivně)                               | music folder (recursive)
cli_jobs         | paralelních procesů (výchozí: počet jader)                 | parallel jobs (default: CPU count)
cli_lang         | jazyk výstupu (výchozí: podle prostředí)                   | output language (default: from the environment)
cli_report       | kam uložit report (výchozí: {path})                        | where to keep the report (default: {path})
cli_all          | vypsat i soubory, které jsou v pořádku                     | list the files that are fine as well
cli_effort       | standard = -8 (výchozí), exhaustive = -8 -e -p (16x pomalejší, +0,05 %%) | standard = -8 (default), exhaustive = -8 -e -p (16x slower, +0.05 %%)
cli_dry_run      | nic nezapisovat, jen spočítat výsledek                     | write nothing, only compute the outcome
cli_yes          | neptat se na potvrzení                                     | do not ask for confirmation
cli_min_saving   | nejmenší úspora v %% na soubor (výchozí: 1.0)              | smallest saving in %% per file (default: 1.0)
cli_force        | analyzovat vše znovu, i beze změny od minule               | re-analyse everything, even what has not changed
cli_cmd_analyze  | projít složku a uložit report (dekóduje, pomalé)           | scan the folder and store a report (decodes, slow)
cli_cmd_show     | znovu vypsat uložený report                                | print the stored report again
cli_cmd_findfake | ověřit kvalitu: zdroj z MP3/AAC, falešné hi-res i hloubka (pomalé) | verify the quality: MP3/AAC source, fake hi-res or depth (slow)
cli_cmd_reencode | překódovat slabě komprimované na místě (bezeztrátově, tagy zůstanou) | re-encode weakly compressed files in place (lossless, tags kept)
cli_cmd_repair   | opravit vady: subset, hlavička, poškozený zvuk (originál → *{suffix}) | fix defects: subset, header, damaged audio (original → *{suffix})
""")


def set_language(name: str) -> None:
    global _lang
    _lang = name if name in LANGUAGES else DEFAULT_LANGUAGE


def detect_language() -> str:
    """Language from the environment; Czech for Slovak locales too."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        if value := os.environ.get(var, ""):
            return "cs" if value[:2].lower() in ("cs", "sk") else "en"
    return DEFAULT_LANGUAGE


def t(key: str, **kw) -> str:
    """Translate a key and fill in its {placeholders}."""
    text = MESSAGES[key][LANGUAGES.index(_lang)]
    return text.format(**kw) if kw else text


def tr(reason: Sequence) -> str:
    """Render a stored (key, params) reason. Reports keep those, not strings,
    so a report written in Czech still prints correctly in English."""
    key, params = reason
    return t(key, **params)


# Block sizes per sample rate: (upper Hz bound, weak, normal).
# Reference flac keeps 1152 for -0..-2 and 4096 from -3 up regardless of rate,
# but ffmpeg scales its blocks for hi-res. Measured on flac 1.5.0 and Lavf62:
#     rate        flac -0..-2   flac -3+   ffmpeg c0   ffmpeg default
#     <= 48 kHz   1152          4096       1152        4608
#     <= 96 kHz   1152          4096       2304        8192
#     >  96 kHz   1152          4096       4608        16384
#
# The trap: at 192 kHz a block of 4608 means WEAK compression (ffmpeg c0),
# while at 44.1 kHz it is a perfectly normal ffmpeg file. One fixed threshold
# is therefore not enough. Measuring block length in ms does not work either:
# 4096 at 192 kHz is 21 ms, SHORTER than 1152 at 44.1 kHz (26 ms), yet fine.
BLOCKSIZE_TABLE = (
    (48000, frozenset({1152}), frozenset({4096, 4608})),
    (96000, frozenset({1152, 2304}), frozenset({4096, 8192})),
    (None, frozenset({1152, 4608}), frozenset({4096, 16384})),
)
MIN_SANE_BLOCKSIZE = 1152      # a shorter block is suspicious at any rate
MIN_LPC_SHARE = 0.5            # fewer LPC subframes than this is worth a warning

# FLAC subset rules, verified against flac 1.5.0 (what it refuses without --lax):
#     rate         max block   max LPC order   max Rice partition order
#     <= 48 kHz    4608        12              8
#     >  48 kHz    16384       unlimited       8
SUBSET_RATE_THRESHOLD = 48000
SUBSET_MAX_BLOCKSIZE = {False: 4608, True: 16384}   # key: is it hi-res?
SUBSET_MAX_LPC_ORDER = 12
SUBSET_MAX_PARTITION_ORDER = 8
# Bit depths and sample rates that fit directly in the frame header. Anything
# else needs a reference into STREAMINFO, which the subset forbids. Note that
# 24 bit and 96/192 kHz are all in here, so ordinary hi-res is inside the
# subset; only exotic depths (18 bit) or rates fall out, and neither can be
# changed without resampling or dithering, i.e. without losing audio.
SUBSET_BITS_PER_SAMPLE = frozenset({8, 12, 16, 20, 24, 32})
SUBSET_CODED_RATES = frozenset({8000, 16000, 22050, 24000, 32000, 44100,
                                48000, 88200, 96000, 176400, 192000})

# Encoder settings. Both stay inside the subset, so the result plays on
# hardware; deliberately no -l 32 / -r 15 and no --lax. Default is -8 because
# -e -p measurably gains 0.044 to 0.059 % for about 16x the time (20 MB track:
# ~10 kB for +13 s).
EFFORT_PRESETS = {"standard": ["-8"], "exhaustive": ["-8", "-e", "-p"]}
DEFAULT_EFFORT = "standard"

# Measured throughput per thread (MB/s), for rough run-time estimates only.
# On 12 cores: deep 832 MB in 2.6 s, encode 832 MB in 5.9 s, fake ~1.5 MB/s.
DEEP_RATE, ENCODE_RATE, FAKE_RATE = 25, 12, 2

#: The stereo search the encoder evidently used, named the way flac names it.
#: Technical tokens, so they need no translation.
STEREO_NONE, STEREO_ADAPTIVE, STEREO_EXHAUSTIVE = "INDEPENDENT", "-M", "-m"
STEREO_UNKNOWN = "?"

MD5_UNSET = b"\x00" * 16       # an MD5 the encoder never filled in
CHUNK = 1 << 16

#: Suffix for the untouched original kept aside by `repair`.
ORIG_SUFFIX = ".orig.flac"


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

def flac_version() -> tuple | None:
    """Version of the `flac` tool. None if it cannot run, empty tuple if it
    runs but the version could not be read (then nothing is held against it)."""
    try:
        out = subprocess.run(["flac", "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    found = re.search(rb"(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(g) for g in found.groups()) if found else ()


def _dotted(version: Sequence) -> str:
    return ".".join(map(str, version))


def check_dependencies(need_flac: bool) -> list:
    """Missing or too old dependencies."""
    problems = []
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append(t("dep_python_old", have=_dotted(sys.version_info[:3]),
                          want=_dotted(MIN_PYTHON)))
    if need_flac:
        version = flac_version()
        if version is None:
            problems.append(t("dep_flac_missing"))
        elif version and version < MIN_FLAC:
            problems.append(t("dep_flac_old", have=_dotted(version),
                              want=_dotted(MIN_FLAC)))
    return problems


# --------------------------------------------------------------------------
# Metadata reading - pure Python, a few kB per file
# --------------------------------------------------------------------------

#: Metadata block types (FLAC format).
BLOCK_STREAMINFO, BLOCK_PADDING, BLOCK_APPLICATION = 0, 1, 2
BLOCK_SEEKTABLE, BLOCK_VORBIS_COMMENT, BLOCK_CUESHEET, BLOCK_PICTURE = 3, 4, 5, 6


@dataclass
class FlacInfo:
    """What the header says about a file, plus analysis results."""

    path: str
    file_size: int = 0
    mtime_ns: int = 0
    audio_size: int = 0          # frames only, without metadata
    streaminfo_offset: int = 0   # where the STREAMINFO body starts in the file
    min_blocksize: int = 0
    max_blocksize: int = 0
    sample_rate: int = 0
    channels: int = 0
    bits_per_sample: int = 0
    total_samples: int = 0
    md5: bytes = b""             # MD5 of the decoded audio, the key to verifying
    vendor: str = ""             # vendor string, i.e. what encoded it
    deep: DeepSummary | None = None
    error: str | None = None
    damaged: bool = False        # the decoder failed on it, see `repair`

    @property
    def hi_rate(self) -> bool:
        return self.sample_rate > SUBSET_RATE_THRESHOLD

    @property
    def ratio(self) -> float | None:
        """Share of the uncompressed PCM size; None when it cannot be told.
        Bit depth need not be a multiple of eight, so divide last."""
        raw = self.total_samples * self.channels * self.bits_per_sample / 8
        if raw <= 0 or self.audio_size <= 0:
            return None
        return self.audio_size / raw


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(t("err_eof"))
    return data


def parse_flac_headers(f: BinaryIO, info: FlacInfo, blocks: list | None = None) -> int:
    """Read the metadata blocks of a stream into `info`.

    Returns the offset of the first audio frame. When `blocks` is given, every
    block is appended to it as (type, raw data). Raises ValueError on non-FLAC.
    """
    magic = _read_exact(f, 4)

    # Some files carry an ID3v2 tag glued in front of 'fLaC' (typically from
    # Windows taggers) - skip it using the syncsafe length.
    if magic[:3] == b"ID3":
        # The tag header is 10 bytes: "ID3", 2 version bytes, flags, 4 for the
        # length. Four of them are already in `magic`, so 2 are left before it.
        _read_exact(f, 2)                            # version minor and flags
        size = 0
        for b in _read_exact(f, 4):
            size = (size << 7) | (b & 0x7F)          # syncsafe: 7 bits per byte
        f.seek(size, os.SEEK_CUR)
        magic = _read_exact(f, 4)

    if magic != b"fLaC":
        raise ValueError(t("err_not_flac"))

    first = True
    while True:
        header = _read_exact(f, 4)
        is_last = bool(header[0] & 0x80)
        block_type = header[0] & 0x7F
        length = int.from_bytes(header[1:4], "big")

        if first and block_type != BLOCK_STREAMINFO:
            raise ValueError(t("err_first_block"))
        first = False

        if blocks is not None:
            blocks.append((block_type, _read_exact(f, length)))
            data = blocks[-1][1]
        else:
            data = None

        if block_type == BLOCK_STREAMINFO:
            if length < 34:
                raise ValueError(t("err_short_streaminfo"))
            info.streaminfo_offset = f.tell() - (length if blocks is not None else 0)
            d = data[:34] if data is not None else _read_exact(f, 34)
            if data is None:
                f.seek(length - 34, os.SEEK_CUR)
            info.min_blocksize = int.from_bytes(d[0:2], "big")
            info.max_blocksize = int.from_bytes(d[2:4], "big")
            # Bytes 10-17 hold 64 bits of packed fields:
            # sample_rate(20) channels(3) bits_per_sample(5) total_samples(36)
            packed = int.from_bytes(d[10:18], "big")
            info.sample_rate = packed >> 44
            info.channels = ((packed >> 41) & 0x07) + 1
            info.bits_per_sample = ((packed >> 36) & 0x1F) + 1
            info.total_samples = packed & ((1 << 36) - 1)
            info.md5 = d[18:34]
        elif block_type == BLOCK_VORBIS_COMMENT:
            d = data if data is not None else _read_exact(f, length)
            if len(d) >= 4:
                # Lengths are little-endian here, unlike the rest of the
                # format - inherited from Ogg Vorbis.
                vlen = int.from_bytes(d[0:4], "little")
                info.vendor = d[4:4 + vlen].decode("utf-8", "replace")
        elif data is None:
            f.seek(length, os.SEEK_CUR)

        if is_last:
            break

    if info.max_blocksize == 0:
        raise ValueError(t("err_no_streaminfo"))
    return f.tell()


def read_flac_metadata(path: str) -> FlacInfo:
    """Read one file's header. Touches the header only, never the audio."""
    stat = os.stat(path)
    info = FlacInfo(path=path, file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    with open(path, "rb") as f:
        info.audio_size = info.file_size - parse_flac_headers(f, info)
    return info


def read_metadata_blocks(path: str) -> list:
    """All metadata blocks of a file as (type, raw data)."""
    blocks = []
    with open(path, "rb") as f:
        parse_flac_headers(f, FlacInfo(path=path), blocks)
    return blocks


def _rebuild_vorbis_comment(data: bytes, vendor: str) -> bytes:
    """Same comments, new vendor string, so a rewritten file does not claim
    to have been encoded by whatever wrote the original."""
    if len(data) < 4:
        return data
    old_len = int.from_bytes(data[0:4], "little")
    rest = data[4 + old_len:]
    new = vendor.encode("utf-8")
    return len(new).to_bytes(4, "little") + new + rest


def graft_metadata(source_blocks: list, target: str) -> None:
    """Copy tags and cover art from `source_blocks` into the FLAC at `target`.

    STREAMINFO stays the target's own. SEEKTABLE and CUESHEET are dropped:
    both index the audio by sample offset, and the target may be shorter than
    the file the blocks came from. PADDING is dropped as noise.
    """
    keep = (BLOCK_APPLICATION, BLOCK_VORBIS_COMMENT, BLOCK_PICTURE)
    own = []
    info = FlacInfo(path=target)
    with open(target, "rb") as f:
        audio_offset = parse_flac_headers(f, info, own)

    out = [(BLOCK_STREAMINFO, dict(own)[BLOCK_STREAMINFO])]
    for block_type, data in source_blocks:
        if block_type not in keep:
            continue
        if block_type == BLOCK_VORBIS_COMMENT:
            data = _rebuild_vorbis_comment(data, info.vendor)
        out.append((block_type, data))

    header = bytearray(b"fLaC")
    for i, (block_type, data) in enumerate(out):
        header.append((0x80 if i == len(out) - 1 else 0) | block_type)
        header += len(data).to_bytes(3, "big")
        header += data

    tmp = target + ".meta.tmp"
    try:
        # The audio is copied in chunks: a whole album track at once would be
        # tens of MB per parallel job.
        with open(target, "rb") as src, open(tmp, "wb") as dst:
            dst.write(bytes(header))
            src.seek(audio_offset)
            shutil.copyfileobj(src, dst, CHUNK)
        os.replace(tmp, target)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Deep analysis via `flac -a`
# --------------------------------------------------------------------------

@dataclass
class DeepStats:
    """Statistics collected from the analysis output of `flac -a`."""

    frames: int = 0
    subframes: int = 0
    subframe_types: dict = field(default_factory=dict)
    lpc_orders: list = field(default_factory=list)
    partition_orders: list = field(default_factory=list)
    channel_assignments: dict = field(default_factory=dict)

    @property
    def max_lpc_order(self) -> int | None:
        return max(self.lpc_orders) if self.lpc_orders else None

    @property
    def avg_lpc_order(self) -> float | None:
        return sum(self.lpc_orders) / len(self.lpc_orders) if self.lpc_orders else None

    @property
    def max_partition_order(self) -> int | None:
        return max(self.partition_orders) if self.partition_orders else None

    @property
    def lpc_share(self) -> float:
        """Share of LPC among the subframes where prediction matters (0..1).
        CONSTANT subframes (digital silence, channels identical after mid/side)
        are optimal at any level, so counting them would only drag it down."""
        predictive = self.subframes - self.subframe_types.get("CONSTANT", 0)
        return self.subframe_types.get("LPC", 0) / predictive if predictive else 1.0

    @property
    def stereo_mode(self) -> str:
        """Which stereo search the encoder evidently used."""
        ca = self.channel_assignments
        if not ca:
            return STEREO_UNKNOWN
        if "LEFT_SIDE" in ca or "RIGHT_SIDE" in ca:
            return STEREO_EXHAUSTIVE       # -2, -5 and up
        if "MID_SIDE" in ca:
            return STEREO_ADAPTIVE         # -1, -4, or just few frames
        return STEREO_NONE                 # -0, -3, or mono

    def summary(self) -> DeepSummary:
        """The aggregates worth keeping - the per-subframe lists behind them
        run into millions and never reach the report."""
        return DeepSummary(self.frames, self.subframes, self.subframe_types,
                           self.max_lpc_order, self.avg_lpc_order,
                           self.max_partition_order, self.lpc_share,
                           self.stereo_mode)


@dataclass
class DeepSummary:
    """Deep statistics restored from a report; same reading interface."""

    frames: int = 0
    subframes: int = 0
    subframe_types: dict = field(default_factory=dict)
    max_lpc_order: int | None = None
    avg_lpc_order: float | None = None
    max_partition_order: int | None = None
    lpc_share: float = 1.0
    stereo_mode: str = STEREO_UNKNOWN

    @property
    def lpc_orders(self) -> bool:
        """Truthy when an LPC order is known, matching DeepStats usage."""
        return self.max_lpc_order is not None



# Output format of `flac -a`, tab separated:
#   frame=0     offset=8304  bits=19928  blocksize=4096  sample_rate=44100
#               channels=2   channel_assignment=INDEPENDENT
#       subframe=0  wasted_bits=0  type=LPC  order=2  qlp_coeff_precision=12
#                   quantization_level=10  residual_type=RICE  partition_order=0
_FRAME_RE = re.compile(r"^frame=\d+\s")
_SUBFRAME_RE = re.compile(r"^\s+subframe=\d+\s")
_CA_RE = re.compile(r"channel_assignment=(\S+)")
_TYPE_RE = re.compile(r"\btype=(\w+)")
_PARTITION_RE = re.compile(r"\bpartition_order=(\d+)")
# The negative lookbehind matters: the line also carries 'partition_order=',
# so a greedy pattern would match that one instead.
_ORDER_RE = re.compile(r"(?<![A-Za-z_])order=(\d+)")


def _bump(counter: dict, key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _last_line(text: str) -> str:
    """The one meaningful line of an error dump. A line with ERROR wins:
    after it flac prints decoder internals that say nothing on their own.
    The file name it glues on is cut, the caller prints that itself."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return t("err_no_output")
    chosen = next((l for l in lines if "ERROR" in l), lines[-1])
    _, _, after = chosen.partition("ERROR")
    return (after.lstrip(" ,:") or chosen) if after else chosen


def analyze_deep(path: str) -> DeepStats:
    """Take a file apart with `flac -a`, streaming the output line by line.

    Note: `flac -a -o OUT` writes the analysis straight into OUT, it does not
    append .ana - that only happens when -o is omitted. Hence stdout, which
    also saves creating and cleaning up temporary files.
    """
    proc = subprocess.Popen(
        ["flac", "-a", "-s", "-o", "-", "--", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    stats = DeepStats()
    for line in proc.stdout:
        if _SUBFRAME_RE.match(line):
            stats.subframes += 1
            found = _TYPE_RE.search(line)
            if not found:
                continue
            subframe_type = found.group(1)
            _bump(stats.subframe_types, subframe_type)
            if subframe_type == "LPC" and (order := _ORDER_RE.search(line)):
                stats.lpc_orders.append(int(order.group(1)))
            if partition := _PARTITION_RE.search(line):
                stats.partition_orders.append(int(partition.group(1)))
        elif _FRAME_RE.match(line):
            stats.frames += 1
            if ca := _CA_RE.search(line):
                _bump(stats.channel_assignments, ca.group(1))

    proc.stdout.close()
    stderr = proc.stderr.read()
    proc.stderr.close()
    if (code := proc.wait()) != 0:
        raise RuntimeError(t("err_analyze_failed", code=code,
                             detail=_last_line(stderr)))
    if stats.frames == 0:
        raise RuntimeError(t("err_no_frames"))
    return stats


# --------------------------------------------------------------------------
# The outcome of writing to one file
#
# Squeezing a file, correcting its header and salvaging damaged audio are
# three different jobs, but there is nothing different to report about them,
# so they share one record and one printer.
# --------------------------------------------------------------------------

class Kind(str, enum.Enum):
    """What was done to the file. Decides both the wording and whether the
    stored spectral check still describes what is on disk."""

    REENCODE = "reencode"        # squeezed, sample for sample the same audio
    SUBSET = "subset"            # re-encoded to get back inside the subset
    HEADER = "header"            # 24 bytes of STREAMINFO corrected
    SALVAGE = "salvage"          # decoded past the damage, so the audio moved


class Status(str, enum.Enum):
    """How the work on one file turned out."""

    DONE = "done"                # the file on disk was replaced
    NO_GAIN = "no_gain"          # saving below the threshold, original kept
    DRY_RUN = "dry_run"          # measured only, nothing written
    FAILED = "failed"            # error, original kept


@dataclass
class Outcome:
    """Result of one write operation, whichever kind it was."""

    info: FlacInfo
    kind: Kind
    status: Status
    old_size: int = 0
    new_size: int = 0
    recovered_samples: int = 0
    backup_path: str = ""
    meta_ok: bool = True
    error: str | None = None

    @property
    def saved(self) -> int:
        return self.old_size - self.new_size

    @property
    def saved_pct(self) -> float:
        return self.saved / self.old_size * 100 if self.old_size else 0.0

    @property
    def lost_samples(self) -> int:
        return max(0, self.info.total_samples - self.recovered_samples)

    @property
    def lost_seconds(self) -> float:
        rate = self.info.sample_rate
        return self.lost_samples / rate if rate else 0.0

    @property
    def recovered_pct(self) -> float:
        total = self.info.total_samples
        return self.recovered_samples / total * 100 if total else 0.0


# --------------------------------------------------------------------------
# Re-encoding in place
# --------------------------------------------------------------------------


def meets_threshold(saving: int, original_size: int,
                    min_saving_pct: float | None) -> bool:
    """Is the saving on ONE file worth rewriting it?

    `None` means no threshold at all: that is the subset fix, where the point
    is playability rather than space, so the file is rewritten even when it
    grows a little. A number is compared against the saving, and a file that
    gains nothing is never rewritten.

    Judged per file, never from a folder average - a few bad files among a
    hundred good ones would dissolve below the threshold even though rewriting
    exactly those pays off. Measured: 1 file saving 50 % among 80 good ones
    averages 4.17 %, i.e. under the default threshold.

    The threshold only exists to skip pointless work: re-encoding is cheap, so
    anything that really shrinks is worth doing. On a real library files with
    block 1152 save 11-17 % and everything else stays under 1 %, so the default
    sits right in that empty gap.
    """
    if min_saving_pct is None:
        return True
    if original_size <= 0:
        return False
    return saving > 0 and saving / original_size * 100 >= min_saving_pct


def raw_audio_md5(path: str) -> bytes:
    """MD5 of the decoded audio the way FLAC computes it for STREAMINFO:
    raw interleaved samples, little-endian, signed. Fallback for originals
    whose header has no MD5."""
    proc = subprocess.Popen(
        ["flac", "-d", "-c", "--totally-silent", "--force-raw-format",
         "--endian=little", "--sign=signed", "--", path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    digest = hashlib.md5()
    while chunk := proc.stdout.read(CHUNK):
        digest.update(chunk)
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError(t("err_md5_decode"))
    return digest.digest()


def flac_test(path: str) -> None:
    """Raise unless `flac -t` accepts the file."""
    test = subprocess.run(["flac", "-t", "-s", "--", path],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if test.returncode != 0:
        raise RuntimeError(t("err_flac_t", detail=_last_line(
            test.stderr.decode("utf-8", "replace"))))


def assert_lossless(original: FlacInfo, new_path: str) -> None:
    """Verify the new file carries bit-for-bit the same audio as the original.

    Three checks: stream parameters, MD5 of the decoded audio, and `flac -t`
    on the new file (which also catches a bad write). Any doubt raises and the
    caller leaves the original alone.
    """
    new = read_flac_metadata(new_path)
    # Named as STREAMINFO names them: this fires only when the encoder is
    # broken, and then the raw field name is what helps.
    for attr in ("sample_rate", "channels", "bits_per_sample", "total_samples"):
        if getattr(new, attr) != getattr(original, attr):
            raise RuntimeError(t("err_mismatch", field=attr,
                                 old=getattr(original, attr),
                                 new=getattr(new, attr)))

    if new.md5 == MD5_UNSET:
        raise RuntimeError(t("err_md5_unset"))
    reference = original.md5
    if reference == MD5_UNSET:
        reference = raw_audio_md5(original.path)     # work it out ourselves
    if reference != new.md5:
        raise RuntimeError(t("err_md5_differs"))
    flac_test(new_path)


def _copy_owner_and_times(source: str, target: str) -> None:
    """Carry permissions, timestamps and ownership over to the new file."""
    shutil.copystat(source, target)
    try:
        st = os.stat(source)
        os.chown(target, st.st_uid, st.st_gid)
    except (OSError, AttributeError):
        pass                        # not enough rights, never mind


def recompress_file(info: FlacInfo, effort: str, min_saving_pct: float | None,
                    dry_run: bool, kind: Kind = Kind.REENCODE) -> Outcome:
    """Re-encode one file in place.

    The original is replaced only when the new file passes the losslessness
    check and the saving is real. Errors are returned as Status.FAILED rather
    than raised - one broken file must not stop a whole library.
    """
    path = info.path
    old_size = os.path.getsize(path)
    # The temporary file must sit in the same directory so os.replace stays
    # atomic and a full disk shows up before the original is touched. The
    # extension is deliberately not .flac so an interrupted run leaves nothing
    # that looks like music.
    tmp = os.path.join(os.path.dirname(path) or ".",
                       f".{os.path.basename(path)}.recompress.tmp")
    try:
        enc = subprocess.run(["flac", "-s", "-f",
                              *EFFORT_PRESETS[effort], "-o", tmp, "--", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if enc.returncode != 0:
            detail = _last_line(enc.stderr.decode("utf-8", "replace"))
            raise RuntimeError(t("err_encode_failed", code=enc.returncode,
                                 detail=detail))

        assert_lossless(info, tmp)
        new_size = os.path.getsize(tmp)

        if not meets_threshold(old_size - new_size, old_size, min_saving_pct):
            os.remove(tmp)
            return Outcome(info, kind, Status.NO_GAIN, old_size, new_size)
        if dry_run:
            os.remove(tmp)
            return Outcome(info, kind, Status.DRY_RUN, old_size, new_size)

        # flac keeps the modification time itself (--preserve-modtime is its
        # default) but carries over neither mode nor owner.
        _copy_owner_and_times(path, tmp)
        os.replace(tmp, path)       # atomic, the original never disappears
        return Outcome(info, kind, Status.DONE, old_size, new_size)

    except (OSError, RuntimeError, ValueError) as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return Outcome(info, kind, Status.FAILED, old_size, error=str(e))


# --------------------------------------------------------------------------
# Files that lie about their quality
#
# Three different lies, three independent tests, because a file can tell one
# of them without the others:
#
#   lossy source   MP3 and AAC encode nothing above their cutoff, so a FLAC
#                  made from one has a hole there. What is looked for is not
#                  the AMOUNT of treble - an analogue master has little of it
#                  by itself - but the CLIFF: a codec applies a brick-wall
#                  filter, natural material rolls off gradually.
#
#   fake hi-res    Upsampling 44.1 or 48 kHz to 96 kHz invents no ultrasound.
#                  Above the original Nyquist the resampler leaves arithmetic
#                  zero, while a genuine 96 kHz capture always carries SOME
#                  noise up there - tape hiss, dither, air.
#
#   padded depth   16 bit written into a 24 bit container. Not statistics at
#                  all: the low byte of every sample is zero.
#
# Measured on 189 real files (125x 44.1/16, 38x 48/16, 26x 96/24) against a
# corpus built from them by ffmpeg. Worst honest file vs. best fake:
#
#   cliff over 2 bands   honest max 29 dB  |  AAC 128k 49, MP3 128k 61
#   ultrasound gap       honest max 50 dB  |  upsampled 90-97
#
# What still gets through: AAC from about 192 kbps up, which no longer cuts
# the band at all (measured 15 dB - indistinguishable from the original).
# --------------------------------------------------------------------------

LOSSY_CLIFF_DB = 40.0        # cliff from which it looks like a codec
LOSSY_CLIFF_SPAN = 2         # the fall may spread over this many bands
LOSSY_BAND_STEP = 500        # spacing of the measured bands, Hz
LOSSY_BAND_FROM = 14000      # below this codecs do not cut
LOSSY_BAND_TO = 25000        # fine bands end here
LOSSY_CLIFF_TO = 20000       # above this a fall is the natural roll-off
                             # towards CD Nyquist, not a codec
LOSSY_WINDOW = 4096          # Goertzel window length
LOSSY_WINDOW_STRIDE = 8      # every Nth window is taken
LOSSY_SECONDS = 30           # how many seconds of the track are measured
LOSSY_SKIP = 20              # and from which second (skips a quiet intro)
LOSSY_FLOOR_DB = -100.0      # below this it is only numerical noise

ULTRA_FROM = 26000           # the ultrasound is sampled from here,
ULTRA_STEP = 2000            # coarsely - only its level matters
ULTRA_GAP_DB = 70.0          # this far below the 15-20 kHz level = silence
ULTRA_EDGE_DB = 20.0         # a band this far above the floor still has content
ULTRA_MIN_RATE = 48000       # only files claiming more than this are checked

#: Cutoff -> likely source. Ranges, not exact figures: what is detected is the
#: last band BEFORE the cliff, so the real cutoff sits a little higher, and
#: encoders differ. Measured: MP3 128k -> 16.5 kHz, AAC 128k -> 17.0 kHz,
#: MP3 320k -> 20.0 kHz (as reported by this detector).
LOSSY_HINTS = ((17000, "MP3/AAC ~128 kbps"), (19000, "MP3 160-192 kbps"),
               (20500, "MP3 256-320 kbps"), (24000, "MP3 320 kbps"))

#: Cutoff edge -> the sample rate the audio really came from.
UPSAMPLE_HINTS = ((22600, 44100), (24600, 48000))

#: Blackman-Harris window. Its -92 dB side lobes are essential: without them
#: a strong bass leaks into the high bands and the measurement is worthless.
_BH_WINDOW = [0.35875
              - 0.48829 * math.cos(2 * math.pi * i / (LOSSY_WINDOW - 1))
              + 0.14128 * math.cos(4 * math.pi * i / (LOSSY_WINDOW - 1))
              - 0.01168 * math.cos(6 * math.pi * i / (LOSSY_WINDOW - 1))
              for i in range(LOSSY_WINDOW)]


@dataclass
class FakeCheck:
    """What a file claims to be versus what is actually in it."""

    path: str
    bands: dict = field(default_factory=dict)   # Hz -> dB relative to 1 kHz
    cutoff_hz: int = 0                          # brick wall, 0 = none found
    cliff_db: float = 0.0                       # how deep it falls
    ultra_gap_db: float = 0.0                   # how far the ultrasound is down
    source_rate: int = 0                        # real rate if upsampled
    claimed_rate: int = 0
    real_bits: int = 0                          # if it is below claimed_bits
    claimed_bits: int = 0
    error: str | None = None                    # a MESSAGES key

    @property
    def lossy_source(self) -> bool:
        return self.cutoff_hz > 0

    @property
    def upsampled(self) -> bool:
        return self.source_rate > 0

    @property
    def padded_depth(self) -> bool:
        return 0 < self.real_bits < self.claimed_bits

    @property
    def suspicious(self) -> bool:
        return self.lossy_source or self.upsampled or self.padded_depth

    @property
    def hint(self) -> str:
        """Guess of the original format from the cutoff."""
        for limit, name in LOSSY_HINTS:
            if self.cutoff_hz <= limit:
                return name
        return LOSSY_HINTS[-1][1]


def _goertzel(samples: Sequence, rate: int, freq: float) -> float:
    """Energy at a single frequency. O(N) and without any library."""
    coeff = 2 * math.cos(2 * math.pi * freq / rate)
    s1 = s2 = 0.0
    for x in samples:
        s1, s2 = x + coeff * s1 - s2, s1
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def _decode_slice(info: FlacInfo) -> bytes:
    """Decode a slice of the track as raw little-endian PCM.

    Bit depth MUST come from the header: files converted from MP3 are often
    24 bit, and reading them as 16 bit would return noise.
    """
    rate, total = info.sample_rate, info.total_samples
    start = min(LOSSY_SKIP * rate, max(0, total - LOSSY_SECONDS * rate))
    # --until past the end makes flac refuse outright, so a track shorter than
    # LOSSY_SECONDS has to be asked for by its real length, and a file with an
    # unknown length (total_samples 0) not bounded at all.
    span = ["--until=" + str(min(total, start + LOSSY_SECONDS * rate))] if total else []
    return subprocess.run(
        ["flac", "-d", "-c", "-s", "--force-raw-format", "--endian=little",
         "--sign=signed", f"--skip={start}", *span, "--", info.path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout


def _to_mono(raw: bytes, channels: int, width: int, bits: int) -> list:
    """Average the channels into one signal in the range -1..+1."""
    scale = float(1 << (bits - 1))
    frame = width * channels
    mono = []
    for off in range(0, len(raw), frame):
        total = sum(int.from_bytes(raw[off + c * width: off + (c + 1) * width],
                                   "little", signed=True) for c in range(channels))
        mono.append(total / channels / scale)
    return mono


def _effective_bits(raw: bytes, width: int, bits: int) -> int:
    """How many bits the samples really use, or `bits` if they use them all.

    Samples are contiguous little-endian words, so `raw[0::width]` is the low
    byte of every sample of every channel. All zero means the audio was widened
    from a shallower depth and the extra bits carry nothing. Only whole bytes
    are counted - a partial fill (a 20 bit ADC) is honest hardware, not a lie.
    """
    for dead in range(width - 1, 0, -1):
        if all(not any(raw[byte::width]) for byte in range(dead)):
            return bits - dead * 8
    return bits


def check_fake_source(info: FlacInfo) -> FakeCheck:
    """Look for the marks of a lossy source, upsampling and a padded depth."""
    result = FakeCheck(path=info.path, claimed_rate=info.sample_rate,
                       claimed_bits=info.bits_per_sample)
    rate, channels = info.sample_rate, info.channels
    width = info.bits_per_sample // 8
    if not (rate and channels and width):
        result.error = "fake_unusable"
        return result
    try:
        raw = _decode_slice(info)
    except OSError:
        result.error = "fake_unusable"
        return result

    raw = raw[:len(raw) - len(raw) % (width * channels)]
    nyquist = rate / 2
    top = nyquist * 0.95
    fine = [f for f in range(LOSSY_BAND_FROM, LOSSY_BAND_TO + 1, LOSSY_BAND_STEP)
            if f < top]
    coarse = ([f for f in range(ULTRA_FROM, int(top) + 1, ULTRA_STEP)]
              if rate > ULTRA_MIN_RATE else [])
    if len(raw) < width * channels * LOSSY_WINDOW * 2 or len(fine) < 3:
        result.error = "fake_unusable"
        return result

    bits = _effective_bits(raw, width, info.bits_per_sample)
    if bits < info.bits_per_sample:
        result.real_bits = bits

    signal = _to_mono(raw, channels, width, info.bits_per_sample)
    energy = {f: 0.0 for f in fine + coarse}
    reference = 0.0
    for start in range(0, len(signal) - LOSSY_WINDOW,
                       LOSSY_WINDOW * LOSSY_WINDOW_STRIDE):
        chunk = [signal[start + i] * _BH_WINDOW[i] for i in range(LOSSY_WINDOW)]
        reference += _goertzel(chunk, rate, 1000)
        for f in energy:
            energy[f] += _goertzel(chunk, rate, f)

    if reference <= 0:
        result.error = "fake_unusable"
        return result
    result.bands = {f: 10 * math.log10(e / reference + 1e-30)
                    for f, e in energy.items()}
    _find_cliff(result, fine)
    _find_upsampling(result, fine, coarse)
    return result


def _find_cliff(result: FakeCheck, fine: list) -> None:
    """The brick wall a lossy codec leaves behind.

    The fall is measured over LOSSY_CLIFF_SPAN neighbouring bands, not one:
    an encoder's transition band is a few hundred Hz wide, so a single step
    catches only part of it and AAC slips through. Bands already at the
    numerical floor are not compared - a "drop" there is just noise - and a
    fall starting above LOSSY_CLIFF_TO is ignored, because that is where a
    CD-rate file naturally runs out of spectrum anyway.
    """
    for i, f in enumerate(fine[:-1]):
        if f > LOSSY_CLIFF_TO or result.bands[f] < LOSSY_FLOOR_DB:
            continue
        after = fine[i + 1: i + 1 + LOSSY_CLIFF_SPAN]
        if not after:
            break
        drop = result.bands[f] - min(result.bands[g] for g in after)
        if drop > result.cliff_db:
            result.cliff_db, result.cutoff_hz = drop, f
    if result.cliff_db < LOSSY_CLIFF_DB:
        result.cutoff_hz = 0          # gradual roll-off, nothing suspicious


def _find_upsampling(result: FakeCheck, fine: list, coarse: list) -> None:
    """Ultrasound that is not merely quiet but arithmetically absent.

    The level is compared with the file's own 15-20 kHz content rather than
    with a fixed figure, so a quiet or dull recording is judged by the same
    ratio as a bright one.
    """
    if not coarse:
        return
    audible = [result.bands[f] for f in fine if 15000 <= f <= 20000]
    if not audible:
        return
    floor = statistics.median(result.bands[f] for f in coarse)
    result.ultra_gap_db = statistics.median(audible) - floor
    if result.ultra_gap_db < ULTRA_GAP_DB:
        return
    # The highest band still carrying anything marks the original Nyquist.
    edge = max((f for f in fine if result.bands[f] > floor + ULTRA_EDGE_DB),
               default=0)
    result.source_rate = next((r for limit, r in UPSAMPLE_HINTS if edge <= limit),
                              0) or int(edge * 2 / 1000) * 1000


def fake_lines(check: FakeCheck) -> list:
    """One translated line per lie the file tells. Empty if it is honest."""
    lines = []
    if check.lossy_source:
        lines.append(t("fake_lossy", khz=check.cutoff_hz / 1000,
                       db=check.cliff_db, hint=check.hint))
    if check.upsampled:
        lines.append(t("fake_upsampled", claimed=check.claimed_rate // 1000,
                       source=check.source_rate / 1000, edge=check.source_rate / 2000,
                       db=check.ultra_gap_db))
    if check.padded_depth:
        lines.append(t("fake_padded", claimed=check.claimed_bits,
                       real=check.real_bits))
    return lines


# --------------------------------------------------------------------------
# Damaged files
# --------------------------------------------------------------------------

#: Up to this percentage of the track an error counts as "near the start",
#: from that one as "near the end".
DAMAGE_HEAD_PCT, DAMAGE_TAIL_PCT = 10.0, 90.0

#: For every error `flac` reports how many samples it had processed.
_DAMAGE_POS_RE = re.compile(r"after processing (\d+) samples")


@dataclass
class Damage:
    """Where exactly a file is damaged."""

    info: FlacInfo
    decoded_samples: int = 0
    error_positions: list = field(default_factory=list)

    @property
    def lost_samples(self) -> int:
        return max(0, self.info.total_samples - self.decoded_samples)

    @property
    def truncated(self) -> bool:
        return self.lost_samples > 0

    def at(self, samples: int) -> str:
        """Position in the track as 'm:ss.s = xx %' plus a word for it."""
        rate = self.info.sample_rate or 1
        total = self.info.total_samples or 1
        seconds, pct = samples / rate, samples / total * 100
        return t("dmg_at_start" if pct < DAMAGE_HEAD_PCT else
                 "dmg_at_end" if pct > DAMAGE_TAIL_PCT else "dmg_at_middle",
                 time=f"{int(seconds) // 60}:{seconds % 60:04.1f}", pct=pct)

    @property
    def whole_frames_only(self) -> bool:
        """Does the end land exactly on a block boundary?

        If it does and less than one block is missing, this is not damaged
        data but an encoder that dropped the last partial frame while writing
        the true sample count into STREAMINFO. GStreamer does this.
        """
        block = self.info.max_blocksize
        return (self.truncated and block > 0
                and self.decoded_samples % block == 0
                and self.lost_samples < block)

    @property
    def harmless(self) -> bool:
        """No audio was lost - only the sample count in STREAMINFO is wrong.

        Worth separating now that `repair` replaces the original: rewriting a
        whole album for a cosmetic header defect costs a full copy of it in
        backups, so it happens only when explicitly asked for.
        """
        return self.whole_frames_only and not self.error_positions


def diagnose_damage(info: FlacInfo) -> Damage:
    """Locate the damage in one pass of `flac -d -F`.

    `-F` keeps going past errors, so ALL bad positions are found, not just the
    first, and the number of samples that come out can be counted.
    """
    proc = subprocess.run(
        ["flac", "-d", "-F", "-s", "-c", "--force-raw-format",
         "--endian=little", "--sign=signed", "--", info.path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    per_sample = info.channels * (info.bits_per_sample // 8)
    damage = Damage(info=info)
    if per_sample:
        damage.decoded_samples = len(proc.stdout) // per_sample
    stderr = proc.stderr.decode("utf-8", "replace")
    # flac reports the same position more than once (CRC and lost sync).
    seen = []
    for match in _DAMAGE_POS_RE.finditer(stderr):
        position = int(match.group(1))
        if position not in seen:
            seen.append(position)
    damage.error_positions = seen
    return damage


def decodable_audio(info: FlacInfo) -> tuple:
    """(sample count, MD5) of everything the decoder can read, in one pass.

    `-F` keeps going past errors, so this is what the file really contains,
    as opposed to what its header claims. The MD5 is taken over the same raw
    interleaved little-endian signed samples that FLAC hashes for STREAMINFO.
    """
    proc = subprocess.Popen(
        ["flac", "-d", "-F", "-s", "-c", "--force-raw-format",
         "--endian=little", "--sign=signed", "--", info.path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    digest, size = hashlib.md5(), 0
    while chunk := proc.stdout.read(CHUNK):
        digest.update(chunk)
        size += len(chunk)
    proc.stdout.close()
    proc.wait()
    per_sample = info.channels * (info.bits_per_sample // 8)
    return (size // per_sample if per_sample else 0), digest.digest()


def _failed(result: Outcome, error: str) -> Outcome:
    """Mark an outcome as failed, so every repair reports errors alike."""
    result.status, result.error = Status.FAILED, error
    return result


def patch_stream_header(info: FlacInfo, dry_run: bool) -> Outcome:
    """Repair a file whose audio is intact but whose header lies about it.

    The encoder dropped the last partial frame and still wrote the full sample
    count (and the matching MD5) into STREAMINFO, so `flac -t` fails on a file
    that has lost nothing. Re-encoding it would be pointless work on tens of
    megabytes: the frames are fine, only 24 bytes of header are wrong.

    So the file is copied and just those bytes are corrected - the packed
    field holding total_samples, and the MD5 recomputed from what the file
    actually contains. Every audio byte stays exactly as it was, which makes
    this the most conservative repair in the script and the reason it keeps
    no backup: there is nothing to lose that the file did not already lack.
    """
    path = info.path
    result = Outcome(info=info, kind=Kind.HEADER,
                     status=Status.DRY_RUN if dry_run else Status.DONE)
    samples, digest = decodable_audio(info)
    result.recovered_samples = samples
    if not samples:
        return _failed(result, t("err_no_frames"))
    if dry_run:
        return result

    try:
        # Where STREAMINFO sits is read from the file itself, never from the
        # report: an ID3 tag in front of 'fLaC' moves it, and writing 24 bytes
        # to a guessed offset would corrupt the header.
        live = read_flac_metadata(path)
    except (OSError, ValueError) as e:
        return _failed(result, str(e))

    # sample_rate(20) channels(3) bits_per_sample(5) total_samples(36),
    # exactly as parse_flac_headers reads it back out.
    packed = ((live.sample_rate << 44) | ((live.channels - 1) << 41)
              | ((live.bits_per_sample - 1) << 36) | samples)

    tmp = os.path.join(os.path.dirname(path) or ".",
                       f".{os.path.basename(path)}.header.tmp")
    try:
        shutil.copyfile(path, tmp)
        with open(tmp, "r+b") as f:
            f.seek(live.streaminfo_offset + 10)
            f.write(packed.to_bytes(8, "big") + digest)
        # The patched copy has to decode cleanly before it replaces anything;
        # that is what makes the missing backup acceptable here.
        flac_test(tmp)
        _copy_owner_and_times(path, tmp)
        os.replace(tmp, path)
    except (OSError, RuntimeError, ValueError) as e:
        _failed(result, str(e))
        try:
            os.remove(tmp)
        except OSError:
            pass
    return result


def repair_damaged_file(info: FlacInfo, effort: str, dry_run: bool) -> Outcome:
    """Salvage what is readable from a damaged file and put it in its place.

    `flac -d -F` keeps decoding past errors, so everything readable comes out
    of a truncated or corrupted stream.

    The original is not thrown away: it is renamed to <name>.orig.flac first.
    Losslessness cannot be verified here - part of the audio is already gone
    and the MD5 in the header will never match again - so the untouched
    original stays the last chance for a better recovery later. An existing
    backup is never overwritten, otherwise a second run would replace the true
    original with an already-salvaged one.
    """
    path = info.path
    backup = os.path.splitext(path)[0] + ORIG_SUFFIX
    result = Outcome(info=info, kind=Kind.SALVAGE, backup_path=backup,
                     status=Status.DRY_RUN if dry_run else Status.DONE)

    if os.path.exists(backup):
        return _failed(result, t("fix_exists", name=os.path.basename(backup)))

    tmp = os.path.join(os.path.dirname(path) or ".",
                       f".{os.path.basename(path)}.repair.tmp")
    target = os.devnull if dry_run else tmp
    try:
        dec = subprocess.Popen(
            ["flac", "-d", "-F", "-s", "-c", "--", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        enc = subprocess.run(
            ["flac", "-s", "-f", *EFFORT_PRESETS[effort], "-o", target, "-"],
            stdin=dec.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        dec.stdout.close()
        dec.wait()
        if enc.returncode != 0:
            raise RuntimeError(_last_line(enc.stderr.decode("utf-8", "replace")))

        if dry_run:
            # Nothing to measure from the output, so count at least how many
            # samples the decoder yields.
            result.recovered_samples = decodable_audio(info)[0]
            return result

        # Tags and cover art live in the original's metadata blocks, which the
        # decode-and-encode pipe does not carry over. A file damaged in its
        # audio still has readable metadata; if it does not, the audio is
        # worth more than the tags.
        try:
            graft_metadata(read_metadata_blocks(path), tmp)
        except (OSError, ValueError):
            result.meta_ok = False

        flac_test(tmp)
        result.recovered_samples = read_flac_metadata(tmp).total_samples
        if not result.recovered_samples:
            raise RuntimeError(t("err_no_frames"))

        _copy_owner_and_times(path, tmp)
        os.rename(path, backup)         # same directory, so atomic
        try:
            os.replace(tmp, path)
        except OSError:
            os.rename(backup, path)     # put the original back
            raise
    except (OSError, RuntimeError, ValueError) as e:
        _failed(result, str(e))
        if not dry_run:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return result


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

class Severity(enum.IntEnum):
    """How suspicious a file is. Comparable, higher is worse."""

    OK = 0
    WARN = 1
    BAD = 2

    @property
    def symbol(self) -> str:
        return ("✅", "⚠️ ", "🚨")[self]

    @property
    def label(self) -> str:
        """Sentence case, so a summary row can use it as it is."""
        return t(("sev_ok", "sev_warn", "sev_bad")[self])

    @property
    def heading(self) -> str:
        """Shouted over a file with a finding, quiet over one that is fine."""
        return self.label.upper() if self else self.label


@dataclass
class Analysis:
    """A file together with everything found out about it."""

    info: FlacInfo
    severity: Severity = Severity.OK
    #: Reasons and violations are kept as (message key, params), never as
    #: finished strings - a report written in Czech must print in English too.
    reasons: list = field(default_factory=list)
    subset_fixable: list = field(default_factory=list)
    subset_inherent: list = field(default_factory=list)
    damage: Damage | None = None
    fake: FakeCheck | None = None
    partial: bool = False        # rewritten since, deep data no longer valid

    @property
    def weak(self) -> bool:
        """Candidate for re-encoding because of the compression level."""
        return self.severity is not Severity.OK

    @property
    def damaged(self) -> bool:
        return self.info.damaged

    @property
    def harmless(self) -> bool:
        """Damaged only by a header the encoder got wrong, no audio missing."""
        return bool(self.damage and self.damage.harmless)

    @property
    def really_damaged(self) -> bool:
        return self.damaged and not self.harmless

    @property
    def unreadable(self) -> bool:
        return bool(self.info.error) and not self.info.damaged

    @property
    def needs_attention(self) -> bool:
        """Worth printing without --all."""
        return bool(self.weak or self.subset_fixable or self.subset_inherent
                    or self.info.error
                    or (self.fake and self.fake.suspicious))


def blocksizes_for_rate(sample_rate: int) -> tuple:
    """(weak, normal) block sizes for a sample rate."""
    for limit, low, normal in BLOCKSIZE_TABLE:
        if limit is None or sample_rate <= limit:
            return low, normal
    return BLOCKSIZE_TABLE[-1][1], BLOCKSIZE_TABLE[-1][2]


def rate_is_codable(sample_rate: int) -> bool:
    """Can the rate go into the frame header, as the subset requires?

    Besides the coded values the header can hold whole kHz up to 255, any Hz
    up to 65535, and tens of Hz up to 655350.
    """
    return (sample_rate in SUBSET_CODED_RATES
            or (sample_rate % 1000 == 0 and sample_rate // 1000 <= 255)
            or sample_rate <= 65535
            or (sample_rate % 10 == 0 and sample_rate // 10 <= 65535))


def subset_violations(info: FlacInfo) -> tuple:
    """(fixable, inherent) subset violations; empty lists mean it is fine.

    Fixable ones are encoder parameters, so re-encoding with -8 removes them:
    -8 uses block 4096, LPC order 12 and partition order 6, all inside the
    subset at every sample rate. Inherent ones are properties of the audio -
    changing the bit depth or the sample rate means dithering or resampling,
    which is not lossless, so they are only ever reported.
    """
    fixable, inherent = [], []
    limit = SUBSET_MAX_BLOCKSIZE[info.hi_rate]
    if info.max_blocksize > limit:
        fixable.append(("sub_blocksize", {"size": info.max_blocksize,
                                          "limit": limit, "rate": info.sample_rate}))
    if info.bits_per_sample not in SUBSET_BITS_PER_SAMPLE:
        allowed = "/".join(map(str, sorted(SUBSET_BITS_PER_SAMPLE)))
        inherent.append(("sub_bps", {"bps": info.bits_per_sample,
                                     "allowed": allowed}))
    if not rate_is_codable(info.sample_rate):
        inherent.append(("sub_rate", {"rate": info.sample_rate}))

    if info.deep:
        order = info.deep.max_lpc_order
        # The LPC order limit only applies up to 48 kHz, above that the format
        # is looser. The partition order limit applies everywhere.
        if not info.hi_rate and order is not None and order > SUBSET_MAX_LPC_ORDER:
            fixable.append(("sub_lpc", {"order": order,
                                        "limit": SUBSET_MAX_LPC_ORDER,
                                        "rate": info.sample_rate}))
        partition = info.deep.max_partition_order
        if partition is not None and partition > SUBSET_MAX_PARTITION_ORDER:
            fixable.append(("sub_partition", {"order": partition,
                                              "limit": SUBSET_MAX_PARTITION_ORDER}))
    return fixable, inherent


def classify(info: FlacInfo) -> Analysis:
    """Decide whether a file looks weakly compressed, and why."""
    if info.error:
        return Analysis(info=info)

    reasons, severity = [], Severity.OK
    low_sizes, normal_sizes = blocksizes_for_rate(info.sample_rate)

    if info.max_blocksize in low_sizes or info.max_blocksize < MIN_SANE_BLOCKSIZE:
        severity = Severity.BAD
        expected = "/".join(map(str, sorted(low_sizes)))
        reasons.append(("why_blocksize_low", {"size": info.max_blocksize,
                                              "rate": info.sample_rate,
                                              "expected": expected}))
    elif info.max_blocksize not in normal_sizes:
        reasons.append(("why_blocksize_odd", {"size": info.max_blocksize,
                                              "rate": info.sample_rate}))

    if deep := info.deep:
        if deep.lpc_share == 0.0:
            severity = max(severity, Severity.BAD)
            reasons.append(("why_no_lpc", {}))
        elif deep.lpc_share < MIN_LPC_SHARE:
            severity = max(severity, Severity.WARN)
            reasons.append(("why_low_lpc", {"pct": deep.lpc_share * 100}))

        # The highest LPC order is supporting information, never a reason to
        # flag: the encoder picks the order from the material, so even an
        # honest -8 track settles for order 6 on simple content.
        order = deep.max_lpc_order
        if order is not None and order <= 6 and severity is not Severity.OK:
            reasons.append(("why_low_order", {"order": order}))

        if info.channels > 1 and deep.stereo_mode == STEREO_NONE:
            severity = max(severity, Severity.WARN)
            reasons.append(("why_no_stereo", {}))

    fixable, inherent = subset_violations(info)
    return Analysis(info, severity, reasons, fixable, inherent)


# --------------------------------------------------------------------------
# Report file
#
# The slow pass runs once, in `analyze`; `reencode` and `repair` only read
# what it found. Every entry carries size and mtime so a file edited in the
# meantime can be skipped instead of acted on from stale findings.
# --------------------------------------------------------------------------

def default_report_path(root: str) -> str:
    """One report per analysed folder, kept out of the music itself so that
    read-only or network mounts work and nothing foreign lands in a library."""
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    real = os.path.realpath(root)
    slug = re.sub(r"[^\w.-]+", "_", os.path.basename(real) or "root")[:40]
    digest = hashlib.sha1(real.encode("utf-8")).hexdigest()[:10]
    return os.path.join(cache, "flac_check", f"{slug}-{digest}.json")


@dataclass
class Report:
    root: str
    created: str = ""
    flac_version: str = ""
    items: list = field(default_factory=list)     # list[Analysis]
    path: str = ""

    def by_path(self) -> dict:
        return {a.info.path: a for a in self.items}


#: Fields a class does not keep in the report: back references to the file
#: itself, offsets that are read from disk anyway, and the band table, which
#: is far bigger than the verdict drawn from it.
NOT_STORED = {FlacInfo: {"streaminfo_offset"}, Damage: {"info"},
              FakeCheck: {"path", "bands"}, Report: {"path"}}

#: Fields that need more than a plain JSON value on the way back.
REVIVE = {"md5": bytes.fromhex, "severity": Severity,
          "deep": lambda d: from_json(DeepSummary, d),
          # JSON turns the (key, params) tuples into lists; put them back so
          # the rest of the code sees what a fresh analysis produces.
          "reasons": lambda raw: [(k, p) for k, p in raw],
          "subset_fixable": lambda raw: [(k, p) for k, p in raw],
          "subset_inherent": lambda raw: [(k, p) for k, p in raw]}


def to_json(obj) -> dict:
    """Any of the report's dataclasses as plain JSON types, nested ones and
    all. Driven by the field list, so adding a field stores it too."""
    out = {}
    for f in fields(obj):
        if f.name in NOT_STORED.get(type(obj), ()):
            continue
        value = getattr(obj, f.name)
        if isinstance(value, bytes):
            value = value.hex()
        elif is_dataclass(value):
            value = to_json(value)
        elif isinstance(value, list) and value and is_dataclass(value[0]):
            value = [to_json(v) for v in value]
        out[f.name] = value
    return out


def from_json(cls, d: dict, **extra):
    """Rebuild a dataclass from JSON. Anything the file does not carry keeps
    the field default, so a report may gain fields without breaking."""
    kw = dict(extra)
    for f in fields(cls):
        if (f.name in kw or f.name in NOT_STORED.get(cls, ())
                or d.get(f.name) is None):
            continue
        revive = REVIVE.get(f.name)
        kw[f.name] = revive(d[f.name]) if revive else d[f.name]
    return cls(**kw)


def _analysis_from_json(d: dict) -> Analysis:
    """One entry. Only the two back references to the file itself have to be
    named here; every other field follows from the dataclass.
    """
    info = from_json(FlacInfo, d["info"])
    item = from_json(Analysis, d, info=info)
    if raw := d.get("damage"):
        item.damage = from_json(Damage, raw, info=info)
    if raw := d.get("fake"):
        item.fake = from_json(FakeCheck, raw, path=info.path)
    return item


def save_report(report: Report) -> None:
    """Write the report atomically, creating the cache directory if needed."""
    os.makedirs(os.path.dirname(report.path) or ".", exist_ok=True)
    payload = {"version": REPORT_VERSION, **to_json(report)}
    tmp = report.path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, report.path)


def load_report(path: str) -> Report | None:
    """Read a report, or None when there is none. Raises ValueError on a
    file that exists but cannot be used."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != REPORT_VERSION:
            raise ValueError(f"version {payload.get('version')}")
        return from_json(Report, payload, path=path,
                         items=[_analysis_from_json(d)
                                for d in payload["items"]])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(str(e)) from e


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def files(n: int) -> str:
    """The only noun the script ever counts. Czech needs three forms
    (1 soubor, 2 soubory, 5 souborů), English two."""
    if _lang == "cs":
        word = "soubor" if n == 1 else "soubory" if 2 <= n <= 4 else "souborů"
    else:
        word = "file" if n == 1 else "files"
    return f"{n} {word}"


def human(n: float) -> str:
    """Bytes readably, e.g. '3.7 MB'."""
    size = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(size) < 1024 or unit == "GB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def estimate(total_bytes: int, rate_per_job: int, jobs: int) -> str:
    """Rough guess at how long the next step will run."""
    return "~" + duration(total_bytes / 1048576 / (rate_per_job * max(1, jobs)))


# --------------------------------------------------------------------------
# Progress
#
# A library is gigabytes and every pass decodes all of it, so the run has to
# say where it is. Progress goes to stderr, which keeps a redirected report
# clean.
# --------------------------------------------------------------------------

BAR_WIDTH = 24
QUIET_INTERVAL = 15.0        # seconds between lines when there is no terminal


class Progress:
    """Live counter on one line; periodic plain lines without a terminal.

    Weighted by bytes when they are known, because file sizes differ enough
    that counting files alone gives a misleading estimate.
    """

    def __init__(self, label: str, total_items: int, total_bytes: int = 0):
        self.label = label
        self.total_items = max(1, total_items)
        self.total_bytes = total_bytes
        self.done_items = 0
        self.done_bytes = 0
        self.started = time.monotonic()
        self.last_print = 0.0
        self.tty = sys.stderr.isatty()
        self.width = shutil.get_terminal_size((100, 24)).columns
        if label:
            self._draw(force=True)

    @property
    def _fraction(self) -> float:
        if self.total_bytes:
            return min(1.0, self.done_bytes / self.total_bytes)
        return min(1.0, self.done_items / self.total_items)

    def advance(self, weight: int = 0) -> None:
        self.done_items += 1
        self.done_bytes += weight
        self._draw()

    def reset(self) -> None:
        """Start the count over, e.g. when a pool had to be swapped out."""
        self.done_items = self.done_bytes = 0
        self.started = time.monotonic()

    def _draw(self, force: bool = False) -> None:
        if not self.label:
            return
        now = time.monotonic()
        if not force and now - self.last_print < (0.1 if self.tty else QUIET_INTERVAL):
            return
        self.last_print = now

        fraction = self._fraction
        elapsed = now - self.started
        tail = ""
        if fraction > 0.02 and elapsed > 2:
            tail = "  " + t("prog_eta",
                            eta=duration(elapsed / fraction - elapsed))
        counter = f"{self.done_items}/{self.total_items}"
        if self.tty:
            filled = round(BAR_WIDTH * fraction)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            line = f"  {self.label}  {bar} {fraction * 100:3.0f} % {counter}{tail}"
            sys.stderr.write("\r" + line[:self.width - 1].ljust(self.width - 1))
            sys.stderr.flush()
        else:
            print(f"  {self.label}  {fraction * 100:.0f} % {counter}{tail}",
                  file=sys.stderr, flush=True)

    def close(self) -> None:
        if not self.label:
            return
        elapsed = duration(time.monotonic() - self.started)
        line = f"  {self.label}  {t('prog_done', elapsed=elapsed)}"
        if self.tty:
            sys.stderr.write("\r" + line[:self.width - 1].ljust(self.width - 1) + "\n")
            sys.stderr.flush()
        else:
            print(line, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Walking the folder
# --------------------------------------------------------------------------

def collect_flac_files(root: str) -> list:
    """Every .flac under `root`, sorted. Symlinked directories are not
    followed, and the backups `repair` leaves behind are skipped - they are
    damaged by definition and would be reported forever."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        found += [os.path.join(dirpath, n) for n in sorted(filenames)
                  if n.lower().endswith(".flac")
                  and not n.lower().endswith(ORIG_SUFFIX)]
    return found


def run_parallel(label: str, items: Sequence, worker: Callable, jobs: int,
                 weight: Callable | None = None, cpu_bound: bool = False) -> list:
    """Run `worker` over the items in parallel, reporting progress.

    Threads are enough while the work happens in a subprocess (`flac`), which
    releases the GIL meanwhile. For counting in Python itself threads do not
    help at all - measured on the spectral analysis of 189 files, where 12
    threads took as long as one - so that work goes to processes.
    """
    items = list(items)
    if not items:
        return []
    total_bytes = sum(weight(i) for i in items) if weight else 0
    progress = Progress(label, len(items), total_bytes)

    def drive(pool_type) -> list:
        results = [None] * len(items)
        with pool_type(max_workers=jobs) as pool:
            pending = {pool.submit(worker, item): n for n, item in enumerate(items)}
            for future in as_completed(pending):
                n = pending[future]
                results[n] = future.result()
                progress.advance(weight(items[n]) if weight else 0)
        return results

    try:
        try:
            return drive(ProcessPoolExecutor if cpu_bound else ThreadPoolExecutor)
        except OSError:
            if not cpu_bound:
                raise
            # Processes may be unavailable (limits, containers); slow beats never.
            progress.reset()
            return drive(ThreadPoolExecutor)
    finally:
        progress.close()


def _by_size(item) -> int:
    """Progress weight: files differ by an order of magnitude in size."""
    info = getattr(item, "info", item)
    return getattr(info, "file_size", 0)


def read_all_metadata(paths: Sequence, jobs: int) -> list:
    """Headers of every file. Unreadable ones come back carrying `error`."""
    def one(path: str) -> FlacInfo:
        try:
            return read_flac_metadata(path)
        except (OSError, ValueError) as e:
            return FlacInfo(path=path, error=str(e))
    return run_parallel(t("run_meta", count=len(paths)), paths, one, jobs)


def deep_worker(info: FlacInfo) -> FlacInfo:
    try:
        info.deep = analyze_deep(info.path).summary()
    except RuntimeError as e:
        # The decoder failed on the file: the audio is damaged, not merely
        # oddly encoded. That is a more serious finding than any level.
        info.error, info.damaged = str(e), True
    except OSError as e:
        info.error = str(e)
    return info


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def rel(path: str, root: str) -> str:
    return os.path.relpath(path, root)


def print_file_report(item: Analysis, root: str) -> None:
    """Everything known about one file."""
    info = item.info
    # A file that is fine on compression but outside the subset must not be
    # headed "fine" - the violation printed below would contradict it.
    subset_only = not item.weak and (item.subset_fixable or item.subset_inherent)
    print(f"{'📻' if subset_only else item.severity.symbol} {rel(info.path, root)}")
    if not subset_only:
        print(f"   {item.severity.heading}"
              + (f"  {t('rep_partial')}" if item.partial else ""))

    detail = [f"{info.sample_rate / 1000:g} kHz/{info.bits_per_sample} bit",
              f"{t('rep_block')}={info.max_blocksize}",
              *([t("rep_compression", pct=info.ratio * 100)]
                if info.ratio is not None else []),
              human(info.file_size),
              *([t("rep_encoder", vendor=info.vendor)] if info.vendor else [])]
    print(f"   {', '.join(detail)}")

    if deep := info.deep:
        types = ", ".join(f"{n}={c}" for n, c in sorted(deep.subframe_types.items()))
        print("   " + ", ".join(
            [t("rep_subframes", types=types),
             *([t("rep_lpc", avg=deep.avg_lpc_order, max=deep.max_lpc_order)]
               if deep.lpc_orders else []),
             t("rep_stereo", mode=deep.stereo_mode)]))

    for reason in item.reasons:
        print(f"   - {tr(reason)}")

    if item.subset_fixable or item.subset_inherent:
        print("   " + t("sub_header"))
        for issues, note in ((item.subset_fixable, "sub_fixable"),
                             (item.subset_inherent, "sub_inherent")):
            for issue in issues:
                print(f"      - {tr(issue)}")
            if issues:
                print(f"        {t(note)}")

    if item.fake and item.fake.suspicious:
        for line in fake_lines(item.fake):
            print("   🎭 " + line)
    print()


def damage_lines(damage: Damage) -> list:
    """Where exactly a file is damaged, one line per fact."""
    lines = [t("dmg_error_at", where=damage.at(p)) for p in damage.error_positions]
    if damage.truncated:
        seconds = damage.lost_samples / (damage.info.sample_rate or 1)
        lines.append(t("dmg_truncated", where=damage.at(damage.decoded_samples),
                       samples=damage.lost_samples, seconds=seconds))
        if damage.whole_frames_only:
            lines.append(t("dmg_encoder_bug",
                           vendor=damage.info.vendor or STEREO_UNKNOWN))
    elif not damage.error_positions:
        lines.append(t("dmg_unknown_position"))
    return lines


def print_group(header: str, rows: Sequence, footer: str = "",
                blank: bool = True) -> None:
    """A headed block: one file per line, its details indented underneath.

    Every listing in the script has this shape - damaged, unreadable, lying -
    so routing them all through here keeps the indentation identical.
    """
    if not rows:
        return
    print(header)
    for name, details in rows:
        print(f"   {name}")
        for line in details:
            print(f"      {line}")
    if footer:
        print(f"   {footer}")
    if blank:
        print()


def print_fakes(checks: Sequence, root: str, blank: bool = True) -> None:
    """Files whose content does not match what their header claims."""
    print_group(t("fake_header", count=files(len(checks))),
                [(rel(c.path, root), fake_lines(c))
                 for c in sorted(checks, key=lambda c: c.path)],
                footer=t("fake_caveat"), blank=blank)


def print_findings(report: Report, show_all: bool) -> None:
    """The four sections of a report: findings, damaged, unreadable, lying."""
    root = report.root
    listed = [a for a in report.items
              if not a.info.error and (show_all or a.needs_attention)]
    for item in sorted(listed, key=lambda a: (-a.severity, a.info.path)):
        print_file_report(item, root)

    def by_name(items: Sequence) -> list:
        return sorted(items, key=lambda a: a.info.path)

    for attr, header in (("really_damaged", "dmg_header"),
                         ("harmless", "dmg_harmless_header")):
        if found := by_name([a for a in report.items if getattr(a, attr)]):
            print_group(t(header, count=files(len(found))),
                        [(rel(a.info.path, root),
                          damage_lines(a.damage) if a.damage
                          else [a.info.error] if a.info.error else [])
                         for a in found])

    if found := by_name([a for a in report.items if a.unreadable]):
        print_group(t("unreadable_header"),
                    [(f"{rel(a.info.path, root)}: {a.info.error}", [])
                     for a in found])

    if found := [a for a in report.items if a.fake and a.fake.suspicious]:
        print_fakes([a.fake for a in found], root)


#: The symbol each status is announced with. Two characters wide throughout,
#: so the file names line up whatever happened to them.
MARKS = {Status.DONE: "✅", Status.DRY_RUN: "○ ",
         Status.NO_GAIN: "○ ", Status.FAILED: "❌"}


def result_detail(res: Outcome, min_saving: float) -> str:
    """The one line saying what happened to a file.

    Failure and "nothing gained" read the same whatever was attempted, so
    only a real result is worded per kind.
    """
    if res.status is Status.FAILED:
        key = "rc_failed" if res.kind in (Kind.REENCODE, Kind.SUBSET) else "fix_failed"
        return t(key, error=res.error)
    if res.status is Status.NO_GAIN:
        return t("rc_skipped", min=min_saving, saved=human(res.saved))

    dry = res.status is Status.DRY_RUN
    if res.kind is Kind.REENCODE:
        return (t("rc_would", saved=human(res.saved), pct=res.saved_pct) if dry
                else t("rc_replaced", old=human(res.old_size),
                       new=human(res.new_size), saved=human(res.saved),
                       pct=res.saved_pct))
    if res.kind is Kind.SUBSET:
        # Getting back inside the subset may cost a little space, so the
        # change is worded rather than printed as a saving.
        change = (t("rc_saved_pct", pct=res.saved_pct) if res.saved > 0
                  else t("rc_grew", pct=-res.saved_pct))
        return (t("rc_would_subset", change=change) if dry
                else t("rc_subset_done", old=human(res.old_size),
                       new=human(res.new_size), change=change))
    if res.kind is Kind.HEADER:
        return t("fix_header_done", old=res.info.total_samples,
                 new=res.recovered_samples)
    lost = (t("fix_lost", seconds=res.lost_seconds) if res.lost_samples
            else t("fix_no_loss"))
    return t("fix_recovered", pct=res.recovered_pct, lost=lost)


def print_result(res: Outcome, root: str, min_saving: float = 0.0) -> None:
    """One outcome: what happened, and what it left behind."""
    lines = [result_detail(res, min_saving)]
    if res.kind in (Kind.HEADER, Kind.SALVAGE) and res.status is Status.DRY_RUN:
        lines.append(t("fix_dry"))
    elif res.status is Status.DONE and res.backup_path:
        lines.append(t("fix_backed_up", name=os.path.basename(res.backup_path)))
        if not res.meta_ok:
            lines.append(t("fix_no_meta"))

    print(f"{MARKS[res.status]} {rel(res.info.path, root)}")
    for line in lines:
        print(f"   {line}")


def print_table(rows: Sequence) -> None:
    if not rows:
        return
    print("-" * 60)
    for label, value in rows:
        print(f"{label + ':':<38}{value}")


def print_advice(lines: Sequence) -> None:
    if lines:
        print("\n" + t("adv_next"))
        for line in lines:
            print(f"  {line}")


# --------------------------------------------------------------------------
# Shared command helpers
# --------------------------------------------------------------------------

def prog_name() -> str:
    return os.path.basename(sys.argv[0]) or "flac_check.py"


def short(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home + os.sep) else path


def quoted(path: str) -> str:
    """Shell-safe path that keeps a leading ~ outside the quotes, so the
    suggested command can be pasted and still expands to the home folder."""
    if path.startswith("~" + os.sep):
        return "~" + os.sep + shlex.quote(path[2:])
    return shlex.quote(path)


def command_hint(sub: str, root: str, extra: str = "") -> str:
    return f"{prog_name()} {sub} {quoted(short(root))}{extra}"


def ask_yes_no() -> bool:
    """Shared confirmation. Always refuses when stdin is not a terminal."""
    if not sys.stdin.isatty():
        print(t("rc_not_tty"), file=sys.stderr)
        return False
    accepted = ("a", "ano", "y", "yes") if _lang == "cs" else ("y", "yes")
    try:
        return input(t("rc_prompt") + " ").strip().lower() in accepted
    except EOFError:
        return False


def require_report(args) -> Report | None:
    """Load the report for the analysed folder, or explain how to make one."""
    path = args.report or default_report_path(args.folder)
    try:
        report = load_report(path)
    except ValueError as e:
        print(t("rp_broken", path=path, error=e), file=sys.stderr)
        return None
    if report is None:
        print(t("rp_missing", root=short(args.folder),
                cmd=command_hint("analyze", args.folder)), file=sys.stderr)
        return None
    print(t("rp_from", when=report.created, count=files(len(report.items))))
    return report


def live_targets(items: Sequence, root: str) -> tuple:
    """Split targets into those still matching the report and those that
    changed on disk since. Acting on stale findings could rewrite a file for
    reasons that no longer hold."""
    fresh, skipped = [], []
    for item in items:
        name = rel(item.info.path, root)
        try:
            stat = os.stat(item.info.path)
        except OSError:
            skipped.append(t("rp_gone_file", name=name))
            continue
        if (stat.st_size != item.info.file_size
                or stat.st_mtime_ns != item.info.mtime_ns):
            skipped.append(t("rp_stale_file", name=name))
            continue
        fresh.append(item)
    return fresh, skipped


def refresh_entry(item: Analysis, keep_fake: bool) -> None:
    """Update one entry after its file was rewritten.

    Only the header is read again - re-running the deep analysis here would
    decode the whole library a second time. The entry is marked partial so
    both the listing and the next `analyze` know the stream itself has not
    been looked at since.
    """
    fake = item.fake if keep_fake else None
    try:
        info = read_flac_metadata(item.info.path)
    except (OSError, ValueError) as e:
        item.info.error, item.info.damaged = str(e), False
        return
    fresh = classify(info)
    item.info = info
    item.severity = fresh.severity
    item.reasons = fresh.reasons
    item.subset_fixable = fresh.subset_fixable
    item.subset_inherent = fresh.subset_inherent
    item.damage = None
    item.fake = fake
    item.partial = True


def store(report: Report, args) -> None:
    report.created = time.strftime("%Y-%m-%d %H:%M")
    save_report(report)
    if getattr(args, "verbose_report", True):
        print(t("rp_saved", path=short(report.path)))


def count_rows(pairs: Sequence, always: bool = False) -> list:
    """(message key, count) pairs as table rows. A zero is dropped unless
    `always`, so a summary only mentions what actually happened."""
    return [(t(key), n) for key, n in pairs if n or always]


def summary_rows(report: Report) -> list:
    items = report.items

    def count(test: Callable) -> int:
        return sum(1 for a in items if test(a))

    return count_rows((("sum_total", len(items)),
                       ("sev_bad", count(lambda a: a.severity is Severity.BAD)),
                       ("sev_warn", count(lambda a: a.severity is Severity.WARN))),
                      always=True) + count_rows(
        (("sum_damaged", count(lambda a: a.really_damaged)),
         ("sum_harmless", count(lambda a: a.harmless)),
         ("sum_unreadable", count(lambda a: a.unreadable)),
         ("sum_subset_fix", count(lambda a: a.subset_fixable)),
         ("sum_subset_keep", count(lambda a: a.subset_inherent)),
         ("sum_fake", count(lambda a: a.fake and a.fake.suspicious))))


def report_advice(report: Report, args) -> list:
    """The one command that makes sense next."""
    root, advice = report.root, []
    # repair fixes defects, reencode saves space - two different questions.
    broken = [a for a in report.items
              if a.really_damaged or a.harmless
              or (a.subset_fixable and not a.info.error)]
    weak = [a for a in report.items if not a.info.error and a.weak]
    inherent = [a for a in report.items if a.subset_inherent]
    unchecked = [a for a in report.items if not a.unreadable and not a.fake]
    lying = [a for a in report.items if a.fake and a.fake.suspicious]

    if broken:
        advice.append(t("adv_repair", cmd=command_hint("repair", root),
                        count=files(len(broken))))
    if weak:
        size = sum(a.info.file_size for a in weak)
        advice.append(t("adv_reencode", cmd=command_hint("reencode", root),
                        count=files(len(weak)),
                        eta=estimate(size, ENCODE_RATE, args.jobs)))
    if inherent:
        advice.append(t("adv_subset_keep", count=files(len(inherent))))
    if lying:
        advice.append(t("adv_fake", count=files(len(lying))))
    if unchecked:
        size = sum(a.info.file_size for a in unchecked)
        advice.append(t("adv_findfake", cmd=command_hint("find-fake", root),
                        eta=estimate(size, FAKE_RATE, args.jobs)))
    return advice


def print_outcome(report: Report, args) -> None:
    """Verdict first, then the one command that makes sense next."""
    if not any(a.weak or a.subset_fixable or a.subset_inherent or a.info.error
               or (a.fake and a.fake.suspicious) for a in report.items):
        print("\n" + t("adv_clean"))
    print_advice(report_advice(report, args))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_analyze(args) -> int:
    root = args.folder
    print(t("run_scanning", root=short(root)))
    paths = collect_flac_files(root)
    if not paths:
        print(t("run_none"))
        return 0

    report_file = args.report or default_report_path(root)
    previous = {}
    if not args.force:
        try:
            if old := load_report(report_file):
                previous = old.by_path()
        except ValueError:
            previous = {}            # unusable report, simply analyse again

    infos = read_all_metadata(paths, args.jobs)
    items, todo = [], []
    for info in infos:
        old = previous.get(info.path)
        # Reuse only a complete entry for a file that has not moved a byte.
        if (old and not old.partial and old.info.file_size == info.file_size
                and old.info.mtime_ns == info.mtime_ns
                and (old.info.deep or old.info.error)):
            items.append(old)
        else:
            items.append(None)
            todo.append(info)

    if previous:
        print("  " + t("rp_reused", count=len(paths) - len(todo), fresh=len(todo)))

    readable = [i for i in todo if not i.error]
    if readable:
        run_parallel(t("run_deep", count=len(readable)), readable, deep_worker,
                     args.jobs, weight=_by_size)

    fresh = [classify(info) for info in todo]
    if broken := [a for a in fresh if a.damaged]:
        reports = run_parallel(t("dmg_locating", count=files(len(broken))),
                               broken, lambda a: diagnose_damage(a.info),
                               args.jobs, weight=_by_size)
        for item, damage in zip(broken, reports):
            item.damage = damage

    queue = iter(fresh)
    items = [item if item is not None else next(queue) for item in items]

    report = Report(root=root, flac_version=_dotted(flac_version() or ()),
                    items=items, path=report_file)
    print()
    print_findings(report, args.all)
    print_table(summary_rows(report))
    store(report, args)
    print_outcome(report, args)
    return 0


def cmd_show(args) -> int:
    report = require_report(args)
    if report is None:
        return 2
    print()
    print_findings(report, args.all)
    print_table(summary_rows(report))
    print_outcome(report, args)
    return 0


def cmd_find_fake(args) -> int:
    """Separate analysis: is the file what its header claims?

    Three lies are looked for - a lossy source, upsampled hi-res and a bit
    depth padded with zeros. None is actionable: a file that was thrown away
    at 128 kbps cannot be repaired, only replaced by a better copy. That is
    why this stays out of the analyze/reencode/repair chain. It runs on the
    stored report when there is one, and on the headers alone when there
    is not.
    """
    report_file = args.report or default_report_path(args.folder)
    try:
        report = load_report(report_file)
    except ValueError:
        report = None

    if report:
        # Damaged files are included: analyze failed them on a decode error,
        # but the sampled window is usually far from the damage and a file
        # that lies about its quality is worth knowing about either way.
        # Only an unreadable header leaves nothing to measure.
        targets = [a for a in report.items if not a.unreadable]
        infos = [a.info for a in targets]
    else:
        print(t("run_scanning", root=short(args.folder)))
        paths = collect_flac_files(args.folder)
        if not paths:
            print(t("run_none"))
            return 0
        infos = [i for i in read_all_metadata(paths, args.jobs) if not i.error]
        targets = []

    if not infos:
        print(t("run_none"))
        return 0

    results = run_parallel(t("fake_running", count=len(infos)), infos,
                           check_fake_source, args.jobs, weight=_by_size,
                           cpu_bound=True)
    found = [r for r in results if r.suspicious]

    print()
    if found:
        print_fakes(found, args.folder, blank=False)
    else:
        print(t("fake_none"))

    if skipped := [r for r in results if r.error]:
        print_group(t("fake_skipped", count=files(len(skipped)),
                      reason=t(skipped[0].error)),
                    [(rel(r.path, args.folder), [])
                     for r in sorted(skipped, key=lambda r: r.path)],
                    blank=False)

    if report:
        for item, res in zip(targets, results):
            item.fake = res
        store(report, args)
    return 0


def run_write_command(args, report: Report, targets: Sequence, nothing: str,
                      intro: Callable, running: Callable, worker: Callable,
                      summary: Callable, min_saving: float = 0.0) -> int:
    """The skeleton both `reencode` and `repair` follow.

    Check the findings still hold, say what is about to happen, ask, do it in
    parallel, report, and write the report back. Only the targets, the wording
    and the summary differ - keeping the rest in one place is what guarantees
    the two commands treat a stale report or a refused confirmation alike.
    """
    if not targets:
        print(t(nothing))
        print_outcome(report, args)
        return 0

    live, skipped = live_targets(targets, report.root)
    for line in skipped:
        print(f"  {line}", file=sys.stderr)
    if skipped:
        print(t("rp_stale_count", count=len(skipped)), file=sys.stderr)
    if not live:
        return 1

    print()
    for line in intro(live):
        print(line)
    if not (args.dry_run or args.yes or ask_yes_no()):
        print(t("rc_cancelled"))
        return 1

    print()
    results = run_parallel(running(live), live, worker, args.jobs,
                           weight=_by_size)

    print()
    for res in sorted(results, key=lambda r: r.info.path):
        print_result(res, report.root, min_saving)

    by_path, changed = report.by_path(), 0
    for res in results:
        if res.status is Status.DONE:
            # Only salvaging rewrites the audio, so only there does the stored
            # spectral check stop describing the file.
            refresh_entry(by_path[res.info.path],
                          keep_fake=res.kind is not Kind.SALVAGE)
            changed += 1

    print_table(summary(results))

    advice = []
    if args.dry_run:
        advice.append(t("adv_was_dry"))
    if any(r.status is Status.FAILED for r in results):
        advice.append(t("adv_failed"))
    if changed:
        store(report, args)
        advice.append(t("adv_reanalyze"))
    print_advice(advice)
    return 0


def cmd_reencode(args) -> int:
    """Squeeze weakly encoded files. The audio is verified sample for sample,
    so there is nothing to lose and no backup to keep."""
    report = require_report(args)
    if report is None:
        return 2

    def intro(live: Sequence) -> list:
        total = human(sum(a.info.file_size for a in live))
        return [t("rc_confirm", count=files(len(live)), total=total),
                "  " + t("rc_settings", opts=" ".join(EFFORT_PRESETS[args.effort])),
                "  " + t("rc_promise")]

    def summary(results: Sequence) -> list:
        rows = []
        for status, key in ((Status.DRY_RUN, "sum_done"), (Status.DONE, "sum_done"),
                            (Status.NO_GAIN, "sum_skipped"),
                            (Status.FAILED, "sum_failed")):
            if group := [r for r in results if r.status is status]:
                rows.append((t(key), len(group)))
                if status in (Status.DRY_RUN, Status.DONE):
                    # A subset fix can cost space, so the total may be
                    # negative; say "grew by" rather than "saved -416 kB".
                    total = sum(r.saved for r in group)
                    rows.append((t("sum_saved" if total >= 0 else "sum_grew"),
                                 human(abs(total))))
        return rows

    return run_write_command(
        args, report,
        [a for a in report.items if not a.info.error and a.weak],
        "rc_nothing", intro,
        lambda live: t("rc_running_dry" if args.dry_run else "rc_running",
                       count=files(len(live))),
        lambda a: recompress_file(a.info, args.effort, args.min_saving,
                                  args.dry_run),
        summary, args.min_saving)


def cmd_repair(args) -> int:
    """Fix defects: playability, a lying header, damaged audio.

    Three findings, three repairs, and they differ in how much can be
    guaranteed - which is what decides whether the original is kept aside:

      outside the subset   Re-encoded in place. The audio is verified against
                           the original MD5, so nothing can be lost and no
                           backup is needed.
      bad header           24 bytes of STREAMINFO corrected, audio untouched.
      damaged audio        Salvaged by decoding past the errors. Losslessness
                           cannot be verified here, so the original is kept
                           as *.orig.flac.
    """
    report = require_report(args)
    if report is None:
        return 2

    groups = ((Kind.SUBSET, [a for a in report.items
                             if not a.info.error and a.subset_fixable]),
              (Kind.HEADER, [a for a in report.items if a.harmless]),
              (Kind.SALVAGE, [a for a in report.items if a.really_damaged]))
    kind = {id(a): name for name, items in groups for a in items}

    def intro(live: Sequence) -> list:
        counts = {name: sum(1 for a in live if kind[id(a)] is name)
                  for name, _ in groups}
        lines = [t("fix_confirm", count=files(len(live)))]
        for name, key in ((Kind.SUBSET, "fix_intro_subset"),
                          (Kind.HEADER, "fix_intro_header"),
                          (Kind.SALVAGE, "fix_intro_salvage")):
            if counts[name]:
                lines.append("  " + t(key, count=files(counts[name]),
                                      suffix=ORIG_SUFFIX))
        return lines

    def repair_one(item: Analysis) -> Outcome:
        if kind[id(item)] is Kind.SUBSET:
            # No saving threshold here: the point is playability, not space,
            # and returning into the subset may cost a fraction of a percent.
            return recompress_file(item.info, args.effort, None, args.dry_run,
                                   kind=Kind.SUBSET)
        if kind[id(item)] is Kind.HEADER:
            return patch_stream_header(item.info, args.dry_run)
        return repair_damaged_file(item.info, args.effort, args.dry_run)

    def summary(results: Sequence) -> list:
        settled = [r for r in results
                   if r.status in (Status.DONE, Status.DRY_RUN)]
        return count_rows((
            ("sum_subset_done", sum(1 for r in settled if r.kind is Kind.SUBSET)),
            ("sum_header_done", sum(1 for r in settled if r.kind is Kind.HEADER)),
            ("sum_repaired", sum(1 for r in settled if r.kind is Kind.SALVAGE)),
            ("sum_failed",
             sum(1 for r in results if r.status is Status.FAILED))))

    return run_write_command(
        args, report, [a for _, items in groups for a in items],
        "fix_nothing", intro,
        lambda live: t("fix_running", count=files(len(live))),
        repair_one, summary)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

COMMANDS = {"analyze": cmd_analyze, "show": cmd_show, "find-fake": cmd_find_fake,
            "reencode": cmd_reencode, "repair": cmd_repair}
NEEDS_FLAC = {"analyze", "find-fake", "reencode", "repair"}


def _add_common(parser: argparse.ArgumentParser, jobs: bool = True) -> None:
    parser.add_argument("folder", help=t("cli_folder"))
    if jobs:
        parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4,
                            help=t("cli_jobs"))
    parser.add_argument("--report", metavar="FILE",
                        help=t("cli_report",
                               path=short(os.path.dirname(default_report_path(".")))))
    parser.add_argument("--lang", choices=LANGUAGES, help=t("cli_lang"))


def _add_write_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--effort", choices=sorted(EFFORT_PRESETS),
                        default=DEFAULT_EFFORT, help=t("cli_effort"))
    parser.add_argument("--dry-run", action="store_true", help=t("cli_dry_run"))
    parser.add_argument("-y", "--yes", action="store_true", help=t("cli_yes"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=t("cli_description"), epilog=t("cli_epilog"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", metavar=t("cli_command"),
                                       required=True)

    analyze = subparsers.add_parser("analyze", help=t("cli_cmd_analyze"))
    _add_common(analyze)
    analyze.add_argument("--all", action="store_true", help=t("cli_all"))
    analyze.add_argument("--force", action="store_true", help=t("cli_force"))

    show = subparsers.add_parser("show", help=t("cli_cmd_show"))
    _add_common(show, jobs=False)
    show.add_argument("--all", action="store_true", help=t("cli_all"))
    show.set_defaults(jobs=os.cpu_count() or 4)

    find_fake = subparsers.add_parser("find-fake", help=t("cli_cmd_findfake"))
    _add_common(find_fake)

    reencode = subparsers.add_parser("reencode", help=t("cli_cmd_reencode"))
    _add_common(reencode)
    _add_write_options(reencode)
    reencode.add_argument("--min-saving", type=float, default=1.0, metavar="PCT",
                          help=t("cli_min_saving"))

    repair = subparsers.add_parser("repair",
                                   help=t("cli_cmd_repair", suffix=ORIG_SUFFIX))
    _add_common(repair)
    _add_write_options(repair)
    return parser


def preselect_language(argv: Sequence) -> None:
    """Set the language before argparse builds its help.

    The help text is assembled in build_parser(), so asking about --lang only
    after parse_args() would print `--help` in the wrong language.
    """
    lang = detect_language()
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            lang = argv[i + 1]
        elif arg.startswith("--lang="):
            lang = arg.split("=", 1)[1]
    set_language(lang)


def main() -> int:
    # Progress goes to stderr, the report to stdout; without this the two
    # arrive in the wrong order as soon as stdout is a pipe.
    sys.stdout.reconfigure(line_buffering=True)
    preselect_language(sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args()
    set_language(args.lang or detect_language())
    args.jobs = max(1, args.jobs)
    args.folder = os.path.abspath(os.path.expanduser(args.folder))

    if not os.path.isdir(args.folder):
        print(t("run_error", problem=t("run_bad_path", root=args.folder)),
              file=sys.stderr)
        return 2
    if problems := check_dependencies(args.command in NEEDS_FLAC):
        for problem in problems:
            print(t("run_error", problem=problem), file=sys.stderr)
        return 2

    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\n" + t("run_interrupted"), file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
