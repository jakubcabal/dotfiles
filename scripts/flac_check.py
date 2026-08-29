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
    flac_check.py inspect   FOLDER     separate check: does the file lie,
                                       and is its sample rate worth its size?
    flac_check.py reencode  FOLDER     rewrite weakly compressed files
                                       (--sample-rate/--bits also convert)
    flac_check.py repair    FOLDER     fix subset, headers, damaged audio
    flac_check.py revert-original FOLDER  put the *.orig.flac back in place
    flac_check.py drop-originals FOLDER   delete the *.orig.flac put aside

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

WHAT IS REALLY IN THE FILE (`inspect`)
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
part of the analyze/reencode/repair chain.

The same pass answers a second and quite different question: is the sample
rate carrying anything? A 192 kHz file whose music stops at 22 kHz is not
lying - the samples are honestly there - it is simply four times the size it
needs to be. Three shapes give ultrasound away as something other than music:
it climbs (nothing acoustic gains energy with frequency), it settles into a
flat floor, or it has faded out from under the audible band. Only then is a
lower rate named, and naming one changes nothing on disk.

Both sets of thresholds, and the files they were measured on, are in the two
comment blocks further down.

MD5 IN THE HEADER
STREAMINFO carries an MD5 of the decoded audio, and that is what makes a FLAC
verifiable at all: `flac -t` compares it, and so does this script after every
write. Some encoders leave it at zero, and then nothing can ever prove the
audio survived - `flac -t` only says "cannot check MD5 signature" and exits 0
regardless. Such files are listed, and re-encoding one writes a proper MD5.

FLAC SUBSET
The subset is what hardware players restrict themselves to. A file outside it
is still valid FLAC, but a set-top box or car radio may refuse it. The limits
are looser above 48 kHz, see the SUBSET_ constants. What counts are the ACTUAL
values in the stream, not the encoder settings.

Two kinds of violation, only one of which re-encoding can fix:
  block size, LPC order, partition order   encoder parameters -> `repair`
                                           rewrites the file losslessly.
  bit depth, sample rate                   properties of the audio itself.
                                           Changing them means resampling or
                                           dithering, i.e. losing audio, so
                                           `repair` never does it - only
                                           `reencode --sample-rate/--bits`,
                                           when explicitly asked. 24 bit and
                                           96 kHz are inside the subset, so
                                           ordinary hi-res is never affected.

OVERWRITE SAFETY
Encoding goes to a temporary file in the same directory. The original is
replaced atomically (os.replace) only after the new file passes both an MD5
check of the decoded audio and `flac -t`. On any problem the temporary file
is removed and the original is left untouched.

`reencode --sample-rate/--bits` is not a lossless write: the audio is
deliberately a different one afterwards, so the original MD5 cannot match.
What is verified instead is that the file really has the requested format, is
as long as the change of rate implies, and that its header MD5 matches the
audio it decodes to - which is what proves the encoder wrote down exactly what
the resampler produced.

`repair` handles three defects. Returning a file into the subset and
correcting a lying header leave the audio alone, so the file is simply
replaced. Salvaging damaged audio does not: part of it is already gone, the
MD5 will never match again, and tags and cover art have to be carried over
into the replacement by hand.

WHAT BECOMES OF THE ORIGINALS
One rule across the commands: a write that changes the audio - converting and
salvaging - puts the original aside as <name>.orig.flac first, and one that
does not need to, because it is verified bit for bit, simply replaces the
file. `--no-keep-original` turns the first half off on both commands, and the
confirmation says which of the two is about to happen. An original already put
aside is never overwritten, so a second run cannot bury the real one. When the
new files have proved themselves, `drop-originals` clears the originals out;
when they have not, `revert-original` moves them back over the replacements
and the folder is as it was.

Requires Python 3.8+ and the `flac` tool 1.3+ in PATH (1.4+ for 32 bit
output). Converting with --sample-rate/--bits also needs `ffmpeg` built with
libsoxr. No PyPI packages.
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
import tempfile
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from dataclasses import dataclass, field, fields, is_dataclass
from typing import BinaryIO, Callable, Sequence

MIN_PYTHON = (3, 8)
MIN_FLAC = (1, 3, 0)
#: Bumped to 4 when the spectral measurement changed: the cliff ceiling became
#: relative to the file's Nyquist and the ultrasound bands became swept cells
#: rather than single probes. To 5 when the bandwidth reading gained the step
#: (see BW_STEP_DB), which reads the same bands to a different answer. Stored
#: inspections were measured and judged by the old ones, and `inspect` reuses
#: them for any file that has not moved a byte, so without this a library
#: would answer half in the new numbers and half in the old, with nothing on
#: screen saying which was which.
REPORT_VERSION = 5

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
dep_python_old     | Python {have} je starý, potřeba {want}+         | Python {have} is too old, need {want}+
dep_flac_missing   | nástroj 'flac' není v PATH                      | the 'flac' tool is not in PATH
dep_flac_old       | flac {have} je starý, potřeba {want}+           | flac {have} is too old, need {want}+
dep_ffmpeg_missing | nástroj 'ffmpeg' není v PATH (nutný pro převod) | the 'ffmpeg' tool is not in PATH (needed for the conversion)
dep_ffmpeg_soxr    | ffmpeg neumí resampler soxr (chybí libsoxr)     | this ffmpeg has no soxr resampler (built without libsoxr)

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
err_resample         | ffmpeg skončil s kódem {code}: {detail}  | ffmpeg exited with {code}: {detail}
err_md5_decode       | dekódování pro MD5 selhalo               | decoding for the MD5 failed

# --- losslessness check -----------------------------------------------
err_mismatch    | nesouhlasí {field}: {old} -> {new}            | {field} does not match: {old} -> {new}
err_md5_unset   | nový soubor nemá MD5, nelze ověřit            | the new file carries no MD5, cannot verify
err_md5_differs | MD5 se liší - překódování NEBYLO bezeztrátové | MD5 differs - the re-encode was NOT lossless
err_flac_t      | flac -t neprošel: {detail}                    | flac -t failed: {detail}

# --- conversion check --------------------------------------------------
err_samples     | převod vrátil {new} vzorků místo {old}               | the conversion returned {new} samples instead of {old}
err_md5_content | MD5 v hlavičce nesedí na zvuk v souboru              | the MD5 in the header does not match the audio in the file
err_level       | hlasitost se posunula o {drift} dB, to není originál | the loudness moved by {drift} dB, that is not the source

# --- severity and labels ----------------------------------------------
sev_ok   | v pořádku                    | fine
sev_warn | Slabší komprese              | Weaker compression
sev_bad  | Velmi slabá komprese (-0/-2) | Very weak compression (-0/-2)

# --- why a file was flagged -------------------------------------------
why_blocksize_low | blok {size} při {rate} Hz (nízké úrovně mají {expected})         | block size {size} at {rate} Hz (low levels use {expected})
why_blocksize_odd | nestandardní blok {size} při {rate} Hz                           | unusual block size {size} at {rate} Hz
why_no_lpc        | žádná LPC predikce (enkodér běžel s -l 0)                        | no LPC prediction at all (encoder ran with -l 0)
why_low_lpc       | jen {pct:.0f} % subrámců používá LPC                             | only {pct:.0f} % of subframes use LPC
why_low_order     | nejvyšší LPC řád jen {order} (odpovídá -3, ale závisí na obsahu) | highest LPC order is only {order} (suggests -3, but depends on the material)
why_no_stereo     | žádná stereo dekorelace (-0 nebo -3)                             | no stereo decorrelation (-0 or -3)

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
dmg_header           | 💥 POŠKOZENÉ SOUBORY ({count}) - dekodér na nich selhal:                                                                                                         | 💥 DAMAGED FILES ({count}) - the decoder failed on them:
dmg_locating         | Zjišťuji rozsah poškození ({count})                                                                                                                             | Locating the damage ({count})
dmg_at_start         | {time} = {pct:.1f} % stopy, na začátku                                                                                                                          | {time} = {pct:.1f} % in, near the start
dmg_at_end           | {time} = {pct:.1f} % stopy, na konci                                                                                                                            | {time} = {pct:.1f} % in, near the end
dmg_at_middle        | {time} = {pct:.1f} % stopy, uprostřed                                                                                                                           | {time} = {pct:.1f} % in, in the middle
dmg_error_at         | ✗ chyba v datech na {where}                                                                                                                                     | ✗ data error at {where}
dmg_truncated        | ✂ useknuto na {where} - chybí {samples} vzorků ({seconds:.2f} s)                                                                                                | ✂ truncated at {where} - {samples} samples missing ({seconds:.2f} s)
dmg_encoder_bug      | → enkodér: {vendor}                                                                                                                                             | → encoder: {vendor}
dmg_harmless_header  | 🔧 VADNÁ HLAVIČKA ({count}) - enkodér zahodil poslední neúplný rámec,\n   ale do hlavičky zapsal plný počet vzorků. Zvuk je celý, chybu\n   hlásí jen `flac -t`. | 🔧 BAD HEADER ({count}) - the encoder dropped the last partial frame\n   but wrote the full sample count into the header. No audio is\n   missing, only `flac -t` complains.
dmg_unknown_position | (přesnou pozici se nepodařilo určit)                                                                                                                            | (the exact position could not be determined)
unreadable_header    | Nepodařilo se přečíst:                                                                                                                                          | Could not be read:
nomd5_header         | 🔓 BEZ MD5 V HLAVIČCE ({count}) - zvuk proti ní nelze ověřit, překódování\n   MD5 doplní:                                                                        | 🔓 NO MD5 IN THE HEADER ({count}) - the audio cannot be verified against\n   it, re-encoding writes one:

# --- repair -----------------------------------------------------------
fix_nothing       | Report nehlásí žádný poškozený soubor.                     | The report lists no damaged files.
fix_confirm       | Chystám se opravit {count}:                                | About to repair {count}:
fix_intro_subset  | {count} mimo subset - překóduji na místě, ověřím proti MD5 | {count} outside the subset - re-encoded in place, MD5 verified
fix_intro_header  | {count} s vadnou hlavičkou - opravím jen ji, zvuk zůstane  | {count} with a bad header - only it is corrected, audio stays
fix_intro_salvage | {count} s poškozeným zvukem - zachráním, co půjde přečíst  | {count} damaged - whatever is readable is salvaged
fix_header_done   | hlavička opravena, {old} → {new} vzorků, zvuk nezměněn     | header corrected, {old} → {new} samples, audio untouched
fix_running       | Zachraňuji ({count})                                       | Salvaging ({count})
fix_recovered     | zachráněno {pct:.2f} %{lost}                               | recovered {pct:.2f} %{lost}
fix_lost          | , ztraceno {seconds:.2f} s                                 | , {seconds:.2f} s lost
fix_no_loss       | , beze ztráty                                              | , nothing lost
fix_backed_up     | originál → {name}                                          | original → {name}
orig_kept         | originál se odkládá jako *{suffix}                         | the original is put aside as *{suffix}
orig_drop         | originál se nikam neodkládá, přepíše se rovnou             | no copy of the original is kept, it is overwritten
orig_exists       | {name} tu už je a ten se nepřepisuje                       | {name} is already there and is never overwritten
fix_dry           | nanečisto, nic se nezapsalo                                | dry run, nothing written
fix_failed        | záchrana selhala: {error}                                  | salvage failed: {error}
fix_no_meta       | tagy se přenést nepodařilo                                 | tags could not be carried over

# --- reencode ---------------------------------------------------------
rc_nothing      | Report nehlásí nic k překódování.                                    | The report lists nothing to re-encode.
rc_confirm      | Chystám se PŘEPSAT {count} na místě ({total}).                       | About to OVERWRITE {count} in place ({total}).
rc_all_any      | všechny soubory z reportu, přepíší se všechny bez ohledu na úsporu   | every file in the report, all of them rewritten whatever the saving
rc_all_skipped  | {count} s vadou vynecháno, na ty je repair                           | {count} with a defect left out, repair is the command for those
rc_all_new      | {count} přibylo od poslední analýzy                                  | {count} appeared since the last analyze
rc_conv_to      | převod na {format}, přepočítá soxr                                   | conversion to {format}, recomputed by soxr
rc_conv_dither  | při 16 bitech se navíc přidá dither (shibata)                        | at 16 bit a dither is added on top (shibata)
rc_conv_num     | {count} se převede, zbytek se jen překóduje                          | {count} will be converted, the rest are only re-encoded
rc_warn_up      | {count} k převzorkování NAHORU - mezi vzorky není co doplnit         | {count} to be resampled UP - there is nothing to put between the samples
rc_warn_family  | {count} přes hranici rodin 44,1 a 48 kHz - zbytečný přepočet         | {count} crossing between the 44.1 and 48 kHz families - a recompute for nothing
rc_warn_bits    | {count} k navýšení hloubky - přibudou jen prázdné bity               | {count} to be given more bits - only empty bits are added
rc_down_only    | jen dolů: co by se mělo zvednout, zůstane ve svém formátu            | down only: whatever would be raised is left in its own format
rc_down_left    | {count} beze změny formátu, šlo by jen nahoru                        | {count} kept in their own format, the target is only above them
rc_down_alone   | --downgrade sám neurčuje nic, přidej --sample-rate nebo --bits       | --downgrade decides nothing on its own, add --sample-rate or --bits
rc_conv_warn    | Zvuk se PŘEPOČÍTÁ, není to bezeztrátové.                             | The audio is RECOMPUTED, this is not lossless.
rc_conv_meta    | Tagy i obal zůstanou, cuesheet ne: odkazuje na vzorky, a ty se mění. | Tags and cover art are kept, the cuesheet is not: it indexes samples, and those change.
rc_conv_done    | {oldfmt} → {newfmt}, {old} → {new} ({change})                        | {oldfmt} → {newfmt}, {old} → {new} ({change})
rc_conv_would   | převedl by se na {newfmt} ({change}), soubor nezměněn                | would be converted to {newfmt} ({change}), file unchanged
rc_conv_rate    | {khz:g} kHz (hloubka beze změny)                                     | {khz:g} kHz (depth left as it is)
rc_conv_bits    | {bits} bit (frekvence beze změny)                                    | {bits} bit (rate left as it is)
rc_conv_bad     | {rate} Hz se do FLACu nezapíše, zvol běžnou frekvenci                | FLAC cannot store {rate} Hz, pick a common rate
rc_settings     | nastavení: flac {opts}                                               | settings: flac {opts}
rc_promise      | Zvuk zůstane bit po bitu stejný (ověřuje se MD5), tagy a obal také.  | The audio stays bit-for-bit identical (MD5 checked), so do tags and cover art.
rc_not_tty      | Vstup není terminál - pro neinteraktivní běh použij --yes.           | Input is not a terminal - use --yes for non-interactive runs.
rc_prompt       | Pokračovat? [ano/Ne]:                                                | Continue? [yes/No]:
rc_cancelled    | Zrušeno, nic se nezměnilo.                                           | Cancelled, nothing changed.
rc_running      | Překódovávám ({count})                                               | Re-encoding ({count})
rc_running_dry  | Zkouším nanečisto ({count})                                          | Dry run over ({count})
rc_replaced     | {old} → {new} (ušetřeno {saved}, {pct:.1f} %)                        | {old} → {new} (saved {saved}, {pct:.1f} %)
rc_would        | ušetřilo by se {saved} ({pct:.1f} %), soubor nezměněn                | would save {saved} ({pct:.1f} %), file unchanged
rc_nogain_done  | {old} → {new} (nic se neušetřilo, +{pct:.2f} %)                      | {old} → {new} (nothing saved, +{pct:.2f} %)
rc_nogain_dry   | neušetřilo by se nic (+{pct:.2f} %), přepsal by se přesto            | nothing would be saved (+{pct:.2f} %), it would be rewritten anyway
rc_would_subset | vrátil by se do subsetu ({change}), soubor nezměněn                  | would return into the subset ({change}), file unchanged
rc_subset_done  | v subsetu, {old} → {new} ({change})                                  | now in subset, {old} → {new} ({change})
rc_grew         | naroste o {pct:.2f} %                                                | grows by {pct:.2f} %
rc_saved_pct    | ušetřeno {pct:.2f} %                                                 | saved {pct:.2f} %
rc_failed       | {error} - originál ponechán                                          | {error} - original kept

# --- report file ------------------------------------------------------
rp_saved       | Report uložen: {path}                                          | Report saved: {path}
rp_missing     | Report pro '{root}' neexistuje - spusť nejdřív:\n  {cmd}       | No report for '{root}' - run this first:\n  {cmd}
rp_broken      | Report {path} nejde přečíst ({error}), spusť analyze znovu.    | Report {path} cannot be read ({error}), run analyze again.
rp_from        | Report z {when} ({count})                                      | Report from {when} ({count})
rp_stale_file  | {name}: od analýzy se změnil, přeskakuji                       | {name}: changed since the analysis, skipping
rp_gone_file   | {name}: už neexistuje, přeskakuji                              | {name}: no longer exists, skipping
rp_stale_count | Přeskočeno {count} (změněno od analýzy) - spusť analyze znovu. | Skipped {count} (changed since the analysis) - run analyze again.
rp_reused      | beze změny od minule: {count}, znovu analyzuji {fresh}         | unchanged since last time: {count}, re-analysing {fresh}

# --- dropping the originals put aside ---------------------------------
drop_none     | Žádné odložené originály (*{suffix}) tu nejsou.                           | There are no originals put aside (*{suffix}) here.
drop_confirm  | Chystám se SMAZAT {count} ({total}).                                      | About to DELETE {count} ({total}).
drop_what     | odložené originály *{suffix}, jejichž náhrada je na místě                 | originals put aside as *{suffix}, whose replacement is in place
drop_final    | Zpátky už se z nich nic nevezme.                                          | Nothing can be taken back from them afterwards.
drop_orphan   | Bez náhrady, proto ponecháno ({count}) - je to poslední kopie toho zvuku: | Kept because nothing replaced them ({count}) - they are the last copy of that audio:
drop_failed   | {name}: {error}                                                           | {name}: {error}
drop_failed_h | Nepodařilo se smazat ({count}):                                           | Could not be deleted ({count}):
drop_done     | Smazáno                                                                   | Deleted
drop_would    | Smazalo by se                                                             | Would be deleted
drop_freed    | Uvolněno                                                                  | Freed

# --- putting the originals back ---------------------------------------
rev_none     | Žádné odložené originály (*{suffix}) tu nejsou, není co vracet. | There are no originals put aside (*{suffix}) here, nothing to revert.
rev_confirm  | Chystám se VRÁTIT {count}, na disku {delta}.                    | About to REVERT {count}, {delta} on disk.
rev_what     | originál *{suffix} se přesune zpět na své místo                 | the original *{suffix} moves back to its place
rev_final    | Současná verze se zahodí, ta už se nevrátí.                     | The current version is thrown away and does not come back.
rev_change   | {old} → {new}                                                   | {old} → {new}
rev_gone     | na místě nic není, originál jen dostane zpět své jméno          | nothing is in its place, the original only gets its name back
rev_broken_h | Nečitelný originál, proto ponecháno ({count}):                  | Kept because the original cannot be read ({count}):
rev_error    | {name}: {error}                                                 | {name}: {error}
rev_failed_h | Nepodařilo se vrátit ({count}):                                 | Could not be reverted ({count}):
rev_done     | Vráceno                                                         | Reverted
rev_would    | Vrátilo by se                                                   | Would be reverted

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
sum_total       | FLAC souborů celkem             | FLAC files in total
sum_damaged     | POŠKOZENÉ                       | DAMAGED
sum_harmless    | Vadná hlavička (zvuk je celý)   | Bad header (audio is complete)
sum_unreadable  | Nečitelná hlavička              | Unreadable header
sum_no_md5      | Bez MD5 v hlavičce              | No MD5 in the header
sum_subset_fix  | Mimo subset (opravitelné)       | Outside subset (fixable)
sum_subset_keep | Mimo subset (neopravitelné)     | Outside subset (not fixable)
sum_fake        | Nesedí deklarovaná kvalita      | Quality is not what it claims
sum_waste       | Frekvence vyšší, než hudba nese | Sample rate higher than the music
sum_grew        | Narostlo o                      | Grew by
sum_done        | Překódováno                     | Re-encoded
sum_converted   | Převedeno                       | Converted
sum_saved       | Ušetřeno                        | Saved
sum_failed      | Selhalo (originál zachován)     | Failed (original kept)
sum_repaired    | Zachráněno                      | Salvaged
sum_subset_done | Vráceno do subsetu              | Returned into the subset
sum_header_done | Opravena hlavička               | Header corrected

# --- next step --------------------------------------------------------
adv_next        | Další krok:                                                                           | Next step:
adv_reencode    | {cmd}  ({count}, ušetří místo, {eta})                                                 | {cmd}  ({count}, saves space, {eta})
adv_repair      | {cmd}  ({count} s vadou)                                                              | {cmd}  ({count} with a defect)
adv_inspect     | {cmd}  (jestli soubory nelžou a jestli je frekvence k něčemu, {eta})                  | {cmd}  (whether the files lie, and whether their sample rate earns its size, {eta})
adv_clean       | Vše v pořádku, nic k opravě.                                                          | All clear, nothing to fix.
adv_fake        | {count} má horší kvalitu, než tvrdí; překódování nepomůže, jen lepší kopie ze zdroje. | {count} are worse than they claim; re-encoding will not help, only a better copy from the source.
adv_was_dry     | Bylo to nanečisto. Spusť totéž bez --dry-run.                                         | That was a dry run. Repeat it without --dry-run.
adv_failed      | U selhaných zůstaly originály; zkontroluj práva a volné místo.                        | Originals of the failed files are untouched; check permissions and free space.
adv_subset_keep | {count} mimo subset nelze opravit beze ztráty, viz výš.                               | {count} outside the subset cannot be fixed losslessly, see above.
adv_reanalyze   | Report je aktualizovaný; pro plný obraz spusť analyze znovu.                          | The report is updated; run analyze again for the full picture.
adv_stale       | {cmd}  ({count} z reportu chybí na disku)                                             | {cmd}  ({count} of the report are no longer on disk)
adv_partial     | {cmd}  ({count} zatím bez hloubkové analýzy)                                          | {cmd}  ({count} not deeply analysed yet)
adv_no_md5      | {cmd}  ({count} bez MD5, překódování ho doplní)                                       | {cmd}  ({count} without an MD5, re-encoding writes one)
adv_reverted    | {cmd}  (vrácené soubory už report nepopisuje)                                         | {cmd}  (the report no longer describes the reverted files)

# --- files that lie about their quality --------------------------------
fake_unusable  | stopa je příliš krátká nebo tichá                                                                                | the track is too short or too quiet
fake_running   | Prohlížím spektrum ({count})                                                                                     | Inspecting the spectrum ({count})
fake_header    | 🎭 KVALITA NEODPOVÍDÁ DEKLARACI ({count}):                                                                        | 🎭 QUALITY IS NOT WHAT IS CLAIMED ({count}):
fake_lossy     | ztrátový zdroj: ostrý ořez na {khz:.1f} kHz, sráz {db:.0f} dB - odpovídá {hint}                                  | lossy source: sharp cutoff at {khz:.1f} kHz, {db:.0f} dB cliff - consistent with {hint}
fake_upsampled | falešné hi-res: {claimed} kHz převzorkováno z {source} kHz, nad {edge:.1f} kHz je ticho ({db:.0f} dB pod hudbou) | fake hi-res: {claimed} kHz upsampled from {source} kHz, silence above {edge:.1f} kHz ({db:.0f} dB below the music)
fake_padded    | falešná hloubka: hlavička říká {claimed} bitů, vzorky využívají {real}                                           | padded depth: the header says {claimed} bits, the samples use {real}
fake_caveat    | Ořez může mít i nahrávka z analogového pásu; ověř spektrogramem. AAC nad ~192 kbps se takhle chytit nedá.        | An analogue tape source can be cut off too; check a spectrogram. AAC above ~192 kbps cannot be caught this way.
fake_none      | Žádný soubor nelže o své kvalitě.                                                                                | No file lies about its quality.
fake_ok_header | ✅ Kvalita odpovídá deklaraci ({count}):                                                                          | ✅ Quality matches the claim ({count}):
fake_ok_cutoff | bez ořezu spektra                                                                                                | no cutoff in the spectrum
fake_ok_ultra  | ultrazvuk jen {db:.0f} dB pod hudbou                                                                             | ultrasound only {db:.0f} dB below the music
fake_ok_bits   | vzorky využívají všech {bits} bitů                                                                               | the samples use all {bits} bits

# --- a sample rate carrying less than it costs -------------------------
bw_header    | 📐 VYŠŠÍ FREKVENCE, NEŽ HUDBA POTŘEBUJE ({count}):                                           | 📐 A HIGHER SAMPLE RATE THAN THE MUSIC NEEDS ({count}):
bw_floor     | hudba končí na {khz} kHz, nad tím ploché šumové dno {db} dB pod ní - stačilo by {rate}      | the music ends at {khz} kHz, above it a flat noise floor {db} dB below it - {rate} would hold it
bw_shaping   | hudba končí na {khz} kHz, nad tím tvarovaný šum ({db} dB na {peak} kHz) - stačilo by {rate} | the music ends at {khz} kHz, above it shaped noise ({db} dB at {peak} kHz) - {rate} would hold it
bw_step      | hudba končí na {khz} kHz, nad tím jen zbytky {db} dB pod ní - stačilo by {rate}             | the music ends at {khz} kHz, above it only remnants {db} dB below it - {rate} would hold it
bw_quiet     | hudba slábne do {khz} kHz, výš je přes {db} dB pod ní - stačilo by {rate}                   | the music fades out by {khz} kHz, above that it is over {db} dB below it - {rate} would hold it
bw_caveat    | Nic to neopravuje, jen šetří místo, a je to nevratný přepočet - originál si nech.           | This fixes nothing, it only saves space, and it is an irreversible recompute - keep the original.
bw_none      | Žádný soubor nenese míň, než na kolik je nastavený.                                         | No file carries less than its sample rate is set for.
bw_ok        | spektrum sahá až k {khz} kHz                                                                | the spectrum runs right up to {khz} kHz
fake_skipped | Nešlo změřit ({count}): {reason}                                                            | Could not be measured ({count}): {reason}

# --- command line help ------------------------------------------------
cli_description  | Kontrola FLAC knihovny: komprese, subset, poškození.                                                     | Checks a FLAC library: compression, subset, damage.
cli_epilog       | Nejdřív analyze, pak podle nálezu reencode nebo repair.                                                  | Run analyze first, then reencode or repair as needed.
cli_command      | příkaz                                                                                                   | command
cli_folder       | složka s hudbou (rekurzivně)                                                                             | music folder (recursive)
cli_lang         | jazyk výstupu (výchozí: podle prostředí)                                                                 | output language (default: from the environment)
cli_report       | kam uložit report (výchozí: {path})                                                                      | where to keep the report (default: {path})
cli_all          | vypsat i soubory, které jsou v pořádku                                                                   | list the files that are fine as well
cli_all_reencode | překódovat všechny soubory, ne jen slabě komprimované                                                    | re-encode every file, not just the weakly compressed ones
cli_dry_run      | nic nezapisovat, jen spočítat výsledek                                                                   | write nothing, only compute the outcome
cli_yes          | neptat se na potvrzení                                                                                   | do not ask for confirmation
cli_sample_rate  | cílová vzorkovací frekvence v Hz (např. 48000), jinak beze změny                                         | target sample rate in Hz (e.g. 48000), otherwise left as it is
cli_bits         | cílová bitová hloubka (16, 24, 32), jinak beze změny                                                     | target bit depth (16, 24, 32), otherwise left as it is
cli_downgrade    | převádět jen dolů, soubory pod cílovou hodnotou nechat být                                               | convert downwards only, leave the files already below the target alone
cli_no_keep_orig | neodkládat originál jako *{suffix} ani tam, kde se mění zvuk                                             | do not put the original aside as *{suffix}, not even where the audio changes
cli_force        | analyzovat vše znovu, i beze změny od minule                                                             | re-analyse everything, even what has not changed
cli_cmd_analyze  | projít složku a uložit report (dekóduje, pomalé)                                                         | scan the folder and store a report (decodes, slow)
cli_cmd_show     | znovu vypsat uložený report                                                                              | print the stored report again
cli_cmd_inspect  | prohlédnout spektrum: zdroj z MP3/AAC, falešné hi-res i hloubka, a jestli je frekvence k něčemu (pomalé) | inspect the spectrum: MP3/AAC source, fake hi-res or depth, and whether the sample rate earns its size (slow)
cli_cmd_reencode | překódovat slabě komprimované na místě (bezeztrátově, tagy zůstanou)                                     | re-encode weakly compressed files in place (lossless, tags kept)
cli_cmd_repair   | opravit vady: subset, hlavička, poškozený zvuk (originál → *{suffix})                                    | fix defects: subset, header, damaged audio (original → *{suffix})
cli_cmd_revert   | vrátit soubory z odložených originálů *{suffix} (současná verze se zahodí)                               | restore the files from the originals put aside as *{suffix} (the current version is thrown away)
cli_cmd_drop     | smazat odložené originály *{suffix}, které už mají náhradu                                               | delete the originals put aside as *{suffix} once their replacement is in place
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

# Encoder settings, the same for every write this script makes. -8 stays
# inside the subset, so the result plays on hardware; deliberately no -l 32 /
# -r 15 and no --lax. Nothing above it is worth offering: -e -p measurably
# gains 0.044 to 0.059 % for about 16x the time (20 MB track: ~10 kB for
# +13 s), which is not a trade anyone would take twice.
ENCODE_OPTS = ["-8"]

# Changing the sample rate or the bit depth is the one thing flac cannot do,
# so ffmpeg does that part and hands the raw samples over - the file itself is
# still written by flac, with the same preset as every other file this tool
# writes. Raw and not WAV: a pipe has no length to put in a WAV header, and
# every parameter is known here anyway.
#: Bit depth -> the raw sample format both sides speak.
RAW_FORMATS = {16: "s16le", 24: "s24le", 32: "s32le"}
#: flac 1.4 was the first that could encode 32 bit.
MIN_FLAC_32 = (1, 4, 0)

#: How far the loudness of a conversion may drift from its source, in dB,
#: before the result is thrown away. Resampling moves it by a fraction of a
#: dB; the ffmpeg dither fault this guard was written for moved it by 26.
MAX_LEVEL_DRIFT = 3.0
#: Seconds of audio the loudness check listens to at each end of the pipe.
LEVEL_WINDOW = 60

# Measured throughput per thread (MB/s), for rough run-time estimates only.
# On 12 cores: deep 832 MB in 2.6 s, encode 832 MB in 5.9 s. The spectral pass
# measured 5.2 MB/s per thread on 96 kHz material and 10.3 on 44.1; the lower
# figure is the one worth promising.
DEEP_RATE, ENCODE_RATE, FAKE_RATE = 25, 12, 5

#: How many files are worked on at once. Every stage here spends its time in
#: `flac` or `ffmpeg`, one process per file, so the core count is the answer
#: and there was never a second one worth offering.
JOBS = max(1, os.cpu_count() or 4)

#: The stereo search the encoder evidently used, named the way flac names it.
#: Technical tokens, so they need no translation.
STEREO_NONE, STEREO_ADAPTIVE, STEREO_EXHAUSTIVE = "INDEPENDENT", "-M", "-m"
STEREO_UNKNOWN = "?"

MD5_UNSET = b"\x00" * 16       # an MD5 the encoder never filled in
CHUNK = 1 << 16

#: Suffix for the untouched original kept aside by `repair`, and by
#: `reencode --keep-original`.
ORIG_SUFFIX = ".orig.flac"


def resample_filter(bits: int) -> str:
    """The ffmpeg filter that does the resampling and the requantising.

    soxr at 28 bit precision is the best resampler ffmpeg has. Dither is asked
    for at 16 bit and nowhere else, for two independent reasons.

    It is the only depth where dither does anything. Above it the word length
    is cut down by ffmpeg's raw writer, which truncates (measured: an
    arithmetic >>8), and that error sits at -144 dBFS - some 25 dB under the
    noise floor of the best converter ever built, which means the recording's
    own noise already dithers the 24th bit.

    And it is the only depth where ffmpeg can be trusted with it. `osf` has no
    s24 - ffmpeg has no 24 bit sample format at all - so 24 and 32 both have
    to ask for packed s32, and over packed s32 several of the noise shaping
    filters go unstable: shibata, low_shibata and f_weighted all turned a
    -24 dBFS tone into 0 dBFS clipping here, while lipshitz and high_shibata
    happened not to. Not a line to walk. Above 16 bit the samples are only cut
    down to the raw format, with no dither_method anywhere near them.
    """
    if bits == 16:
        return ("aresample=resampler=soxr:precision=28"
                ":osf=s16:dither_method=shibata")
    return "aresample=resampler=soxr:precision=28:osf=s32"


def converting(args) -> bool:
    """Did the user ask for a different rate or depth? That one question
    decides everything: which files are targets, which tool writes them,
    whether ffmpeg is needed and what the confirmation may promise."""
    return bool(getattr(args, "sample_rate", None)
                or getattr(args, "bits", None))


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


def ffmpeg_can_resample() -> str | None:
    """The message to print if ffmpeg cannot do the conversion, else None.

    Answered by doing it: a hundredth of a second of silence through the real
    filter chain settles both whether ffmpeg is there and whether it was built
    with libsoxr, which no version string says out loud.
    """
    try:
        probe = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
             "-i", "anullsrc=r=44100:cl=stereo", "-t", "0.01",
             "-af", resample_filter(16), "-ar", "48000", "-f", "s16le", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return t("dep_ffmpeg_missing")
    return None if probe.returncode == 0 else t("dep_ffmpeg_soxr")


def check_dependencies(args) -> list:
    """Missing or too old dependencies for the command about to run."""
    problems = []
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append(t("dep_python_old", have=_dotted(sys.version_info[:3]),
                          want=_dotted(MIN_PYTHON)))
    if args.command in NEEDS_FLAC:
        # 32 bit output is the only thing here that needs a newer flac than
        # the script otherwise gets by with.
        want = MIN_FLAC_32 if getattr(args, "bits", None) == 32 else MIN_FLAC
        version = flac_version()
        if version is None:
            problems.append(t("dep_flac_missing"))
        elif version and version < want:
            problems.append(t("dep_flac_old", have=_dotted(version),
                              want=_dotted(want)))
    if converting(args) and (problem := ffmpeg_can_resample()):
        problems.append(problem)
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
    CONVERT = "convert"          # resampled or requantised, so the audio moved
    SUBSET = "subset"            # re-encoded to get back inside the subset
    HEADER = "header"            # 24 bytes of STREAMINFO corrected
    SALVAGE = "salvage"          # decoded past the damage, so the audio moved


class Status(str, enum.Enum):
    """How the work on one file turned out."""

    DONE = "done"                # the file on disk was replaced
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
    rate: int = 0                # what a conversion turned the stream into,
    bits: int = 0                # both zero for every other kind of write
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


def set_aside(path: str, keep: bool) -> str:
    """The name the original of `path` is put aside under, "" when the caller
    asked for it not to be.

    Every write that changes the audio goes through here, so the rule is the
    same wherever it happens: an original already put aside is never
    overwritten, otherwise a second run would replace the true original with
    an already rewritten one.
    """
    if not keep:
        return ""
    backup = os.path.splitext(path)[0] + ORIG_SUFFIX
    if os.path.exists(backup):
        raise RuntimeError(t("orig_exists", name=os.path.basename(backup)))
    return backup


def put_in_place(path: str, tmp: str, backup: str = "") -> None:
    """Move the finished temporary file over the original, atomically.

    With a `backup` the original is renamed aside first and put back if the
    swap then fails, so there is no moment in which neither of them is there.
    """
    # flac keeps the modification time itself (--preserve-modtime is its
    # default) but carries over neither mode nor owner.
    _copy_owner_and_times(path, tmp)
    if not backup:
        os.replace(tmp, path)           # atomic, the original never disappears
        return
    os.rename(path, backup)             # same directory, so atomic
    try:
        os.replace(tmp, path)
    except OSError:
        os.rename(backup, path)         # put the original back
        raise


def recompress_file(info: FlacInfo, dry_run: bool,
                    kind: Kind = Kind.REENCODE) -> Outcome:
    """Re-encode one file in place.

    The original is replaced whenever the new file passes the losslessness
    check, whatever that did to the size: a file is judged on its own, and
    the few that come out a fraction of a percent larger are not worth a knob
    to hold them back. Errors are returned as Status.FAILED rather than
    raised - one broken file must not stop a whole library.
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
                              *ENCODE_OPTS, "-o", tmp, "--", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if enc.returncode != 0:
            detail = _last_line(enc.stderr.decode("utf-8", "replace"))
            raise RuntimeError(t("err_encode_failed", code=enc.returncode,
                                 detail=detail))

        assert_lossless(info, tmp)
        new_size = os.path.getsize(tmp)
        if dry_run:
            os.remove(tmp)
            return Outcome(info, kind, Status.DRY_RUN, old_size, new_size)

        put_in_place(path, tmp)     # verified lossless, so nothing to keep
        return Outcome(info, kind, Status.DONE, old_size, new_size)

    except (OSError, RuntimeError, ValueError) as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return Outcome(info, kind, Status.FAILED, old_size, error=str(e))


def audio_levels(path: str) -> tuple | None:
    """Peak and RMS of the start of a file, in dBFS, or None if unreadable.

    A window and not the whole file, because this is a sanity check and not a
    measurement: anything that wrecks a stream wrecks its first minute too,
    and a full pass would cost about as much as the conversion itself. Digital
    silence reports -inf, which no number can be compared against, so that
    reads as None too and the check simply does not apply.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-vn",
             "-t", str(LEVEL_WINDOW), "-i", path,
             "-af", "astats=measure_perchannel=none", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stderr.decode("utf-8", "replace")
    found = [re.search(rf"{name} level dB:\s*(-?\d+(?:\.\d+)?)", text)
             for name in ("Peak", "RMS")]
    return None if not all(found) else tuple(float(m.group(1)) for m in found)


def assert_converted(original: FlacInfo, new_path: str, rate: int, bits: int,
                     expected_samples: int) -> None:
    """Verify a converted file is the one that was asked for, and intact.

    Losslessness is out of the question here - the audio was meant to change -
    so these are the strongest checks that still hold: the stream really has
    the requested format, its length is what the change of rate implies, and
    the MD5 in the header matches the audio the file decodes to. That last one
    is the same guarantee as always, only anchored differently: the header MD5
    is computed over what the encoder was fed, so if it survives a decode, the
    file holds exactly the samples the resampler produced.
    """
    new = read_flac_metadata(new_path)
    for attr, want in (("sample_rate", rate), ("bits_per_sample", bits),
                       ("channels", original.channels)):
        if getattr(new, attr) != want:
            raise RuntimeError(t("err_mismatch", field=attr, old=want,
                                 new=getattr(new, attr)))
    # soxr hits the count exactly; the tolerance is only so that a resampler
    # rounding the last block differently does not fail a whole library. A
    # stream that stopped early is caught by ffmpeg's exit code instead.
    if abs(new.total_samples - expected_samples) > max(8, expected_samples // 1000):
        raise RuntimeError(t("err_samples", new=new.total_samples,
                             old=expected_samples))
    if new.md5 == MD5_UNSET:
        raise RuntimeError(t("err_md5_unset"))
    if new.md5 != raw_audio_md5(new_path):
        raise RuntimeError(t("err_md5_content"))
    flac_test(new_path)

    # Everything above would pass just as happily on a stream the resampler
    # wrecked: the header would be right, the length right, the MD5 honestly
    # self-consistent. Nothing so far has compared the audio to the audio it
    # came from - and once, an ffmpeg dither fault quietly turned a whole
    # album into clipping. Loudness is a coarse thing to compare, but that is
    # the point: resampling barely moves it, and a fault of that kind moves it
    # enormously. The peak is allowed to fall - taking the ultrasound out
    # takes energy with it - so only a rise counts against it.
    old_level, new_level = audio_levels(original.path), audio_levels(new_path)
    if old_level and new_level:
        drift = max(new_level[0] - old_level[0],
                    abs(new_level[1] - old_level[1]))
        if drift > MAX_LEVEL_DRIFT:
            raise RuntimeError(t("err_level", drift=f"{drift:.1f}"))


def convert_file(info: FlacInfo, rate: int, bits: int,
                 dry_run: bool, keep_original: bool = True) -> Outcome:
    """Resample and requantise one file in place.

    ffmpeg only moves the samples to the wanted rate and depth and passes them
    on raw; the file is written by flac, so it comes out with the same preset,
    the same seek table and the same vendor string as everything else here.
    `-vn` keeps the cover art out of that pipe - handed to ffmpeg as a video
    stream it would be re-encoded, and a broken one fails the whole run.

    This is not a lossless write, so the original is put aside as
    <name>.orig.flac like every other write that changes the audio - unless
    the caller asked for it not to be, which the confirmation says out loud
    before any of this starts.
    """
    path = info.path
    old_size = os.path.getsize(path)
    result = Outcome(info, Kind.CONVERT, Status.FAILED, old_size,
                     rate=rate, bits=bits)
    expected = round(info.total_samples * rate / info.sample_rate)
    tmp = os.path.join(os.path.dirname(path) or ".",
                       f".{os.path.basename(path)}.convert.tmp")
    try:
        # Asked before any work: a run that would fail has to say so in a dry
        # run too.
        result.backup_path = set_aside(path, keep_original)
        # ffmpeg's own errors go to a file, not a pipe: flac is holding the
        # other pipe open, and two full buffers would deadlock the pair.
        with tempfile.TemporaryFile() as log:
            src = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-nostdin", "-vn", "-i", path,
                 "-af", resample_filter(bits), "-ar", str(rate),
                 "-f", RAW_FORMATS[bits], "-"],
                stdout=subprocess.PIPE, stderr=log)
            enc = subprocess.run(
                ["flac", "-s", "-f", *ENCODE_OPTS,
                 "--force-raw-format", "--endian=little", "--sign=signed",
                 f"--channels={info.channels}", f"--bps={bits}",
                 f"--sample-rate={rate}", "-o", tmp, "-"],
                stdin=src.stdout, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
            src.stdout.close()
            # flac happily encodes a stream that stopped early, so ffmpeg's
            # exit code is what says the audio came over whole - and it is
            # asked about first, because it explains a failing flac too.
            if src.wait() != 0:
                log.seek(0)
                raise RuntimeError(t("err_resample", code=src.returncode,
                                     detail=_last_line(log.read().decode(
                                         "utf-8", "replace"))))
        if enc.returncode != 0:
            raise RuntimeError(t("err_encode_failed", code=enc.returncode,
                                 detail=_last_line(enc.stderr.decode(
                                     "utf-8", "replace"))))

        # Tags and cover art before the checks, so what is verified is the
        # file that will land on disk and not a stage before it.
        graft_metadata(read_metadata_blocks(path), tmp)
        assert_converted(info, tmp, rate, bits, expected)
        new_size = os.path.getsize(tmp)

        result.new_size = new_size
        if dry_run:
            os.remove(tmp)
            result.status = Status.DRY_RUN
            return result

        put_in_place(path, tmp, result.backup_path)
        result.status = Status.DONE
        return result

    except (OSError, RuntimeError, ValueError) as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return _failed(result, str(e))


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
# corpus built from them by ffmpeg, and the ultrasound again later on 34
# genuine hi-res files (96, 176.4 and 192 kHz) against 9 upsampled ones.
# Worst honest file vs. best fake:
#
#   cliff over 2 bands   honest max 29 dB  |  AAC 128k 49, MP3 128k 61
#   ultrasound gap       honest max 50 dB  |  upsampled 90-97
#   ultrasound spread    honest min 39 dB  |  upsampled 1.8-19.5
#
# What still gets through: AAC from about 192 kbps up, which no longer cuts
# the band at all (measured 15 dB - indistinguishable from the original).
#
# Two of those thresholds were later found to be guarding the wrong ground,
# neither of them by being the wrong number:
#
#   the cliff ceiling was fixed at 20 kHz, which is CD Nyquist's roll-off and
#   nobody else's, so a 48 kHz file could carry a 95 dB wall at 21 kHz - the
#   AAC and Opus cut - and be called honest. It is a fraction of the file's
#   own rate now; see LOSSY_CLIFF_TO.
#
#   the ultrasound was sampled at one narrow probe per 2000 Hz, which reads
#   3 to 6 % of the band it claims to measure, so a genuine partial between
#   two probes was read as arithmetic silence and its file called upsampled.
#   Each band now sweeps its own cell; see ULTRA_JITTER.
# --------------------------------------------------------------------------

LOSSY_CLIFF_DB = 40.0        # cliff from which it looks like a codec
LOSSY_CLIFF_SPAN = 2         # the fall may spread over this many bands
LOSSY_BAND_STEP = 500        # spacing of the measured bands, Hz
LOSSY_BAND_FROM = 14000      # below this codecs do not cut
LOSSY_BAND_TO = 25000        # fine bands end here
#: Above this a fall is the file's own roll-off towards Nyquist, not a codec.
#: 20 kHz is right for CD rate and for nothing else: in a 48 kHz file Nyquist
#: is at 24 kHz, so 20-22 kHz is perfectly measurable spectrum - and it is
#: exactly where AAC and Opus cut. Measured: a 95 dB brick wall at 21 kHz in a
#: 48 kHz file scored 7.5 dB against the flat 20 kHz ceiling (invisible) and
#: 95.3 dB once the ceiling followed the rate.
#: Scaled from 44100 rather than from Nyquist so that CD rate reproduces this
#: number exactly, and every file already measured keeps the answer it had.
LOSSY_CLIFF_TO = 20000       # the ceiling at 44.1 kHz, scaled from there
LOSSY_CLIFF_CAP = 22500      # ... but no codec cuts above this, at any rate
#: A wall standing this far up its source's Nyquist is the resampler's own,
#: and the upsampling charge already accounts for it. Resamplers pass 0.90 to
#: 0.95 of Nyquist (soxr's default is 0.913), so this sits just under them.
LOSSY_WALL_NYQUIST = 0.88
LOSSY_WINDOW = 4096          # Goertzel window length
LOSSY_WINDOW_STRIDE = 8      # every Nth window is taken
LOSSY_FLOOR_DB = -100.0      # below this it is only numerical noise

ULTRA_FROM = 26000           # the ultrasound is sampled from here,
ULTRA_STEP = 2000            # coarsely - only its level matters
ULTRA_GAP_DB = 70.0          # this far below the 15-20 kHz level = silence
ULTRA_SPAN_DB = 30.0         # but only if it is also featureless: more spread
                             # than this means something is up there
ULTRA_EDGE_DB = 20.0         # a band this far above the floor still has content
ULTRA_MIN_RATE = 48000       # only files claiming more than this are checked

#: A Goertzel over a Blackman-Harris window is a NARROW probe, not a band: a
#: tone reads within 6 dB of its true level only +-32 Hz away at 96 kHz, +-64
#: at 192. On a 2000 Hz spacing that is 3 to 6 % of the ultrasound actually
#: looked at, and the other 94 % was assumed to hold whatever the probes hit.
#: Measured: a 35 kHz tone at -26 dBFS in a 192 kHz file was invisible - the
#: file was called upsampled and its rate wasted - while the same tone at
#: 36 kHz, sitting on a probe, read -7 dB.
#:
#: Four times the probes would have cost four times the run time, so instead
#: each coarse band is probed at a DIFFERENT frequency inside its own cell in
#: every window. The energy summed over a track's hundreds of windows is then
#: the average across the cell rather than one point of it: same cost, whole
#: band covered. The offsets walk by the golden ratio because that spreads
#: evenly at every prefix length - a short track still gets even coverage of
#: the windows it did have.
#:
#: The fine bands are deliberately left as exact point probes: the cliff
#: thresholds above were calibrated on them, and averaging across 500 Hz would
#: soften the very edge they measure.
ULTRA_JITTER = 0.6180339887498949

# --------------------------------------------------------------------------
# IS THE SAMPLE RATE WORTH ITS SIZE
#
# A second question about the same spectrum, and a different one. Above is a
# lie detector: the file claims a rate it was never recorded at. This is not
# about lying at all - the samples are honestly there - it is about whether
# anything is IN them. A 192 kHz file whose music stops at 22 kHz is not
# dishonest, it is four times larger than it needs to be.
#
# Four ways the ultrasound gives itself away as something other than music,
# and a file is only ever accused when one of them can be pointed at:
#
#   it climbs    Nothing acoustic gains energy with frequency. A spectrum
#                that turns and rises is noise shaping - a DSD transfer, or
#                a requantisation - and everything from the turn up is that.
#   it goes flat A converter's noise floor is featureless. Where the spectrum
#                settles and stays settled, the music has already stopped.
#   it steps     It falls off a step too steep to be acoustic and never
#                climbs back over it. What trails away above is a tail.
#   it fades     None of those, but the level has fallen so far under the
#                audible band that no rate change could lose anything.
#
# Measured on 43 hi-res files (96, 176.4 and 192 kHz) and 9 upsampled ones.
# The first two need no threshold worth arguing about - the shape decides -
# and they carried 33 of the 43. The other two rest on a chosen number.
# --------------------------------------------------------------------------

BW_FROM = 20000              # the question is only ever about ultrasound
BW_RISE_DB = 6.0             # a climb this far off the low point is shaping
BW_FLAT_DB = 4.0             # a stretch this level is a noise floor,
BW_FLAT_SPAN = 8000          # if it holds that level over this many Hz
BW_EDGE_DB = 6.0             # music ends where it rises this far off a floor
BW_FLOOR_DB = 20.0           # ... and a floor has to be this far down to be
                             # one: a drum solo's cymbals are flat up there
                             # too, and they are the music, not the floor
BW_QUIET_DB = 50.0           # or falls this far under the 15-20 kHz level
#: What fraction of the sample rate a converter really delivers. Not the whole
#: Nyquist: measured, 176.4 -> 44.1 kHz came out flat to 20.0 kHz and
#: 176.4 -> 48 kHz to 22 kHz, both a shade under half.
BW_PASSBAND = 0.49
#: The rates worth being asked to drop to, per family. 22.05 and lower are not
#: offered: no one wants them, and nothing in a library is measured that low.
BW_LADDER = {11025: (44100, 88200, 176400), 8000: (48000, 96000, 192000)}

#: The grid is refined to this spacing around a rung of the ladder that falls
#: in the coarse region. The whole 96-versus-192 decision used to rest on the
#: two probes at 46 and 48 kHz - a 2000 Hz quantisation guarding a boundary at
#: 47040 Hz, against the 960 Hz of margin BW_PASSBAND leaves. Below
#: LOSSY_BAND_TO the fine grid already resolves those rungs to 500 Hz, so only
#: the 88.2 and 96 kHz ones need this.
BW_EDGE_STEP = 500
BW_EDGE_SPAN = 2             # refined bands each side of the boundary

#: What share of the bands above the turn have to have come up with it before
#: a climb counts as a shaped floor rather than one loud partial. Not a second
#: height threshold - the peak that fires the test is at BW_RISE_DB by
#: definition, so nothing above it can be asked for. Width is the whole
#: question, and it separates cleanly: measured across this library, a real
#: shaped transfer keeps 75 % of them up, a lone ultrasonic partial 13 to
#: 26 %, and a local wiggle in an honest transfer 23 %.
BW_SHAPED_SHARE = 0.5

#: How far the spectrum has to fall between two neighbouring bands before the
#: fall is read as the end of the music rather than its roll-off. The other
#: three readings all describe what sits ABOVE the music, and a transfer whose
#: noise tail merely slopes away answers to none of them: it is not flat, it
#: never climbs, and it never gets BW_QUIET_DB down. What it does have is a
#: step nothing acoustic makes - 10 dB in 500 Hz is a 400 dB per octave
#: roll-off, which is a filter and not an instrument. The example is a 96 kHz
#: transfer of a 1988 album: ten of its eleven hi-res tracks fall 10.4 to
#: 18.2 dB across the one step at 20.5 kHz and then trail off, 29 to 42 dB
#: under the music, to Nyquist. Every reading above missed all ten.
#:
#: Unlike the shape readings this one IS a chosen number, and the population
#: it is chosen out of has no gap to hide in: of 112 hi-res files here 25 have
#: a step of 6 dB or more, 16 reach 10 and 7 reach 14. What 10 buys is that
#: the files just under it - 9.4, 9.3, 8.8 dB - are ones already caught by the
#: floor or ones whose spectrum genuinely runs on, while from 6 to 10 the list
#: is honest wideband transfers. Raising it loses whole albums; lowering it
#: starts charging roll-offs that are merely steep.
#:
#: Only neighbours BW_EDGE_STEP apart are compared, so the test reads a slope
#: and not an artefact of the grid: that is the 500 Hz fine grid up to
#: LOSSY_BAND_TO, and the refined bands _edge_bands puts around the 88.2 and
#: 96 kHz rungs. The 2000 Hz ultrasound bands are far too coarse to ask.
BW_STEP_DB = 10.0

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
class Inspection:
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
    windows: int = 0                            # how much was listened to
    content_hz: int = 0                         # where the music stops
    content_why: str = ""                       # and what is above it
    ultra_level_db: float = 0.0                 # the floor, or the shaping peak
    ultra_peak_hz: int = 0                      # where that peak sits
    enough_rate: int = 0                        # a rate that would hold it all
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
    def wasteful(self) -> bool:
        """Does the file carry less than its sample rate is paying for?"""
        return self.enough_rate > 0

    @property
    def suspicious(self) -> bool:
        """Does the file lie? A check that could not be made accuses nobody.

        The depth is read from the samples before the spectrum is measured, so
        a track too silent to have a 1 kHz reference reaches the error with
        real_bits already set - and on an all-zero track that would "prove" a
        padded 8 bit. Such a file belongs in the "could not be measured" group
        and nowhere else.
        """
        if self.error:
            return False
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


def _windows(info: FlacInfo):
    """Every LOSSY_WINDOW-th sample block of the whole track, mono, -1..+1.

    A stream and not a slice. Measuring 30 seconds and calling it the track
    was a lottery: on this library the depth of the deepest cliff moved by up
    to 12 dB depending on where the 30 seconds landed, which is most of the
    margin between the worst honest file and the threshold. The whole track
    costs almost nothing extra, because only every STRIDE-th window is looked
    at and this reads only those - the old code converted every sample of the
    slice and then used one eighth of them.

    Yields (mono window, low bytes seen) so the caller can measure the depth
    from the same bytes; nothing but one window is ever held in memory.

    Bit depth MUST come from the header: files converted from MP3 are often
    24 bit, and reading them as 16 bit would return noise.
    """
    width = info.bits_per_sample // 8
    frame = width * info.channels
    scale = float(1 << (info.bits_per_sample - 1))
    proc = subprocess.Popen(
        ["flac", "-d", "-c", "-s", "--force-raw-format", "--endian=little",
         "--sign=signed", "--", info.path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        gap = LOSSY_WINDOW * (LOSSY_WINDOW_STRIDE - 1) * frame
        while True:
            raw = proc.stdout.read(LOSSY_WINDOW * frame)
            if len(raw) < LOSSY_WINDOW * frame:
                return
            mono = []
            for i in range(LOSSY_WINDOW):
                off = i * frame
                total = sum(int.from_bytes(raw[off + c * width:
                                               off + (c + 1) * width],
                                           "little", signed=True)
                            for c in range(info.channels))
                mono.append(total / info.channels / scale * _BH_WINDOW[i])
            yield mono, raw
            if len(proc.stdout.read(gap)) < gap:
                return
    finally:
        proc.stdout.close()
        proc.wait()


def _dead_bytes(raw: bytes, width: int, dead: int) -> int:
    """How many of the low `dead` bytes of every sample are still all zero.

    Samples are contiguous little-endian words, so `raw[0::width]` is the low
    byte of every sample of every channel. All zero across the whole track
    means the audio was widened from a shallower depth and the extra bits
    carry nothing. Only whole bytes are counted - a partial fill (a 20 bit
    ADC) is honest hardware, not a lie.
    """
    for byte in range(dead):
        if any(raw[byte::width]):
            return byte
    return dead


def _edge_bands(rate: int, top: float) -> dict:
    """Extra probes around the rungs of the ladder that sit in the coarse grid.

    `enough_rate` turns on whether the music stops below `r * BW_PASSBAND`, so
    that boundary deserves better than the 2000 Hz spacing the ultrasound is
    otherwise swept at. Only the rungs above LOSSY_BAND_TO need it; the ones
    below are already standing on the 500 Hz fine grid.
    """
    ladder = BW_LADDER.get(rate_family(rate), BW_LADDER[8000])
    bands = {}
    for rung in ladder:
        if rung >= rate:
            continue
        centre = round(rung * BW_PASSBAND / BW_EDGE_STEP) * BW_EDGE_STEP
        for step in range(-BW_EDGE_SPAN, BW_EDGE_SPAN + 1):
            f = centre + step * BW_EDGE_STEP
            # Never below LOSSY_BAND_TO: down there the fine bands own the
            # frequency, and they have to stay exact point probes.
            if LOSSY_BAND_TO < f < top:
                bands[f] = BW_EDGE_STEP
    return bands


def inspect_audio(info: FlacInfo) -> Inspection:
    """Look for the marks of a lossy source, upsampling and a padded depth."""
    result = Inspection(path=info.path, claimed_rate=info.sample_rate,
                       claimed_bits=info.bits_per_sample)
    rate, width = info.sample_rate, info.bits_per_sample // 8
    if not (rate and info.channels and width):
        result.error = "fake_unusable"
        return result
    top = rate / 2 * 0.95
    fine = [f for f in range(LOSSY_BAND_FROM, LOSSY_BAND_TO + 1, LOSSY_BAND_STEP)
            if f < top]
    coarse = ([f for f in range(ULTRA_FROM, int(top) + 1, ULTRA_STEP)]
              if rate > ULTRA_MIN_RATE else [])
    if len(fine) < 3:
        result.error = "fake_unusable"
        return result

    # Band -> the width of the cell it stands for, 0 for an exact point probe.
    # The fine bands measure a frequency; everything above measures a stretch
    # of spectrum, swept across the windows. See ULTRA_JITTER.
    cells = {f: 0 for f in fine}
    cells.update({f: ULTRA_STEP for f in coarse})
    if coarse:
        cells.update(_edge_bands(rate, top))

    energy = {f: 0.0 for f in cells}
    reference = 0.0
    dead = width - 1              # low bytes still seen all zero everywhere
    windows = 0
    try:
        for chunk, raw in _windows(info):
            windows += 1
            dead = _dead_bytes(raw, width, dead)
            reference += _goertzel(chunk, rate, 1000)
            # One offset per window, shared by every band, walking the whole
            # cell over the track. Never leaves the cell, so a band still
            # reports its own stretch of spectrum and nobody else's.
            spread = (windows * ULTRA_JITTER % 1.0) - 0.5
            for f, cell in cells.items():
                energy[f] += _goertzel(chunk, rate, f + spread * cell)
    except OSError:
        result.error = "fake_unusable"
        return result

    if windows < 2 or reference <= 0:
        result.error = "fake_unusable"
        return result
    if dead:
        result.real_bits = info.bits_per_sample - dead * 8
    result.windows = windows
    result.bands = {f: 10 * math.log10(e / reference + 1e-30)
                    for f, e in energy.items()}
    _find_cliff(result, fine, rate)
    _find_upsampling(result, fine, coarse)
    _blame_the_resampler(result)
    if coarse:
        _find_bandwidth(result, rate)
    return result


def _blame_the_resampler(result: Inspection) -> None:
    """One wall, one accusation: a resampler's cut is not a codec's.

    Letting the cliff ceiling rise above 20 kHz brought a confusion with it.
    Upsampling 44.1 kHz to 96 leaves a brick wall just under 22.05 kHz, and
    that is indistinguishable from what MP3 at 320 kbps leaves behind - the
    file would now be charged with both, and the lossy line would name a
    bitrate nobody measured. Where the wall stands at the source's own
    Nyquist, the upsampling already explains it and the lossy charge goes.

    Only where they COINCIDE, though. An MP3 128 upsampled to 96 kHz walls at
    16.5 kHz while its source Nyquist is 22.05, and that file really is guilty
    of both - both lines are worth reading, and it keeps them.
    """
    if not (result.upsampled and result.lossy_source):
        return
    if result.cutoff_hz >= result.source_rate / 2 * LOSSY_WALL_NYQUIST:
        result.cutoff_hz, result.cliff_db = 0, 0.0


def _cliff_ceiling(rate: int) -> float:
    """Above this a fall is the file's own roll-off, not a codec's brick wall.

    Proportional to the rate, not a fixed 20 kHz. At CD rate the two agree,
    and there the fixed number was right: 20 kHz is where a 44.1 kHz file runs
    out of spectrum by itself. At 48 kHz it was simply wrong - the roll-off is
    up at 24 kHz, 20 to 22 kHz is honest measurable band, and it is exactly
    where AAC and Opus put their cut. Capped, because whatever the container's
    rate, no codec cuts above LOSSY_CLIFF_CAP; a wall higher than that is a
    resampler's, and _find_upsampling is the test that names it.

    Scaled from 44100 and not from rate/2 on purpose: the ceiling has to come
    out at exactly 20000 for a CD-rate file, so that the 44.1 kHz half of a
    library - which this ceiling was never wrong about - keeps every number it
    already had. A fraction of Nyquist lands a hair under and quietly drops
    the 20 kHz band from the search.
    """
    return min(LOSSY_CLIFF_CAP, LOSSY_CLIFF_TO * rate / 44100)


def _find_cliff(result: Inspection, fine: list, rate: int) -> None:
    """The brick wall a lossy codec leaves behind.

    The fall is measured over LOSSY_CLIFF_SPAN neighbouring bands, not one:
    an encoder's transition band is a few hundred Hz wide, so a single step
    catches only part of it and AAC slips through. Bands already at the
    numerical floor are not compared - a "drop" there is just noise - and a
    fall starting above the ceiling is ignored, because that is where the file
    naturally runs out of spectrum anyway.
    """
    ceiling = _cliff_ceiling(rate)
    for i, f in enumerate(fine[:-1]):
        if f > ceiling or result.bands[f] < LOSSY_FLOOR_DB:
            continue
        after = fine[i + 1: i + 1 + LOSSY_CLIFF_SPAN]
        if not after:
            break
        drop = result.bands[f] - min(result.bands[g] for g in after)
        if drop > result.cliff_db:
            result.cliff_db, result.cutoff_hz = drop, f
    if result.cliff_db < LOSSY_CLIFF_DB:
        result.cutoff_hz = 0          # gradual roll-off, nothing suspicious


def _find_upsampling(result: Inspection, fine: list, coarse: list) -> None:
    """Ultrasound that is not merely quiet but arithmetically absent.

    The level is compared with the file's own 15-20 kHz content rather than
    with a fixed figure, so a quiet or dull recording is judged by the same
    ratio as a bright one.

    Quiet is not enough on its own, though, and the median that measures it is
    the reason. A 192 kHz file is sampled from 26 kHz to 91 kHz, and a genuine
    wide recording spends most of that range in its own tail, which drags the
    median down until it looks like silence - seven of one album here were
    accused of being upsampled while carrying real content out to 46 kHz. What
    a resampler actually leaves behind is not just quiet, it is FEATURELESS:
    arithmetic zero has no shape. So the spread across those bands has to be
    small too. Measured on 9 upsampled files against 34 genuine hi-res ones:
    upsampled spans 1.8-19.5 dB (the widest being one dithered on the way),
    genuine 39-83. Nothing lives in between.
    """
    if not coarse:
        return
    audible = [result.bands[f] for f in fine if 15000 <= f <= 20000]
    if not audible:
        return
    levels = [result.bands[f] for f in coarse]
    result.ultra_gap_db = statistics.median(audible) - statistics.median(levels)
    if result.ultra_gap_db < ULTRA_GAP_DB:
        return
    if max(levels) - min(levels) > ULTRA_SPAN_DB:
        return          # structure up there, so something is up there
    # The highest band still carrying anything marks the original Nyquist.
    floor = statistics.median(levels)
    edge = max((f for f in fine if result.bands[f] > floor + ULTRA_EDGE_DB),
               default=0)
    result.source_rate = next((r for limit, r in UPSAMPLE_HINTS if edge <= limit),
                              0) or int(edge * 2 / 1000) * 1000


def _find_bandwidth(result: Inspection, rate: int) -> None:
    """How far up the file actually carries music, and what would hold it.

    Three readings of the one spectrum, and the lowest wins: whichever of them
    happens first walking up is where the music stopped, whatever the ones
    above it say. Nothing is claimed at all unless one of them fires - a file
    whose spectrum simply keeps going is a file whose rate is earned.

    "Whatever the ones above it say" is true of the three READINGS and not of
    the spectrum itself, which is why the edge is checked against the bands
    afterwards: a reading that puts the music's end below something still
    audibly loud has misread the file, not found its limit.

    Each hit carries the level worth quoting with it, because the same number
    means different things: for a floor it is the level of the flat stretch
    that identified it, for shaping the height of the hump.
    """
    up = sorted(f for f in result.bands if f >= BW_FROM)
    audible = [v for f, v in result.bands.items() if 15000 <= f <= 20000]
    if len(up) < 4 or not audible:
        return
    reference = statistics.median(audible)
    hits = {}                        # why -> (edge, level to quote, peak)

    # It climbs. Nothing acoustic does, so the turn is the last place the
    # music could still have been - but only where the whole floor climbed
    # with it. A shaped floor is BROAD: it turns and stays up all the way to
    # Nyquist. One band this high is a partial - a cymbal, a string's upper
    # harmonic, a resonance - and that is music, not a floor to be cut away.
    # Once the ultrasound stopped being sampled at a single point per 2000 Hz
    # (see ULTRA_JITTER) such partials started landing in the bands, and
    # without this each of them would have been read as noise shaping.
    #
    # The share above the turn is what decides, never a second height: the
    # peak that fires this test sits at BW_RISE_DB by definition, so asking
    # for more height only rejects the gradual transfers this test exists for.
    low, turn = result.bands[up[0]], up[0]
    for f in up:
        if result.bands[f] < low:
            low, turn = result.bands[f], f
        elif result.bands[f] > low + BW_RISE_DB:
            above = [result.bands[g] for g in up if g > turn]
            raised = [v for v in above if v > low]
            if len(raised) >= len(above) * BW_SHAPED_SHARE:
                peak = max((g for g in up if g > turn),
                           key=lambda g: result.bands[g])
                hits["shaping"] = (turn, result.bands[peak], peak)
            break

    # It goes flat, and does so far enough down to be a converter's floor and
    # not the music itself. The music stopped where it left that level.
    for i, f in enumerate(up):
        if up[-1] < f + BW_FLAT_SPAN:
            break
        run = [result.bands[g] for g in up[i:] if g <= f + BW_FLAT_SPAN]
        level = statistics.median(run)
        if max(run) - min(run) < BW_FLAT_DB and level < reference - BW_FLOOR_DB:
            edge = max((g for g in up if g < f
                        and result.bands[g] > level + BW_EDGE_DB), default=f)
            hits["floor"] = (edge, level, 0)
            break

    # It steps down, and what is left above the step never climbs back over
    # it. Nothing acoustic falls BW_STEP_DB in 500 Hz, so the step is the end
    # of the music whatever put it there - the tape's own bandwidth, the desk,
    # the converter in front of the transfer. This is the reading for the file
    # that has a LITTLE something up there: not flat enough to be a floor, not
    # quiet enough to have faded, and still not worth the samples it costs.
    #
    # The quoted level is the median of the whole tail rather than the band
    # just under the step, because that is the depth the tail actually sits
    # at, and because it is what the content guard below measures against.
    for prev, f in zip(up, up[1:]):
        if f - prev > BW_EDGE_STEP or f == up[-1]:
            continue
        if result.bands[prev] - result.bands[f] < BW_STEP_DB:
            continue
        tail = [result.bands[g] for g in up if g >= f]
        if max(tail) <= result.bands[f] + BW_EDGE_DB:
            hits["step"] = (prev, statistics.median(tail), 0)
            break

    # It fades out from under the audible band.
    if faded := [f for f in up if result.bands[f] < reference - BW_QUIET_DB]:
        hits["quiet"] = (faded[0], reference - BW_QUIET_DB, 0)

    if not hits:
        result.content_hz = max(up)
        return
    result.content_why = min(hits, key=lambda k: hits[k][0])
    result.content_hz, level, peak = hits[result.content_why]

    # The winning reading found the FIRST thing on the way up, and it can be
    # wrong about everything above it: an ultrasonic partial sitting past the
    # flat stretch that "proved" the music had already ended leaves the file
    # told its rate is being wasted on silence it is in fact using. So the
    # edge moves up past anything that is unmistakably still content.
    #
    # Unmistakably, and it takes both halves to say it. A band has to TOWER
    # over the level the reading was taken at - BW_FLOOR_DB, the same margin
    # that makes a floor a floor, read upwards - because a transfer's noise
    # can decline smoothly from under the music all the way to Nyquist
    # without ever being 50 dB down, and reading that as "the music reaches
    # 90 kHz" costs four 192 kHz files here their perfectly good advice. And
    # it has to be within BW_QUIET_DB of the music, on audible scale, because
    # 20 dB above a floor 120 dB down is still 100 dB down: a rounding error,
    # not a cymbal. A partial clears both by tens of dB; nothing else clears
    # either.
    #
    # Shaping caps the search instead of extending it: everything above ITS
    # turn was identified as noise, and a shaped hump can be louder than the
    # music - it is the one thing up there that is loud and still not music.
    ceiling = hits["shaping"][0] if "shaping" in hits else up[-1]
    loud = max((f for f in up if f <= ceiling
                and result.bands[f] > level + BW_FLOOR_DB
                and result.bands[f] > reference - BW_QUIET_DB), default=0)
    result.content_hz = max(result.content_hz, loud)

    result.ultra_level_db = level - reference
    result.ultra_peak_hz = peak
    ladder = BW_LADDER.get(rate_family(rate), BW_LADDER[8000])
    result.enough_rate = next((r for r in ladder
                               if r * BW_PASSBAND >= result.content_hz
                               and r < rate), 0)


def fake_ok_lines(check: Inspection) -> list:
    """What was measured on a file that turned out to tell the truth.

    A negative result is worth printing too - `inspect --all` is asked for
    exactly when someone wants to see that a file WAS checked - so each test
    says what it found, joined into the one line per file the listings use.
    """
    parts = [t("fake_ok_cutoff")]
    # A gap of zero means the test never ran: the ultrasound is only looked at
    # above 48 kHz, and only where there are bands left to look at.
    if check.ultra_gap_db:
        parts.append(t("fake_ok_ultra", db=check.ultra_gap_db))
    # An honest file leaves real_bits unset, so the claim is also the truth.
    parts.append(t("fake_ok_bits", bits=check.claimed_bits))
    # Said only where nothing found the rate wanting; otherwise the file has
    # its own line in the other group saying how far it really goes.
    if check.content_hz and not check.wasteful:
        parts.append(t("bw_ok", khz=f"{check.content_hz / 1000:g}"))
    return [", ".join(parts)]


def bandwidth_lines(check: Inspection) -> list:
    """The one line saying how far a file's music reaches, and what would hold it.

    The level is worded the way each finding is actually read: a floor is
    quoted as a depth below the music, because that is what makes it a floor,
    while shaping keeps its sign - the whole point of the Cyndi Lauper
    transfer here is that its ultrasound is ELEVEN dB LOUDER than the music.
    """
    key = "bw_" + check.content_why
    extra = {"db": f"{-check.ultra_level_db:.0f}"}
    if check.content_why == "shaping":
        extra = {"db": f"{check.ultra_level_db:+.0f}",
                 "peak": f"{check.ultra_peak_hz / 1000:g}"}
    elif check.content_why == "quiet":
        extra = {"db": f"{BW_QUIET_DB:.0f}"}
    return [t(key, khz=f"{check.content_hz / 1000:g}",
              rate=f"{check.enough_rate / 1000:g} kHz", **extra)]


def fake_lines(check: Inspection) -> list:
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
        put_in_place(path, tmp)     # the audio is untouched, nothing to keep
    except (OSError, RuntimeError, ValueError) as e:
        _failed(result, str(e))
        try:
            os.remove(tmp)
        except OSError:
            pass
    return result


def repair_damaged_file(info: FlacInfo, dry_run: bool,
                        keep_original: bool = True) -> Outcome:
    """Salvage what is readable from a damaged file and put it in its place.

    `flac -d -F` keeps decoding past errors, so everything readable comes out
    of a truncated or corrupted stream.

    The original is put aside as <name>.orig.flac, the same as any other write
    that changes the audio - and the one where it matters most: losslessness
    cannot be verified here, part of the audio is already gone and the MD5 in
    the header will never match again, so the untouched original is the last
    chance for a better recovery later. Dropping it is the caller's to ask for.
    """
    path = info.path
    result = Outcome(info=info, kind=Kind.SALVAGE,
                     status=Status.DRY_RUN if dry_run else Status.DONE)
    tmp = os.path.join(os.path.dirname(path) or ".",
                       f".{os.path.basename(path)}.repair.tmp")
    target = os.devnull if dry_run else tmp
    try:
        # Asked before any work: a run that would fail has to say so in a dry
        # run too.
        result.backup_path = set_aside(path, keep_original)
        dec = subprocess.Popen(
            ["flac", "-d", "-F", "-s", "-c", "--", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        enc = subprocess.run(
            ["flac", "-s", "-f", *ENCODE_OPTS, "-o", target, "-"],
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

        put_in_place(path, tmp, result.backup_path)
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
    inspection: Inspection | None = None
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
    def no_md5(self) -> bool:
        """Header without the MD5 of the audio, so nothing can ever verify it.

        Not a defect of the audio and not worth a severity - the file plays
        and decodes - but it is the one property that cannot be found out
        later, and re-encoding quietly fixes it.
        """
        return self.info.md5 == MD5_UNSET

    @property
    def needs_attention(self) -> bool:
        """Worth printing without --all."""
        return bool(self.weak or self.subset_fixable or self.subset_inherent
                    or self.info.error
                    or (self.inspection and self.inspection.suspicious))


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
              Inspection: {"path", "bands"}, Report: {"path"}}

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
    if raw := d.get("inspection"):
        item.inspection = from_json(Inspection, raw, path=info.path)
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


def size_of(path: str) -> int:
    """Bytes on disk, 0 for a file that is not there."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def audio_format(rate: int, bits: int) -> str:
    """A stream format the way the whole script says it: '44.1 kHz/16 bit'."""
    return f"{rate / 1000:g} kHz/{bits} bit"


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
    followed, and the originals put aside by `repair` and `reencode` are
    skipped - they are superseded by definition and would otherwise be
    reported, re-encoded and converted forever."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        found += [os.path.join(dirpath, n) for n in sorted(filenames)
                  if n.lower().endswith(".flac")
                  and not n.lower().endswith(ORIG_SUFFIX)]
    return found


def collect_originals(root: str) -> list:
    """Every original put aside under `root`, sorted - exactly what
    collect_flac_files deliberately leaves out."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        found += [os.path.join(dirpath, n) for n in sorted(filenames)
                  if n.lower().endswith(ORIG_SUFFIX)]
    return found


def replaced_file(backup: str) -> str:
    """The file an original was put aside for."""
    return backup[:-len(ORIG_SUFFIX)] + ".flac"


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

    detail = [audio_format(info.sample_rate, info.bits_per_sample),
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

    if item.inspection and item.inspection.wasteful:
        for line in bandwidth_lines(item.inspection):
            print(f"   {line}")
    if item.inspection and item.inspection.suspicious:
        for line in fake_lines(item.inspection):
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


def print_wasteful(checks: Sequence, root: str) -> None:
    """Files carrying less music than their sample rate is paying for."""
    print_group(t("bw_header", count=files(len(checks))),
                [(rel(c.path, root), bandwidth_lines(c))
                 for c in sorted(checks, key=lambda c: c.path)],
                footer=t("bw_caveat"), blank=False)


def print_honest(checks: Sequence, root: str) -> None:
    """Files that turned out to be what their header claims."""
    print_group(t("fake_ok_header", count=files(len(checks))),
                [(rel(c.path, root), fake_ok_lines(c))
                 for c in sorted(checks, key=lambda c: c.path)], blank=False)


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

    if found := by_name([a for a in report.items if a.no_md5]):
        print_group(t("nomd5_header", count=files(len(found))),
                    [(rel(a.info.path, root), []) for a in found])

    if found := [a for a in report.items if a.inspection and a.inspection.suspicious]:
        print_fakes([a.inspection for a in found], root)


#: The symbol each status is announced with. Two characters wide throughout,
#: so the file names line up whatever happened to them.
MARKS = {Status.DONE: "✅", Status.DRY_RUN: "○ ", Status.FAILED: "❌"}


def result_detail(res: Outcome) -> str:
    """The one line saying what happened to a file.

    Failure and "nothing gained" read the same whatever was attempted, so
    only a real result is worded per kind.
    """
    if res.status is Status.FAILED:
        key = ("rc_failed" if res.kind in (Kind.REENCODE, Kind.CONVERT,
                                           Kind.SUBSET) else "fix_failed")
        return t(key, error=res.error)
    dry = res.status is Status.DRY_RUN
    if res.kind is Kind.REENCODE:
        if res.saved <= 0:
            # These are rewritten too, so say plainly that it bought nothing.
            return (t("rc_nogain_dry", pct=abs(res.saved_pct)) if dry
                    else t("rc_nogain_done", old=human(res.old_size),
                           new=human(res.new_size), pct=abs(res.saved_pct)))
        return (t("rc_would", saved=human(res.saved), pct=res.saved_pct) if dry
                else t("rc_replaced", old=human(res.old_size),
                       new=human(res.new_size), saved=human(res.saved),
                       pct=res.saved_pct))
    if res.kind is Kind.CONVERT:
        # A conversion is not judged on size at all - it was asked for - so
        # the format is what the line leads with and the size only follows.
        new_fmt = audio_format(res.rate, res.bits)
        change = (t("rc_saved_pct", pct=res.saved_pct) if res.saved > 0
                  else t("rc_grew", pct=-res.saved_pct))
        return (t("rc_conv_would", newfmt=new_fmt, change=change) if dry
                else t("rc_conv_done", oldfmt=audio_format(
                           res.info.sample_rate, res.info.bits_per_sample),
                       newfmt=new_fmt, old=human(res.old_size),
                       new=human(res.new_size), change=change))
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


def print_result(res: Outcome, root: str) -> None:
    """One outcome: what happened, and what it left behind."""
    lines = [result_detail(res)]
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


def original_note(keep: bool) -> str:
    """The one line both commands say about what becomes of the originals."""
    return t("orig_kept", suffix=ORIG_SUFFIX) if keep else t("orig_drop")


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


def matches_disk(info: FlacInfo) -> bool:
    """Is the file still the one the report describes?

    Size and modification time, not a checksum: everything stored about a file
    was read from those exact bytes, so anything that touched them invalidates
    the lot - and a report must never be trusted about a file that has moved
    on underneath it.
    """
    try:
        stat = os.stat(info.path)
    except OSError:
        return False
    return (stat.st_size == info.file_size
            and stat.st_mtime_ns == info.mtime_ns)


def live_targets(items: Sequence, root: str) -> tuple:
    """Split targets into those still matching the report and those that
    changed on disk since. Acting on stale findings could rewrite a file for
    reasons that no longer hold."""
    fresh, skipped = [], []
    for item in items:
        name = rel(item.info.path, root)
        if not os.path.exists(item.info.path):
            skipped.append(t("rp_gone_file", name=name))
        elif not matches_disk(item.info):
            skipped.append(t("rp_stale_file", name=name))
        else:
            fresh.append(item)
    return fresh, skipped


def adopt_new_files(report: Report, jobs: int,
                    paths: Sequence | None = None) -> list:
    """Take in files that appeared under the folder since the last analyze.

    `reencode --all` promises every file in the folder, and the report is only
    as fresh as the last analyze. Re-encoding one needs nothing but its header
    - the deep analysis decides which files are WORTH encoding, not whether it
    is safe - so the header is read here and the entry marked partial, leaving
    the stream itself for the next analyze.
    """
    known = report.by_path()
    if paths is None:
        paths = collect_flac_files(report.root)
    fresh = [p for p in paths if p not in known]
    if not fresh:
        return []
    items = [classify(info) for info in read_all_metadata(fresh, jobs)]
    for item in items:
        item.partial = True
    report.items += items
    return items


def refresh_entry(item: Analysis, keep_inspection: bool) -> None:
    """Update one entry after its file was rewritten.

    Only the header is read again - re-running the deep analysis here would
    decode the whole library a second time. The entry is marked partial so
    both the listing and the next `analyze` know the stream itself has not
    been looked at since.
    """
    kept = item.inspection if keep_inspection else None
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
    item.inspection = kept
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
         ("sum_no_md5", count(lambda a: a.no_md5)),
         ("sum_subset_fix", count(lambda a: a.subset_fixable)),
         ("sum_subset_keep", count(lambda a: a.subset_inherent)),
         ("sum_fake", count(lambda a: a.inspection and a.inspection.suspicious)),
         ("sum_waste", count(lambda a: a.inspection and a.inspection.wasteful))))


def report_advice(report: Report, args) -> list:
    """The one command that makes sense next."""
    root, advice = report.root, []
    # Everything below is read out of the report, so a report describing files
    # that are not there any more has to be said first - it makes every other
    # line of advice fiction. Only `analyze` rebuilds it from the folder.
    if gone := [a for a in report.items if not os.path.exists(a.info.path)]:
        advice.append(t("adv_stale", cmd=command_hint("analyze", root),
                        count=files(len(gone))))
    # repair fixes defects, reencode saves space - two different questions.
    broken = [a for a in report.items
              if a.really_damaged or a.harmless
              or (a.subset_fixable and not a.info.error)]
    weak = [a for a in report.items if not a.info.error and a.weak]
    inherent = [a for a in report.items if a.subset_inherent]
    unchecked = [a for a in report.items if not a.unreadable and not a.inspection]
    lying = [a for a in report.items if a.inspection and a.inspection.suspicious]

    if broken:
        advice.append(t("adv_repair", cmd=command_hint("repair", root),
                        count=files(len(broken))))
    if weak:
        size = sum(a.info.file_size for a in weak)
        advice.append(t("adv_reencode", cmd=command_hint("reencode", root),
                        count=files(len(weak)),
                        eta=estimate(size, ENCODE_RATE, JOBS)))
    # An entry can sit in the report without ever having been decoded: taken
    # in by inspect or reencode, or rewritten since. Only analyze fills it
    # in, and until it does, "nothing found" means "nothing looked at".
    if shallow := [a for a in report.items if a.partial and not a.info.error]:
        advice.append(t("adv_partial", cmd=command_hint("analyze", root),
                        count=files(len(shallow))))
    # Only --all reaches a file that is otherwise fine, which most of these
    # are - so the hint has to carry the flag with it.
    if no_md5 := [a for a in report.items if a.no_md5]:
        advice.append(t("adv_no_md5",
                        cmd=command_hint("reencode", root, " --all"),
                        count=files(len(no_md5))))
    if inherent:
        advice.append(t("adv_subset_keep", count=files(len(inherent))))
    if lying:
        advice.append(t("adv_fake", count=files(len(lying))))
    if unchecked:
        size = sum(a.info.file_size for a in unchecked)
        advice.append(t("adv_inspect", cmd=command_hint("inspect", root),
                        eta=estimate(size, FAKE_RATE, JOBS)))
    return advice


def has_findings(report: Report) -> bool:
    """Is there anything in the report worth acting on?

    A partial entry counts: it has never been decoded, so calling it clean
    would be a verdict on a file nobody has looked at.
    """
    return any(a.weak or a.subset_fixable or a.subset_inherent or a.info.error
               or a.partial or a.no_md5 or (a.inspection and a.inspection.suspicious)
               for a in report.items)


def print_outcome(report: Report, args) -> None:
    """Verdict first, then the one command that makes sense next."""
    if not has_findings(report):
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

    infos = read_all_metadata(paths, JOBS)
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
                     JOBS, weight=_by_size)

    fresh = [classify(info) for info in todo]
    if broken := [a for a in fresh if a.damaged]:
        reports = run_parallel(t("dmg_locating", count=files(len(broken))),
                               broken, lambda a: diagnose_damage(a.info),
                               JOBS, weight=_by_size)
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


def cmd_inspect(args) -> int:
    """Separate analysis: is the file what its header claims?

    Three lies are looked for - a lossy source, upsampled hi-res and a bit
    depth padded with zeros. None is actionable: a file that was thrown away
    at 128 kbps cannot be repaired, only replaced by a better copy. That is
    why this stays out of the analyze/reencode/repair chain. --all also lists
    what each honest file was cleared on.

    The folder decides which files are looked at, never the report: measuring
    a file that is no longer there would only produce "too short or too quiet"
    for every entry of a library that has since been moved. The report is used
    for what it is good for - keeping the results, and handing back the ones
    still describing the file on disk.
    """
    report_file = args.report or default_report_path(args.folder)
    try:
        report = load_report(report_file)
    except ValueError:
        report = None

    print(t("run_scanning", root=short(args.folder)))
    paths = collect_flac_files(args.folder)
    if not paths:
        print(t("run_none"))
        return 0

    if report:
        # Files that appeared since the last analyze are taken in, entries
        # whose file is gone simply do not survive the rebuild.
        adopt_new_files(report, JOBS, paths)
        known = report.by_path()
        report.items = [known[path] for path in paths]
        # Damaged files are included: analyze failed them on a decode error,
        # but the sampled window is usually far from the damage and a file
        # that lies about its quality is worth knowing about either way.
        # Only an unreadable header leaves nothing to measure.
        targets = [a for a in report.items if not a.unreadable]
    else:
        targets = [Analysis(info=info)
                   for info in read_all_metadata(paths, JOBS)
                   if not info.error]

    if not targets:
        print(t("run_none"))
        return 0

    # A stored result still describes a file that has not moved a byte since,
    # which is the same test analyze uses to reuse its own work.
    todo = [a for a in targets
            if args.force or not (a.inspection and matches_disk(a.info))]
    if report and not args.force:
        print("  " + t("rp_reused", count=len(targets) - len(todo),
                       fresh=len(todo)))

    if todo:
        measured = run_parallel(t("fake_running", count=len(todo)),
                                [a.info for a in todo], inspect_audio,
                                JOBS, weight=_by_size, cpu_bound=True)
        for item, res in zip(todo, measured):
            item.inspection = res
    results = [a.inspection for a in targets if a.inspection]
    found = [r for r in results if r.suspicious]

    print()
    if found:
        print_fakes(found, args.folder, blank=False)
    else:
        print(t("fake_none"))

    # The blank line is the group's own: every other section here is printed
    # tight against the one before it, and this one can run for pages.
    if wasteful := [r for r in results if r.wasteful and not r.error]:
        print()
        print_wasteful(wasteful, args.folder)
    elif not found:
        print(t("bw_none"))

    if args.all and (honest := [r for r in results
                                if not r.suspicious and not r.error]):
        print()
        print_honest(honest, args.folder)

    if skipped := [r for r in results if r.error]:
        print_group(t("fake_skipped", count=files(len(skipped)),
                      reason=t(skipped[0].error)),
                    [(rel(r.path, args.folder), [])
                     for r in sorted(skipped, key=lambda r: r.path)],
                    blank=False)

    if report:
        store(report, args)
    return 0


def run_write_command(args, report: Report, targets: Sequence, nothing: str,
                      intro: Callable, running: Callable, worker: Callable,
                      summary: Callable) -> int:
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
    results = run_parallel(running(live), live, worker, JOBS,
                           weight=_by_size)

    print()
    for res in sorted(results, key=lambda r: r.info.path):
        print_result(res, report.root)

    by_path, changed = report.by_path(), 0
    for res in results:
        if res.status is Status.DONE:
            # Salvaging and converting rewrite the audio, so only there does
            # the stored spectral check stop describing the file.
            refresh_entry(by_path[res.info.path],
                          keep_inspection=res.kind not in (Kind.SALVAGE, Kind.CONVERT))
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


def target_format(info: FlacInfo, args) -> tuple:
    """(rate, bits) the file is meant to end up in. What the user did not ask
    about stays as it is, so `--bits 16` alone never touches a sample rate.

    --downgrade is applied right here, and to each half on its own, which is
    the whole reason it is a clamp and not a filter over files. The two halves
    are two independent asks: `--sample-rate 48000 --bits 16` against a
    44.1 kHz / 24 bit file says "the rate is already below, leave it" and "yes,
    take the depth down". Declining the file whole would throw away the half
    that was worth doing, and declining it in the target is what makes every
    count, warning and decision downstream come out right without knowing the
    option exists - a file left at its own format simply needs no conversion.
    """
    rate = args.sample_rate or info.sample_rate
    bits = args.bits or info.bits_per_sample
    if getattr(args, "downgrade", False):
        rate = min(rate, info.sample_rate)
        bits = min(bits, info.bits_per_sample)
    return rate, bits


def would_raise(info: FlacInfo, args) -> bool:
    """Would the asked-for format lift this file's rate or depth? That is the
    half --downgrade declines, and it is worth counting out loud: a run that
    quietly left most of a folder alone otherwise looks like a broken run."""
    return ((args.sample_rate or 0) > info.sample_rate
            or (args.bits or 0) > info.bits_per_sample)


def needs_conversion(info: FlacInfo, args) -> bool:
    """Is this file not already in the wanted format? A file that is stays out
    of ffmpeg's way and is only re-encoded, bit for bit, as usual."""
    return target_format(info, args) != (info.sample_rate, info.bits_per_sample)


def wanted_format(args) -> str:
    """The target as the confirmation says it, with the half the user left
    alone named as such rather than filled in with a number it is not."""
    if args.sample_rate and args.bits:
        return audio_format(args.sample_rate, args.bits)
    if args.sample_rate:
        return t("rc_conv_rate", khz=args.sample_rate / 1000)
    return t("rc_conv_bits", bits=args.bits)


#: The divisor that names each of the two sample rate families: 11025 catches
#: 44.1 kHz and its multiples (22.05, 88.2, 176.4), 8000 the 48 kHz ones (16,
#: 24, 32, 96, 192). A rate divisible by neither belongs to no family.
RATE_FAMILIES = (11025, 8000)


def rate_family(rate: int) -> int:
    """Which family a sample rate belongs to, 0 for neither."""
    return next((base for base in RATE_FAMILIES if rate % base == 0), 0)


def conversion_warnings(live: Sequence, args) -> list:
    """Say out loud which conversions can only cost, never gain.

    None of the three is refused - someone may have a reason, and one file in
    a folder must not veto the rest - but all three are worth a line before an
    irreversible write, because each only invents data. Upsampling has nothing
    to put between the samples it already holds. Extra bits have nothing to
    put in them. And crossing between the 44.1 and 48 kHz families recomputes
    every single sample to land somewhere no better than where it started -
    a 44.1 kHz master asked for 48 kHz is the usual way into that.
    """
    up = across = deeper = 0
    for a in live:
        if not needs_conversion(a.info, args):
            continue
        rate, bits = target_format(a.info, args)
        old_rate, old_bits = a.info.sample_rate, a.info.bits_per_sample
        old_family, new_family = rate_family(old_rate), rate_family(rate)
        up += rate > old_rate
        across += (rate != old_rate and old_family and new_family
                   and old_family != new_family)
        deeper += bits > old_bits
    return [t(key, count=files(n)) for n, key in
            ((up, "rc_warn_up"), (across, "rc_warn_family"),
             (deeper, "rc_warn_bits")) if n]


def cmd_reencode(args) -> int:
    """Squeeze weakly encoded files, or with --all everything the report holds.

    The audio is verified sample for sample either way, so there is nothing
    to lose and no backup to keep, and every file that passes is written back
    whatever it did to the size.

    --sample-rate and --bits break that promise on purpose: a file that is not
    in the wanted format goes through ffmpeg instead and comes back as
    different audio. Everything already in the wanted format is re-encoded as
    always, and the original is therefore put aside only for what was really
    converted.

    --downgrade narrows that to the direction that can actually pay: a folder
    of mixed rates asked for 48 kHz has some files above it and some below,
    and only the ones above have anything to give up. The ones below are left
    in their own format instead of being resampled up into invented samples.
    """
    report = require_report(args)
    if report is None:
        return 2
    if args.sample_rate is not None and not (
            args.sample_rate > 0 and rate_is_codable(args.sample_rate)):
        print(t("run_error", problem=t("rc_conv_bad", rate=args.sample_rate)),
              file=sys.stderr)
        return 2
    if args.downgrade and not converting(args):
        print(t("run_error", problem=t("rc_down_alone")), file=sys.stderr)
        return 2

    # The report already lists every file of the folder, the clean ones
    # included - but only as of the last analyze, so --all also picks up what
    # has appeared since.
    added = len(adopt_new_files(report, JOBS)) if args.all else 0
    # A damaged file always carries info.error too, so this one condition
    # drops both the unreadable and the damaged: re-encoding fixes neither,
    # and the MD5 of a file with a lying header cannot even be checked.
    healthy = [a for a in report.items if not a.info.error]
    defective = len(report.items) - len(healthy)
    convert = converting(args)

    def intro(live: Sequence) -> list:
        total = human(sum(a.info.file_size for a in live))
        lines = [t("rc_confirm", count=files(len(live)), total=total)]
        # Asking for a format every live file is already in leaves an ordinary
        # lossless re-encode, which must not be announced as anything else.
        turning = sum(1 for a in live
                      if needs_conversion(a.info, args)) if convert else 0
        if turning:
            lines.append(t("rc_conv_to", format=wanted_format(args)))
            # Only 16 bit is dithered, so only 16 bit may be promised it.
            if any(target_format(a.info, args)[1] == 16 for a in live
                   if needs_conversion(a.info, args)):
                lines.append(t("rc_conv_dither"))
            if turning < len(live):
                lines.append(t("rc_conv_num", count=files(turning)))
        # Said whenever the option is on, converting or not: "nothing to do"
        # after asking for a conversion is a result, and it needs its reason.
        if args.downgrade:
            lines.append(t("rc_down_only"))
            if spared := sum(1 for a in live if would_raise(a.info, args)
                             and not needs_conversion(a.info, args)):
                lines.append(t("rc_down_left", count=files(spared)))
        if args.all:
            lines.append(t("rc_all_any"))
            if added:
                lines.append(t("rc_all_new", count=files(added)))
            if defective:
                lines.append(t("rc_all_skipped", count=files(defective)))
        lines.append(t("rc_settings", opts=" ".join(ENCODE_OPTS)))
        # The usual promise is the opposite of what a conversion does, so the
        # two are never printed together.
        if not turning:
            lines.append(t("rc_promise"))
        else:
            lines += [t("rc_conv_warn"), *conversion_warnings(live, args),
                      original_note(args.keep_original), t("rc_conv_meta")]
        return lines[:1] + ["  " + line for line in lines[1:]]

    def summary(results: Sequence) -> list:
        rows = []
        for status, key in ((Status.DRY_RUN, "sum_done"), (Status.DONE, "sum_done"),
                            (Status.FAILED, "sum_failed")):
            if not (group := [r for r in results if r.status is status]):
                continue
            if status in (Status.DRY_RUN, Status.DONE):
                # A converted file holds different audio than it did, so it is
                # never counted among the re-encoded ones.
                if turned := [r for r in group if r.kind is Kind.CONVERT]:
                    rows.append((t("sum_converted"), len(turned)))
                if kept := [r for r in group if r.kind is not Kind.CONVERT]:
                    rows.append((t(key), len(kept)))
                # A subset fix or a conversion can cost space, so the total may
                # be negative; say "grew by" rather than "saved -416 kB".
                total = sum(r.saved for r in group)
                rows.append((t("sum_saved" if total >= 0 else "sum_grew"),
                             human(abs(total))))
            else:
                rows.append((t(key), len(group)))
        return rows

    def worker(a: Analysis) -> Outcome:
        if convert and needs_conversion(a.info, args):
            return convert_file(a.info, *target_format(a.info, args),
                                args.dry_run, args.keep_original)
        return recompress_file(a.info, args.dry_run)

    if args.all:
        targets = healthy
    elif convert:
        targets = [a for a in healthy
                   if a.weak or needs_conversion(a.info, args)]
    else:
        targets = [a for a in healthy if a.weak]

    return run_write_command(
        args, report, targets, "rc_nothing", intro,
        lambda live: t("rc_running_dry" if args.dry_run else "rc_running",
                       count=files(len(live))),
        worker, summary)


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
                lines.append("  " + t(key, count=files(counts[name])))
        # Salvaging is the only repair that changes the audio, so it is the
        # only one the originals are about.
        if counts[Kind.SALVAGE]:
            lines.append("  " + original_note(args.keep_original))
        return lines

    def repair_one(item: Analysis) -> Outcome:
        if kind[id(item)] is Kind.SUBSET:
            return recompress_file(item.info, args.dry_run,
                                   kind=Kind.SUBSET)
        if kind[id(item)] is Kind.HEADER:
            return patch_stream_header(item.info, args.dry_run)
        return repair_damaged_file(item.info, args.dry_run,
                                   args.keep_original)

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

def cmd_revert_original(args) -> int:
    """Put the originals back, undoing what `reencode` or `repair` wrote.

    The mirror image of `drop-originals`, including where the two disagree.
    That one keeps an original nothing replaced, because it is the last copy
    of that audio; this one restores it there too, because an original with
    nothing in its place is not something to leave alone, it is audio waiting
    to get its name back.

    Each file is put back by a single os.replace, so the name is never
    momentarily empty however the run ends. The header of every original is
    read first: a backup that is not a readable FLAC is no reason at all to
    throw a working file away.
    """
    originals = collect_originals(args.folder)
    if not originals:
        print(t("rev_none", suffix=ORIG_SUFFIX))
        return 0

    revert, broken = [], []
    for path in originals:
        try:
            revert.append((path, read_flac_metadata(path)))
        except (OSError, ValueError) as e:
            broken.append((t("rev_error", name=rel(path, args.folder),
                             error=str(e)), []))
    if broken:
        print_group(t("rev_broken_h", count=files(len(broken))), broken)
    if not revert:
        return 0

    rows, delta = [], 0
    for path, info in revert:
        target = replaced_file(path)
        # What the revert costs, not what it brings back: going home to a
        # 176.4 kHz original from a 48 kHz file wants the space three times
        # over, and that is worth knowing before the disk finds out.
        delta += os.path.getsize(path) - size_of(target)
        # What is about to be lost, said in the same words reencode used to
        # promise it. A file that is there but unreadable gets no line: the
        # header above already says it is going away, and inventing a format
        # for it would be worse than silence.
        detail = []
        if not os.path.exists(target):
            detail = [t("rev_gone")]
        else:
            try:
                now = read_flac_metadata(target)
                detail = [t("rev_change",
                            old=audio_format(now.sample_rate, now.bits_per_sample),
                            new=audio_format(info.sample_rate, info.bits_per_sample))]
            except (OSError, ValueError):
                pass
        rows.append((rel(target, args.folder), detail))

    print(t("rev_confirm", count=files(len(revert)),
            delta=("+" if delta >= 0 else "-") + human(abs(delta))))
    print("  " + t("rev_what", suffix=ORIG_SUFFIX))
    print("  " + t("rev_final"))
    for name, detail in rows:
        print(f"   {name}")
        for line in detail:
            print(f"      {line}")
    if args.dry_run:
        print_table([(t("rev_would"), len(revert))])
        print_advice([t("adv_was_dry")])
        return 0
    if not (args.yes or ask_yes_no()):
        print(t("rc_cancelled"))
        return 1

    done, failed = 0, []
    for path, _ in revert:
        try:
            os.replace(path, replaced_file(path))
            done += 1
        except OSError as e:
            failed.append((t("rev_error", name=rel(path, args.folder),
                             error=str(e)), []))
    print()
    if failed:
        print_group(t("rev_failed_h", count=files(len(failed))), failed)
    print_table([(t("rev_done"), done)])
    # Every restored file is a different file under a name the report still
    # describes, so the report is now wrong about exactly the ones that moved.
    print_advice([t("adv_reverted", cmd=command_hint("analyze", args.folder))]
                 if done else [])
    return 0


def cmd_drop_originals(args) -> int:
    """Delete the originals `repair` and `reencode --keep-original` put aside.

    Only where the file that replaced one is really there. Without it the
    backup is the last copy of that audio, and throwing it away is the one
    thing this script must never do - so those are counted, named and kept.
    """
    originals = collect_originals(args.folder)
    if not originals:
        print(t("drop_none", suffix=ORIG_SUFFIX))
        return 0

    drop, orphans = [], []
    for path in originals:
        (drop if os.path.exists(replaced_file(path)) else orphans).append(path)

    if orphans:
        print_group(t("drop_orphan", count=files(len(orphans))),
                    [(rel(path, args.folder), []) for path in orphans])
    if not drop:
        return 0

    total = sum(os.path.getsize(path) for path in drop)
    print(t("drop_confirm", count=files(len(drop)), total=human(total)))
    print("  " + t("drop_what", suffix=ORIG_SUFFIX))
    print("  " + t("drop_final"))
    for path in drop:
        print(f"   {rel(path, args.folder)}")
    if args.dry_run:
        print_table([(t("drop_would"), len(drop)), (t("drop_freed"), human(total))])
        print_advice([t("adv_was_dry")])
        return 0
    if not (args.yes or ask_yes_no()):
        print(t("rc_cancelled"))
        return 1

    freed, failed = 0, []
    for path in drop:
        try:
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
        except OSError as e:
            failed.append((t("drop_failed", name=rel(path, args.folder),
                             error=str(e)), []))
    print()
    if failed:
        print_group(t("drop_failed_h", count=files(len(failed))), failed)
    print_table([(t("drop_done"), len(drop) - len(failed)),
                 (t("drop_freed"), human(freed))])
    return 0


COMMANDS = {"analyze": cmd_analyze, "show": cmd_show, "inspect": cmd_inspect,
            "reencode": cmd_reencode, "repair": cmd_repair,
            "revert-original": cmd_revert_original,
            "drop-originals": cmd_drop_originals}
NEEDS_FLAC = {"analyze", "inspect", "reencode", "repair"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("folder", help=t("cli_folder"))
    parser.add_argument("--report", metavar="FILE",
                        help=t("cli_report",
                               path=short(os.path.dirname(default_report_path(".")))))
    parser.add_argument("--lang", choices=LANGUAGES, help=t("cli_lang"))


def _add_write_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help=t("cli_dry_run"))
    parser.add_argument("-y", "--yes", action="store_true", help=t("cli_yes"))
    # Both commands can change the audio - reencode by converting, repair by
    # salvaging - so both keep the original by default and both take the one
    # flag that says not to.
    parser.add_argument("--no-keep-original", dest="keep_original",
                        action="store_false",
                        help=t("cli_no_keep_orig", suffix=ORIG_SUFFIX))


def _add_originals_command(subparsers, name: str, help_text: str) -> None:
    """The two commands that only shuffle the originals about.

    Neither reads the report, neither runs `flac`, and neither analyses
    anything - they take a folder, say what they are about to do to files that
    are already on disk, and ask. So they share a command line of their own
    rather than the one the analysing commands use.
    """
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("folder", help=t("cli_folder"))
    parser.add_argument("--lang", choices=LANGUAGES, help=t("cli_lang"))
    parser.add_argument("--dry-run", action="store_true", help=t("cli_dry_run"))
    parser.add_argument("-y", "--yes", action="store_true", help=t("cli_yes"))
    parser.set_defaults(report=None)


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
    _add_common(show)
    show.add_argument("--all", action="store_true", help=t("cli_all"))

    inspect = subparsers.add_parser("inspect", help=t("cli_cmd_inspect"))
    _add_common(inspect)
    inspect.add_argument("--all", action="store_true", help=t("cli_all"))
    inspect.add_argument("--force", action="store_true", help=t("cli_force"))

    reencode = subparsers.add_parser("reencode", help=t("cli_cmd_reencode"))
    _add_common(reencode)
    _add_write_options(reencode)
    reencode.add_argument("--all", action="store_true",
                          help=t("cli_all_reencode"))
    reencode.add_argument("--sample-rate", type=int, metavar="HZ",
                          help=t("cli_sample_rate"))
    reencode.add_argument("--bits", type=int, choices=sorted(RAW_FORMATS),
                          help=t("cli_bits"))
    reencode.add_argument("--downgrade", action="store_true",
                          help=t("cli_downgrade"))

    repair = subparsers.add_parser("repair",
                                   help=t("cli_cmd_repair", suffix=ORIG_SUFFIX))
    _add_common(repair)
    _add_write_options(repair)

    _add_originals_command(subparsers, "revert-original",
                           t("cli_cmd_revert", suffix=ORIG_SUFFIX))
    _add_originals_command(subparsers, "drop-originals",
                           t("cli_cmd_drop", suffix=ORIG_SUFFIX))
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
    args.folder = os.path.abspath(os.path.expanduser(args.folder))

    if not os.path.isdir(args.folder):
        print(t("run_error", problem=t("run_bad_path", root=args.folder)),
              file=sys.stderr)
        return 2
    if problems := check_dependencies(args):
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
