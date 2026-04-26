#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esempio_codice.py
================
Script dimostrativo per il modulo ``codes`` e la classe ``Code``.

Esegue cinque esempi progressivi che coprono:
    1. Codifica da stringa in memoria
    2. Analisi delle frequenze a partire da un file
    3. Compressione e decompressione di un file binario
    4. Serializzazione/deserializzazione del codice via JSON
    5. Confronto dell'efficienza su distribuzioni di probabilità diverse

Requisiti:
    - Python >= 3.10
    - Modulo ``codes`` accessibile nella stessa directory di questo script

Utilizzo::

    python esempio_codice.py [--keep-files]

    --keep-files  Se specificato, i file temporanei NON vengono eliminati
                  al termine dell'esecuzione (utile per ispezione manuale).
"""

# ---------------------------------------------------------------------------
# Import della libreria standard
# ---------------------------------------------------------------------------

import sys
import os
import logging
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Configurazione del logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Costanti di configurazione
# ---------------------------------------------------------------------------

_SCRIPT_DIR: Path = Path(__file__).resolve().parent

FILE_TESTO:       str = "esempio_testo.txt"
FILE_INPUT:       str = "esempio_input.txt"
FILE_COMPRESSO:   str = "esempio_compresso.bin"
FILE_DECOMPRESSO: str = "esempio_decompresso.txt"
FILE_CODICE:      str = "esempio_codice.json"

TESTO_ESEMPIO_1: str = (
    "Ho fatto la parte pratica del concorso per il Capes in un liceo di Lione, "
    "sulla collina della Croix-Rousse."
)

TESTO_ESEMPIO_2: str = (
    "Quando il signor Bilbo Baggins di Casa Baggins annunziò che avrebbe presto "
    "festeggiato il suo centoundicesimo compleanno con una festa oltremodo fastosa, "
    "i commenti e i fermenti a Hobbiton si sprecarono."
)

DISTRIBUZIONI: dict[str, str] = {
    "Uniforme":          "abcdefghijklmnop" * 10,
    "Molto sbilanciata": "a" * 80 + "b" * 40 + "c" * 20 + "defghijk",
    "Testo naturale":    "la compressione sfrutta le ridondanze del linguaggio naturale",
}


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def setup_python_path() -> None:
    """Aggiunge la directory dello script al ``sys.path``."""
    script_dir_str: str = str(_SCRIPT_DIR)
    if script_dir_str not in sys.path:
        sys.path.insert(0, script_dir_str)
        logger.debug("Aggiunto al sys.path: %s", script_dir_str)


def parse_args() -> argparse.Namespace:
    """Analizza gli argomenti passati da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Script dimostrativo dei moduli codes.",
    )
    parser.add_argument(
        "class_name",               # argomento posizionale
        default='FIXED_LENGTH',      # valore di default se non specificato
        help="Nome della classe da istanziare (es. Auto)",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        default=False,
        help=(
            "Se specificato, i file temporanei generati dallo script "
            "NON vengono eliminati al termine dell'esecuzione."
        ),
    )
    return parser.parse_args()


def cleanup_files(*file_paths: str) -> None:
    """Elimina i file temporanei generati durante gli esempi."""
    for path in file_paths:
        try:
            Path(path).unlink(missing_ok=True)
            logger.debug("File eliminato: %s", path)
        except OSError as exc:
            logger.warning("Impossibile eliminare '%s': %s", path, exc)


def separatore(titolo: str, larghezza: int = 70) -> None:
    """Stampa un separatore visivo con titolo centrato."""
    print("\n" + "=" * larghezza)
    print(titolo.center(larghezza))
    print("=" * larghezza)


# ---------------------------------------------------------------------------
# Funzioni per ciascun esempio
# ---------------------------------------------------------------------------

def codifica_da_stringa(Coder) -> None:
    """**Esempio 1** — Codifica a partire da una stringa.

    Costruisce un ``Code`` analizzando le frequenze del testo sorgente,
    stampa la tabella del codebook, codifica e decodifica un messaggio
    e verifica la correttezza del ciclo encode/decode.

    Args:
        Coder: La classe ``Code`` importata dal modulo ``codes``.
    """
    separatore("1. CODIFICA DA STRINGA")

    #print(f"\nTesto sorgente: '{TESTO_ESEMPIO_1}'")

    #coder = Coder(source=TESTO_ESEMPIO_1)
    coder = Coder(file_path=FILE_TESTO)
    coder.write_code(FILE_CODICE)  # Salva il codebook in JSON per ispezione

    # introduce_yourself() stampa tipo di codifica e file JSON associato
    coder.introduce_yourself()

    # print_code() stampa la tabella simbolo → codeword prodotta da code_table()
    coder.print_code()

    # Tutti i caratteri devono essere nel testo sorgente; si usa una parola
    # ricavata dal sorgente stesso per garantire la copertura dell'alfabeto
    msg = f"Testo sorgente: '{TESTO_ESEMPIO_1}'"

    try:
        encoded = coder.encode(msg)
        decoded = coder.decode(encoded)
        print(f"\nMessaggio originale : '{msg}'")
        print(f"Sequenza codificata : {encoded}")
        print(f"Lunghezza           : {len(encoded)} bit")
        print(f"Messaggio decodific.: '{decoded}'")
        print(f"Verifica OK         : {msg == decoded}")
    except ValueError as exc:
        print(f"  ✗ Errore: simboli fuori alfabeto: {exc}")


def costruisci_coder_da_file(Coder) -> "Coder":
    """**Esempio 2** — Generazione del codice analizzando un file di testo.

    Costruisce un ``Code`` dal path di un file, stampa le metriche teoriche
    dell'informazione (entropia, lunghezza media, efficienza), verifica la
    disuguaglianza di Kraft e serializza il codebook in JSON.

    Variazioni rispetto alla versione precedente:

    - ``coder.entropy`` e ``coder.average_length`` sono attributi di istanza
      (non più metodi): vengono letti direttamente senza ``()``.
    - ``verify_kraft_inequality()`` è sostituito da ``kraft_check()``, che
      restituisce un dizionario; la stampa è delegata a questo livello.
    - ``save_code()`` è sostituito da ``write_code()``, che aggiorna
      ``coder.code_file`` e restituisce il percorso; la stampa è qui.

    Args:
        Coder: La classe ``Code`` importata dal modulo ``codes``.

    Returns:
        Istanza ``Code`` costruita sul file, da riutilizzare negli esempi
        successivi.
    """
    separatore("2. CODIFICA DA FILE")

    coder = Coder(file_path=FILE_TESTO)

    # --- Metriche teoriche ---
    # .entropy e .average_length sono attributi calcolati in __init__,
    # non metodi: non servono le parentesi ()
    print(f"Caratteri totali nel file : {len(coder.text)}")
    print(f"Simboli unici (alfabeto)  : {len(coder.code)}")
    print(f"Entropia H(X)             : {coder.entropy:.4f} bit/simbolo")
    print(f"Lunghezza media L(c,X)    : {coder.average_length:.4f} bit/simbolo")
    efficienza: float = (coder.entropy / coder.average_length) * 100
    print(f"Efficienza H/L            : {efficienza:.2f}%")

    # --- Verifica disuguaglianza di Kraft ---
    # kraft_check() restituisce {"kraft_sum": float, "satisfied": bool};
    # la stampa (prima interna a verify_kraft_inequality) è ora qui
    kraft = coder.kraft_check()
    print(f"\nVerifica disuguaglianza di Kraft:")
    print(f"  Σ 2^(-lᵢ) = {kraft['kraft_sum']:.6f}")
    print(f"  Condizione soddisfatta: {kraft['satisfied']}")

    # --- Serializzazione del codebook ---
    # write_code() sostituisce save_code(): aggiorna coder.code_file
    # e restituisce il percorso del file scritto; la conferma è stampata qui
    saved_path = coder.write_code(FILE_CODICE)
    print(f"\nCodice salvato in: '{saved_path}'")

    return coder


def compressione_file(coder) -> None:
    """**Esempio 3** — Compressione e decompressione di un file binario.

    Esegue il ciclo completo encode → decode su file e verifica l'integrità
    lossless del risultato. Stampa le statistiche di compressione.

    Variazioni rispetto alla versione precedente:

    - ``encode_file()`` restituisce un dizionario di statistiche
      (``original_chars``, ``encoded_bits``, ``encoded_bytes``,
      ``padding_bits``, ``format``) invece di stampare direttamente;
      la stampa è delegata a questo livello.
    - ``decode_file()`` restituisce analogamente un dizionario
      (``decoded_chars``, ``format``); la stampa è delegata qui.

    Args:
        coder: Istanza ``Code`` già inizializzata (da Esempio 2).
    """
    separatore("3. COMPRESSIONE E DECOMPRESSIONE FILE")

    # --- Fase di compressione ---
    print(f"\nCompressione: '{FILE_INPUT}' → '{FILE_COMPRESSO}'")

    # encode_file() restituisce le statistiche dell'operazione; in passato
    # le stampava internamente — ora quella responsabilità è qui
    enc_stats = coder.encode_file(FILE_INPUT, FILE_COMPRESSO, save_as_binary=True)

    print(f"  File codificato salvato in formato binario: {enc_stats['output_file']}")
    print(f"  Dimensione originale : {enc_stats['original_chars']} caratteri")
    print(f"  Byte scritti         : {enc_stats['encoded_bytes']} byte")
    print(f"  Bit effettivi        : {enc_stats['encoded_bits']}  "
          f"(padding: {enc_stats['padding_bits']} bit)")

    # --- Fase di decompressione ---
    print(f"\nDecompressione: '{FILE_COMPRESSO}' → '{FILE_DECOMPRESSO}'")

    # decode_file() restituisce le statistiche; stampa delegata qui
    dec_stats = coder.decode_file(FILE_COMPRESSO, FILE_DECOMPRESSO, is_binary=True)

    print(f"  File decodificato salvato: {dec_stats['output_file']}")
    print(f"  Lunghezza testo ricostruito: {dec_stats['decoded_chars']} caratteri")

    # --- Verifica di integrità ---
    try:
        with open(FILE_INPUT,      "r", encoding="utf-8") as f: originale    = f.read()
        with open(FILE_DECOMPRESSO,"r", encoding="utf-8") as f: ripristinato = f.read()
    except IOError as exc:
        logger.error("Errore durante la verifica di integrità: %s", exc)
        raise

    integrita_ok: bool = originale == ripristinato
    print(f"\nVerifica integrità (originale == decompresso): {integrita_ok}")
    if not integrita_ok:
        logger.warning("ATTENZIONE: il file decompresso differisce dall'originale!")

    # --- Statistiche di compressione ---
    dim_originale: int = os.path.getsize(FILE_INPUT)
    dim_compresso: int = os.path.getsize(FILE_COMPRESSO)
    rapporto: float    = (dim_compresso / dim_originale) * 100

    print("\nStatistiche di compressione:")
    print(f"  Dimensione originale  : {dim_originale} byte")
    print(f"  Dimensione compressa  : {dim_compresso} byte")
    print(f"  Rapporto di compr.    : {rapporto:.1f}%  ({100 - rapporto:.1f}% di risparmio)")

    # --- Confronto con ASCII (baseline 8 bit/carattere) ---
    # enc_stats['encoded_bits'] contiene i bit utili già calcolati da encode_file()
    bit_ascii:     int = enc_stats['original_chars'] * 8
    bit_compressi: int = enc_stats['encoded_bits']

    print("\nConfronto con codifica ASCII (8 bit/carattere):")
    print(f"  ASCII (8 bit fissi)   : {bit_ascii} bit")
    print(f"  Huffman               : {bit_compressi} bit")
    print(f"  Rapporto              : {bit_compressi / bit_ascii * 100:.1f}%")


def esempio_4_riutilizzo_codice(Coder) -> None:
    """**Esempio 4** — Ricaricamento e riutilizzo di un codice serializzato.

    Deserializza il codebook dal file JSON prodotto nell'Esempio 2 tramite
    il costruttore alternativo ``Code.load_code()``, poi testa la codifica
    e decodifica di un messaggio arbitrario.

    Args:
        Coder: La classe ``Code`` importata dal modulo ``codes``.
    """
    separatore("4. RIUTILIZZO CODICE SALVATO")

    print(f"\nCaricamento del codice da: '{FILE_CODICE}'")

    # load_code() è un classmethod: restituisce una nuova istanza Code
    # con code_file già impostato al percorso letto
    coder3 = Coder.load_code(FILE_CODICE)

    # introduce_yourself() stampa tipo di codifica + file JSON associato
    coder3.introduce_yourself()

    new_msg: str = "Shannon ha dimostrato"
    print(f"\nTest codifica/decodifica per: '{new_msg}'")

    try:
        enc: str = coder3.encode(new_msg)
        dec: str = coder3.decode(enc)
        print(f"  ✓ Sequenza codificata  : {len(enc)} bit")
        print(f"  ✓ Decodifica corretta  : {new_msg == dec}")
    except ValueError as exc:
        print(f"  ✗ Errore di codifica: {exc}")
        logger.warning(
            "Il messaggio '%s' contiene simboli non previsti dal codice: %s",
            new_msg, exc,
        )


def esempio_5_confronto_distribuzioni(Coder) -> None:
    """**Esempio 5** — Confronto dell'efficienza su diverse distribuzioni.

    Per ciascuna distribuzione costruisce un ``Code`` e ne stampa le metriche
    in forma tabellare: H(X), L(c,X), efficienza e overhead.

    Variazione rispetto alla versione precedente:

    - ``c.entropy()`` e ``c.average_length()`` erano chiamate a metodo;
      ora sono attributi di istanza letti senza ``()``.

    Args:
        Coder: La classe ``Code`` importata dal modulo ``codes``.
    """
    separatore("5. CONFRONTO TRA DIVERSE DISTRIBUZIONI")

    print(
        f"\n{'Distribuzione':<20} {'H(X)':<10} {'L(c,X)':<10} "
        f"{'Efficienza':<12} {'Overhead':<10}"
    )
    print("-" * 70)

    for nome, testo in DISTRIBUZIONI.items():
        c = Coder(source=testo)

        # .entropy e .average_length: attributi di istanza, non metodi
        H   = c.entropy
        L   = c.average_length
        eff = (H / L) * 100
        ovh = L - H

        print(
            f"{nome:<20} {H:<10.4f} {L:<10.4f} {eff:<12.2f}% {ovh:<10.4f}"
        )


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
import importlib
def main() -> int:
    """Orchestra l'esecuzione dei cinque esempi.

    Returns:
        int: 0 = successo, 1 = errore di import.
    """
    coding_types = {
    "FIXED_LENGTH": "fixed_length_code",
    "SHANNON_FANO": "shannon_fano_code",
    "HUFFMAN": "huffman_code",
    "ARITHMETIC": "arithmetic_code",
    "LEMPEL_ZIV": "lzw_code"
    }

    args = parse_args()
    setup_python_path()

    coding_type = args.class_name.upper()
    logger.info("Classe di codifica specificata: %s", coding_type)

    module_name = coding_types.get(coding_type)

    if module_name is None:
        raise ValueError(
            f"Tipo di codifica non riconosciuto: '{coding_type}'. "
            f"Valori validi: {list(coding_types.keys())}"
        )

    try:
        module = importlib.import_module(f"{module_name}")
        Coder = getattr(module, "Code")
    except ImportError as exc:
        raise ImportError(
            f"Impossibile importare il modulo '{module_name}': {exc}"
        ) from exc
    except AttributeError as exc:
        raise AttributeError(
            f"Il modulo '{module_name}' non contiene la classe 'Code': {exc}"
        ) from exc
        return 1

    

    # --- Esempio 1: codifica in memoria da stringa ---
    # Coder viene passato esplicitamente: codifica_da_stringa non ha accesso
    # a variabili di modulo e riceve la classe come argomento
    
    codifica_da_stringa(Coder)

    # --- Esempio 2: codifica da file + metriche + serializzazione ---

    # print_code() sostituisce print_code_table(): stampa la tabella
    # prodotta da code_table() senza duplicare la logica di formattazione
    coder = costruisci_coder_da_file(Coder)
    coder.print_code()

    # --- Esempio 3: compressione binaria e verifica di integrità ---
    compressione_file(coder)

    # --- Esempio 4: deserializzazione del codice e test encode/decode ---
    esempio_4_riutilizzo_codice(Coder)

    # --- Esempio 5: tabella comparativa su distribuzioni diverse ---
    esempio_5_confronto_distribuzioni(Coder)

    # --- Riepilogo finale ---
    separatore("ESEMPI COMPLETATI")

    file_generati = [FILE_COMPRESSO, FILE_DECOMPRESSO, FILE_CODICE]
    print("\nFile generati durante l'esecuzione:")
    for fp in file_generati:
        if Path(fp).exists():
            size = Path(fp).stat().st_size
            print(f"  - {fp}  ({size} byte)")
        else:
            print(f"  - {fp}  [non trovato]")

    print("\nPer la documentazione completa del modulo:")
    print("  cat codes/README.md")

    # --- Pulizia file temporanei ---
    if args.keep_files:
        print("\n[INFO] Flag --keep-files attivo: i file temporanei NON sono stati eliminati.")
    else:
        print("\nPulizia dei file temporanei...")
        cleanup_files(*file_generati)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
