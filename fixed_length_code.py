import os
import math
import json
from collections import Counter
from typing import Dict, Union


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class Code:
    """
    Genera e gestisce codici binari a lunghezza fissa minima.

    La lunghezza fissa ``l`` è la minima tale che  2^l ≥ n,  ossia:

        l = max(1, ceil(log₂(n)))

    dove  n  è il numero di simboli distinti nell'alfabeto sorgente.
    Ad ogni simbolo viene assegnata una codeword di esattamente  l  bit,
    ottenuta dalla rappresentazione binaria del suo indice nell'elenco
    ordinato dei simboli.

    ═══════════════════════════════════════════════════════════════════════════
    INTERFACCIA PUBBLICA
    ═══════════════════════════════════════════════════════════════════════════

    Attributi di istanza (popolati durante la costruzione, da trattare in
    sola lettura dopo l'inizializzazione):

        text            str | None
            Testo sorgente usato per calcolare le frequenze.
            È None quando l'istanza è stata caricata da file JSON.

        source_file     str | None
            Percorso del file sorgente, se il costruttore è stato invocato
            con ``file_path``; None altrimenti.

        frequencies     Dict[str, int]
            Frequenze assolute di ogni simbolo nel testo sorgente.

        probabilities   Dict[str, float]
            Probabilità di ogni simbolo (frequenza / lunghezza totale).

        code            Dict[str, str]
            Codebook: {simbolo → codeword binaria a lunghezza fissa}.

        entropy         float
            Entropia della sorgente  H(X) = Σ pᵢ · log₂(1/pᵢ)  [bit].

        average_length  float
            Lunghezza media del codice  L(c,X) = Σ pᵢ · lᵢ  [bit/simbolo].
            Per un codice a lunghezza fissa coincide con  l  (la lunghezza
            fissa comune a tutti i simboli).

        kraft_sum       float
            Valore della somma di Kraft  Σ 2^(−lᵢ).
            Per un codice a lunghezza fissa vale  n / 2^l  ≤ 1.

        coding_type     str
            Tipo di algoritmo di codifica.
            Valore fisso: ``'Fixed-length coding'``.

        code_file       str | None
            Percorso del file JSON che contiene (o da cui è stato caricato) il
            codebook. Vale None finché non viene chiamato write_code() o
            read_code().

    ───────────────────────────────────────────────────────────────────────────
    Costruttori
    ───────────────────────────────────────────────────────────────────────────

        Code(source=<str>)
        Code(file_path=<str>)
            Esattamente uno dei due parametri deve essere fornito.
            Calcola frequenze, probabilità, codebook e metriche derivate.

        Code.load_code(code_file) → Code                          [classmethod]
            Ricrea un'istanza a partire da un file JSON prodotto da write_code().
            Il testo originale non viene ripristinato (``text`` sarà None).
            Imposta ``code_file`` con il percorso del file letto.

    ───────────────────────────────────────────────────────────────────────────
    Codifica / Decodifica
    ───────────────────────────────────────────────────────────────────────────

        encode(text) → str
            Codifica una stringa restituendo la sequenza di bit come stringa.

        encode_file(input_file, output_file, save_as_binary=False) → dict
            Legge un file, lo codifica e salva il risultato su disco.
            Restituisce un dizionario con le statistiche dell'operazione:
                {
                  "input_file"       : str,
                  "output_file"      : str,
                  "original_chars"   : int,
                  "encoded_bits"     : int,
                  "encoded_bytes"    : int | None,  # solo in modalità binaria
                  "padding_bits"     : int | None,  # solo in modalità binaria
                  "format"           : "binary" | "text"
                }

        decode(binary_string) → str
            Decodifica una stringa di bit restituendo il testo originale.

        decode_file(input_file, output_file, is_binary=False) → dict
            Legge un file codificato (testo o binario), lo decodifica e salva
            il risultato su disco.
            Restituisce un dizionario con le statistiche dell'operazione:
                {
                  "input_file"     : str,
                  "output_file"    : str,
                  "decoded_chars"  : int,
                  "format"         : "binary" | "text"
                }

    ───────────────────────────────────────────────────────────────────────────
    Metodi informativi / diagnostici
    ───────────────────────────────────────────────────────────────────────────

        introduce_yourself()
            Stampa il tipo di codifica e, se disponibile, il percorso del file
            JSON in cui il codebook è memorizzato.

        print_code()
            Stampa la tabella del codebook prodotta da code_table().

        code_table() → str
            Restituisce (senza stampare) una tabella formattata che riporta,
            per ogni simbolo: codeword, probabilità, auto-informazione hᵢ,
            lunghezza lᵢ; più entropia, lunghezza media, efficienza e overhead.

        kraft_check() → dict
            Verifica la disuguaglianza di Kraft e restituisce:
                {
                  "kraft_sum"  : float,   # Σ 2^(−lᵢ)
                  "satisfied"  : bool     # True se ≤ 1
                }

        write_code(code_file) → str
            Serializza codebook e probabilità in un file JSON.
            Aggiorna ``self.code_file`` e restituisce il percorso del file.

        read_code(code_file)
            Carica codebook e probabilità da un file JSON nell'istanza corrente,
            sovrascrivendo i valori esistenti e aggiornando tutte le metriche.
            Aggiorna ``self.code_file`` con il percorso letto.

    ═══════════════════════════════════════════════════════════════════════════
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Costruzione
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        source:    Union[str, None] = None,
        file_path: Union[str, None] = None,
    ):
        """
        Inizializza il codice a partire da un testo diretto o da un file.

        Args:
            source:    Testo da cui estrarre le frequenze.
            file_path: Percorso del file di testo da cui estrarre le frequenze.

        Raises:
            ValueError:        Se non viene fornito alcun input, o vengono
                               forniti entrambi.
            FileNotFoundError: Se ``file_path`` non esiste sul filesystem.
            IOError:           Se la lettura del file fallisce.
        """
        if source is None and file_path is None:
            raise ValueError("Fornire esattamente uno tra 'source' e 'file_path'.")
        if source is not None and file_path is not None:
            raise ValueError("Fornire solo uno tra 'source' e 'file_path'.")

        if file_path is not None:
            self.text        = self._read_file(file_path)
            self.source_file = file_path
        else:
            self.text        = source
            self.source_file = None

        # Tipo di algoritmo (costante di istanza)
        self.coding_type : str       = "Fixed-length coding"
        # Percorso del file JSON associato al codebook (None finché non si
        # chiama write_code / read_code)
        self.code_file   : str | None = None

        # Attributi calcolati e memorizzati una sola volta
        self.frequencies    : Dict[str, int]   = self._calculate_frequencies()
        self.probabilities  : Dict[str, float] = self._calculate_probabilities()
        self.code           : Dict[str, str]   = self._fixed_length_coding()
        self.entropy        : float            = self._compute_entropy()
        self.average_length : float            = self._compute_average_length()
        self.kraft_sum      : float            = self._compute_kraft_sum()

    # ─────────────────────────────────────────────────────────────────────────
    # Metodi privati: I/O e calcolo
    # ─────────────────────────────────────────────────────────────────────────

    def _read_file(self, file_path: str) -> str:
        """Legge e restituisce il contenuto di un file di testo UTF-8."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            raise IOError(f"Errore nella lettura del file: {exc}") from exc

    def _calculate_frequencies(self) -> Dict[str, int]:
        """Conta le occorrenze di ogni simbolo nel testo."""
        return dict(Counter(self.text))

    def _calculate_probabilities(self) -> Dict[str, float]:
        """Calcola la probabilità di ogni simbolo (frequenza / totale)."""
        total = len(self.text)
        return {sym: cnt / total for sym, cnt in self.frequencies.items()}

    def _fixed_length_coding(self) -> Dict[str, str]:
        """
        Costruisce il codebook a lunghezza fissa minima.

        Algoritmo:
            1. Determina  n = numero di simboli distinti.
            2. Calcola la lunghezza fissa minima:
                   l = max(1, ceil(log₂(n)))
               Con l bit si possono rappresentare fino a 2^l ≥ n simboli.
            3. Ordina i simboli in modo deterministico (ordine lessicografico).
            4. Assegna ad ogni simbolo la rappresentazione binaria del suo
               indice (0-based) su esattamente l bit.

        Returns:
            Dizionario {simbolo: codeword}.
        """
        if not self.frequencies:
            return {}

        symbols = sorted(self.frequencies.keys())
        n       = len(symbols)

        # Lunghezza fissa minima: ceil(log2(n)), almeno 1 bit
        bit_length = max(1, math.ceil(math.log2(n))) if n > 1 else 1

        return {sym: format(idx, f"0{bit_length}b")
                for idx, sym in enumerate(symbols)}

    def _compute_entropy(self) -> float:
        """Calcola H(X) = Σ pᵢ · log₂(1/pᵢ)."""
        return sum(
            p * math.log2(1.0 / p)
            for p in self.probabilities.values()
            if p > 0
        )

    def _compute_average_length(self) -> float:
        """Calcola L(c,X) = Σ pᵢ · lᵢ.

        Per un codice a lunghezza fissa tutte le lᵢ sono uguali, quindi
        il risultato coincide con la lunghezza fissa comune.
        """
        return sum(
            prob * len(self.code[sym])
            for sym, prob in self.probabilities.items()
        )

    def _compute_kraft_sum(self) -> float:
        """Calcola Σ 2^(−lᵢ) per tutte le codeword del codebook.

        Per un codice a lunghezza fissa  l  su  n  simboli vale:
            Σ 2^(−l) = n / 2^l  ≤ 1
        """
        return sum(2 ** (-len(cw)) for cw in self.code.values())

    # ─────────────────────────────────────────────────────────────────────────
    # Metodi pubblici: codifica / decodifica
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> str:
        """
        Codifica una stringa tramite il codebook a lunghezza fissa.

        Args:
            text: Testo da codificare.

        Returns:
            Stringa di bit (es. '010110...').

        Raises:
            ValueError: Se il testo contiene simboli assenti nel codebook.
        """
        parts = []
        for char in text:
            if char not in self.code:
                raise ValueError(f"Simbolo '{char}' non presente nel codice.")
            parts.append(self.code[char])
        return "".join(parts)

    def encode_file(
        self,
        input_file:     str,
        output_file:    str,
        save_as_binary: bool = False,
    ) -> dict:
        """
        Legge un file di testo, lo codifica e salva il risultato su disco.

        Args:
            input_file:     Percorso del file di testo da codificare.
            output_file:    Percorso del file di output.
            save_as_binary: Se True salva in formato binario compatto (byte),
                            altrimenti salva come stringa di '0'/'1'.

        Returns:
            Dizionario con le statistiche dell'operazione:
            ``original_chars``, ``encoded_bits``, ``format``,
            più ``encoded_bytes`` e ``padding_bits`` in modalità binaria.
        """
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        encoded = self.encode(text)
        stats   = {
            "input_file"    : input_file,
            "output_file"   : output_file,
            "original_chars": len(text),
            "encoded_bits"  : len(encoded),
        }

        if save_as_binary:
            padding        = (8 - len(encoded) % 8) % 8
            encoded_padded = encoded + "0" * padding
            byte_array     = bytearray(
                int(encoded_padded[i : i + 8], 2)
                for i in range(0, len(encoded_padded), 8)
            )
            with open(output_file, "wb") as f:
                f.write(bytes([padding]))
                f.write(bytes(byte_array))
            stats.update({
                "encoded_bytes": len(byte_array) + 1,
                "padding_bits" : padding,
                "format"       : "binary",
            })
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(encoded)
            stats.update({
                "encoded_bytes": None,
                "padding_bits" : None,
                "format"       : "text",
            })

        return stats

    def decode(self, binary_string: str) -> str:
        """
        Decodifica una stringa di bit tramite il codebook a lunghezza fissa.

        Poiché tutte le codeword hanno la stessa lunghezza  l,  la
        decodifica si riduce a spezzare la stringa in blocchi di  l  bit
        e cercare ognuno nella tabella inversa.

        Args:
            binary_string: Sequenza di bit da decodificare (es. '010110').

        Returns:
            Testo decodificato.

        Raises:
            ValueError: Se la lunghezza della stringa non è multipla di  l,
                        o se un blocco non corrisponde ad alcun simbolo.
        """
        if not self.code:
            return ""

        # Tutti i codici hanno la stessa lunghezza; prendiamo quella del primo
        l            = len(next(iter(self.code.values())))
        reverse_code = {cw: sym for sym, cw in self.code.items()}

        if len(binary_string) % l != 0:
            raise ValueError(
                f"La lunghezza della stringa binaria ({len(binary_string)}) "
                f"non è multipla della lunghezza fissa del codice ({l})."
            )

        decoded = []
        for i in range(0, len(binary_string), l):
            block = binary_string[i : i + l]
            if block not in reverse_code:
                raise ValueError(
                    f"Blocco '{block}' non presente nel codebook."
                )
            decoded.append(reverse_code[block])

        return "".join(decoded)

    def decode_file(
        self,
        input_file:  str,
        output_file: str,
        is_binary:   bool = False,
    ) -> dict:
        """
        Decodifica un file codificato (testo o binario) e salva il risultato.

        Args:
            input_file:  Percorso del file codificato.
            output_file: Percorso del file di output.
            is_binary:   Se True legge il file come binario compatto,
                         altrimenti come stringa di '0'/'1'.

        Returns:
            Dizionario con le statistiche dell'operazione:
            ``input_file``, ``output_file``, ``decoded_chars``, ``format``.
        """
        if is_binary:
            with open(input_file, "rb") as f:
                data = f.read()
            padding       = data[0]
            binary_string = "".join(format(b, "08b") for b in data[1:])
            if padding:
                binary_string = binary_string[:-padding]
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                binary_string = f.read().strip()

        decoded_text = self.decode(binary_string)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(decoded_text)

        return {
            "input_file"   : input_file,
            "output_file"  : output_file,
            "decoded_chars": len(decoded_text),
            "format"       : "binary" if is_binary else "text",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Metodi pubblici: informativi / diagnostici
    # ─────────────────────────────────────────────────────────────────────────

    def introduce_yourself(self) -> None:
        """
        Stampa il tipo di codifica e, se disponibile, il percorso del file JSON
        in cui il codebook è memorizzato.

        Esempio di output::

            Tipo di codifica : Fixed-length coding
            File JSON        : codebook.json
        """
        print(f"Tipo di codifica : {self.coding_type}")
        if self.code_file is not None:
            print(f"File JSON        : {self.code_file}")
        else:
            print("File JSON        : non ancora associato (usa write_code() per salvarlo)")

    def print_code(self) -> None:
        """
        Stampa la tabella del codebook generata da :meth:`code_table`.

        Equivale a ``print(self.code_table())``: delegare la produzione della
        stringa a ``code_table()`` garantisce che la logica di formattazione
        risieda in un unico punto.
        """
        print(self.code_table())

    def code_table(self) -> str:
        """
        Costruisce e restituisce una tabella testuale formattata del codebook.

        La tabella riporta, per ogni simbolo (ordinato per probabilità
        decrescente): codeword, probabilità, auto-informazione hᵢ = log₂(1/pᵢ),
        lunghezza lᵢ.  In coda: entropia H(X), lunghezza media L(c,X),
        efficienza e overhead.

        Returns:
            Stringa multi-riga pronta per essere stampata o salvata.
        """
        n          = len(self.code)
        bit_length = len(next(iter(self.code.values()))) if self.code else 0

        SEP  = "=" * 70
        DASH = "-" * 70
        lines = [
            "",
            SEP,
            f"TABELLA DEL CODICE BINARIO (Fixed-length  l={bit_length} bit,  n={n} simboli)",
            SEP,
            f"{'Simbolo':<10} {'Codice':<15} {'Prob':<12} {'h_i':<12} {'l_i':<8}",
            DASH,
        ]

        for symbol, prob in sorted(
            self.probabilities.items(), key=lambda x: x[1], reverse=True
        ):
            codeword = self.code[symbol]
            h_i      = math.log2(1.0 / prob) if prob > 0 else 0.0
            l_i      = len(codeword)
            display  = repr(symbol) if symbol in "\n\t " else symbol
            lines.append(
                f"{display:<10} {codeword:<15} {prob:<12.4f} "
                f"{h_i:<12.2f} {l_i:<8}"
            )

        efficiency = (self.entropy / self.average_length * 100) if self.average_length > 0 else 0.0
        overhead   = self.average_length - self.entropy

        lines += [
            DASH,
            f"\nEntropia H(X)          = {self.entropy:.4f} bit",
            f"Lunghezza fissa l        = {bit_length} bit",
            f"Lunghezza media L(c,X)   = {self.average_length:.4f} bit",
            f"Efficienza               = {efficiency:.2f}%",
            f"Overhead                 = {overhead:.4f} bit",
            SEP + "\n",
        ]
        return "\n".join(lines)

    def kraft_check(self) -> dict:
        """
        Verifica la disuguaglianza di Kraft  Σ 2^(−lᵢ) ≤ 1.

        Per un codice a lunghezza fissa  l  su  n  simboli il valore è
        esattamente  n / 2^l,  che è ≤ 1 per costruzione.

        Returns:
            Dizionario ``{"kraft_sum": float, "satisfied": bool}``.
        """
        return {
            "kraft_sum" : self.kraft_sum,
            "satisfied" : self.kraft_sum <= 1.0,
        }

    def write_code(self, code_file: str) -> str:
        """
        Serializza codebook, probabilità e metadato sorgente in un file JSON.
        Aggiorna ``self.code_file`` con il percorso del file scritto.

        Args:
            code_file: Percorso del file di destinazione.

        Returns:
            Percorso del file salvato (uguale a ``code_file``).
        """
        payload = {
            "code"         : self.code,
            "probabilities": self.probabilities,
            "source_file"  : str(self.source_file) if self.source_file else None,
        }
        with open(code_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.code_file = code_file
        return code_file

    def read_code(self, code_file: str) -> None:
        """
        Carica codebook e probabilità da un file JSON nell'istanza corrente,
        sovrascrivendo i valori esistenti e ricalcolando tutte le metriche
        derivate (entropia, lunghezza media, somma di Kraft).
        Aggiorna ``self.code_file`` con il percorso letto.

        Args:
            code_file: Percorso del file JSON prodotto da :meth:`write_code`.

        Raises:
            FileNotFoundError: Se ``code_file`` non esiste.
            KeyError:          Se il file JSON non contiene i campi attesi.
        """
        with open(code_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.code          = data["code"]
        self.probabilities = data["probabilities"]
        self.source_file   = data.get("source_file")
        self.text          = None
        self.frequencies   = {s: int(p * 1000) for s, p in self.probabilities.items()}
        self.entropy        = self._compute_entropy()
        self.average_length = self._compute_average_length()
        self.kraft_sum      = self._compute_kraft_sum()
        self.code_file      = code_file

    # ─────────────────────────────────────────────────────────────────────────
    # Costruttore alternativo
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load_code(cls, code_file: str) -> "Code":
        """
        Ricrea un'istanza ``Code`` a partire da un file JSON prodotto da
        :meth:`write_code`, senza richiedere il testo sorgente originale.

        Args:
            code_file: Percorso del file JSON.

        Returns:
            Istanza ``Code`` pronta per encode/decode, con ``code_file``
            già impostato al percorso letto.

        Note:
            ``text`` sarà ``None``; ``frequencies`` è un'approssimazione intera
            ricavata dalle probabilità salvate.
        """
        with open(code_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance                = cls.__new__(cls)
        instance.coding_type    = "Fixed-length coding"
        instance.code_file      = code_file
        instance.code           = data["code"]
        instance.probabilities  = data["probabilities"]
        instance.source_file    = data.get("source_file")
        instance.text           = None
        instance.frequencies    = {s: int(p * 1000) for s, p in instance.probabilities.items()}
        instance.entropy        = instance._compute_entropy()
        instance.average_length = instance._compute_average_length()
        instance.kraft_sum      = instance._compute_kraft_sum()
        return instance
