"""
Modulo per l'analisi dell'entropia di testi e la generazione di testo basata su modelli probabilistici.

Questo modulo fornisce strumenti per:
- Analizzare l'entropia di un testo basandosi su statistiche di parole e prefissi
- Calcolare probabilità condizionate di parole dato un contesto
- Generare testo usando un modello probabilistico basato sui dati analizzati

Differenze rispetto alla versione basata su caratteri:
- L'unità di base è la parola (token), non il carattere
- La pulizia del testo rimuove i caratteri non alfabetici ma mantiene gli accenti
- I prefissi sono sequenze di k parole (tuple)
- La generazione produce sequenze di parole
- I report mostrano solo valori derivati (entropia, entropia condizionata media),
  non distribuzioni per singoli elementi del lessico

Modalità speciale k=-1 (modello uniforme):
- Tutte le parole del lessico hanno probabilità equiprobabile 1/|V|
- L'entropia è massima: H(W) = log₂(|V|)
- Il contesto non viene utilizzato; la generazione campiona uniformemente sul lessico
- Rappresenta il caso di riferimento (baseline) con massima incertezza

Classe principale:
    LanguageModel: Modello di linguaggio che include analisi entropica e generazione di testo

Classe interna:
    LanguageModel.TextEntropyAnalyzer: Analizzatore di entropia (embedded in LanguageModel)
"""

import math
import random
import re
from collections import defaultdict
from pathlib import Path


class LanguageModel:
    """
    Modello di linguaggio basato su distribuzioni di probabilità condizionate a livello di parola.

    Questa classe integra l'analisi entropica di un testo e la generazione di nuovo testo.
    Quando si crea un'istanza, viene automaticamente creato un analizzatore interno
    (TextEntropyAnalyzer) che processa il file di testo e calcola tutte le statistiche
    necessarie per la generazione.

    Il modello può:
    - Analizzare le proprietà entropiche di un testo a livello lessicale
    - Calcolare probabilità assolute e condizionate sulle parole
    - Generare nuovo testo campionando dalle distribuzioni apprese

    Attributes:
        k (int): Lunghezza del contesto (prefisso) in numero di parole;
                 -1 indica il modello uniforme (lessico equiprobabile)
        analyzer (TextEntropyAnalyzer): Istanza embedded dell'analizzatore
        all_words (list): Lista di tutte le parole possibili (lessico)

    Esempio:
        >>> model = LanguageModel("testo.txt", k=2)
        >>> print(f"Entropia: {model.get_entropy():.3f} bit")
        >>> text = model.generate_text(length=50, temperature=0.3)
        >>> # Modello uniforme (baseline):
        >>> model_uni = LanguageModel("testo.txt", k=-1)
        >>> print(f"Entropia uniforme: {model_uni.get_entropy():.3f} bit")
    """

    class TextEntropyAnalyzer:
        """
        Analizzatore di entropia per testi a livello di parola (classe interna di LanguageModel).

        Questa classe legge un file di testo, lo pulisce (rimuovendo i caratteri non alfabetici
        e portando tutto in minuscolo, ma mantenendo i caratteri accentati), e calcola varie
        statistiche probabilistiche ed entropiche basate su parole e prefissi di k parole.

        Attributes:
            k (int): Lunghezza dei prefissi (in parole) per l'analisi condizionata;
                     -1 indica il modello uniforme (nessun prefisso calcolato)
            text (str): Testo pulito come stringa
            words (list): Sequenza di parole pulite
            word_counts (defaultdict): Conteggio delle occorrenze di ogni parola
            prefix_counts (defaultdict): Conteggio delle occorrenze di ogni prefisso di k parole
                                         (vuoto se k=-1)
            conditional_counts (defaultdict): Conteggio delle occorrenze di parole dato un prefisso
                                              (vuoto se k=-1)
        """

        def __init__(self, filepath, k=1):
            """
            Inizializza l'analizzatore di testo.

            Legge il file specificato, lo pulisce rimuovendo i caratteri non alfabetici
            (mantenendo gli accenti) e convertendo in minuscolo, poi calcola tutte le
            statistiche necessarie per l'analisi entropica a livello di parola.

            Con k=-1 viene istanziato il modello uniforme: si calcola solo il lessico
            (word_counts), mentre prefissi e conteggi condizionati non vengono popolati.

            Args:
                filepath (str o Path): Percorso del file di testo da analizzare
                k (int, optional): Lunghezza dei prefissi (in parole).
                                   Usare -1 per il modello uniforme. Default: 1

            Raises:
                FileNotFoundError: Se il file specificato non esiste
            """
            # Memorizza la lunghezza dei prefissi (in parole)
            self.k = k

            # Legge e pulisce il testo dal file; ottiene anche la lista di parole
            self.text, self.words = self._read_and_clean(filepath)

            # Inizializza le strutture dati per le statistiche
            # word_counts: conta quante volte appare ogni parola nel testo
            self.word_counts = defaultdict(int)

            # prefix_counts: conta quante volte appare ogni prefisso di k parole (come tupla)
            self.prefix_counts = defaultdict(int)

            # conditional_counts: per ogni prefisso, conta quante volte appare ogni parola successiva
            # Struttura: conditional_counts[prefisso_tupla][parola] = conteggio
            self.conditional_counts = defaultdict(lambda: defaultdict(int))

            # Calcola tutte le statistiche dal testo
            self._calculate_statistics()

        def _read_and_clean(self, filepath):
            """
            Legge il file e pulisce il testo a livello di parola.

            Operazioni eseguite:
            1. Legge il file con encoding UTF-8
            2. Converte tutto in minuscolo
            3. Sostituisce i caratteri non alfabetici (eccetto spazi) con spazi
               I caratteri accentati (à, è, ì, ò, ù, ecc.) vengono mantenuti
            4. Normalizza sequenze multiple di spazi in spazi singoli
            5. Tokenizza il testo in una lista di parole

            Args:
                filepath (str o Path): Percorso del file da leggere

            Returns:
                tuple: (testo_pulito: str, parole: list)

            Note:
                In caso di errore nella lettura, stampa un messaggio e ritorna
                stringa vuota e lista vuota.
            """
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            except FileNotFoundError:
                print(f"Errore: file '{filepath}' non trovato")
                return "", []

            # Converte tutto in minuscolo
            text = text.lower()

            # Sostituisce ogni carattere che non è una lettera Unicode con uno spazio.
            # \w in Python con flag re.UNICODE include le lettere accentate,
            # ma include anche cifre e underscore, quindi usiamo la categoria Unicode
            # esplicitamente: manteniamo solo caratteri che hanno categoria "L" (Letter).
            cleaned_chars = []
            for c in text:
                import unicodedata
                cat = unicodedata.category(c)
                # Le categorie "L*" sono tutte le lettere (comprese quelle accentate)
                if cat.startswith('L'):
                    cleaned_chars.append(c)
                else:
                    cleaned_chars.append(' ')
            cleaned = ''.join(cleaned_chars)

            # Normalizza sequenze multiple di spazi in uno spazio singolo
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            # Tokenizza in parole (split su spazio)
            words = cleaned.split()

            return cleaned, words

        def _calculate_statistics(self):
            """
            Calcola le statistiche del testo a livello di parola.

            Popola le strutture dati con:
            1. Conteggio di ogni parola nel testo
            2. Conteggio di ogni prefisso di k parole (come tupla)    [saltato se k=-1]
            3. Conteggio di parole che seguono ogni prefisso           [saltato se k=-1]

            Complessità temporale: O(n) dove n è il numero di parole nel testo
            """
            # Prima passata: conta le occorrenze di ogni singola parola
            for word in self.words:
                self.word_counts[word] += 1

            # Con k=-1 (modello uniforme) non servono prefissi né conteggi condizionati
            if self.k == -1:
                return

            # Seconda passata: conta prefissi e parole condizionate
            # Itera fino a len(words) - k per avere sempre una parola successiva
            for i in range(len(self.words) - self.k):
                # Estrae il prefisso come tupla di k parole consecutive
                prefix = tuple(self.words[i:i + self.k])

                # Prende la parola che segue il prefisso
                next_word = self.words[i + self.k]

                # Incrementa il contatore del prefisso
                self.prefix_counts[prefix] += 1

                # Incrementa il contatore per "next_word dato prefix"
                self.conditional_counts[prefix][next_word] += 1

        def get_word_probabilities(self):
            """
            Restituisce le probabilità di occorrenza delle parole.

            Con k=-1 (modello uniforme) ritorna distribuzione equiprobabile 1/|V|.
            Altrimenti calcola P(W = w) come frequenza relativa osservata:
            count(w) / numero_totale_di_parole

            Returns:
                dict: Dizionario con parole come chiavi e probabilità come valori
                      {parola: probabilità}
            """
            vocab = list(self.word_counts.keys())
            if self.k == -1:
                # Distribuzione uniforme: ogni parola del lessico ha prob 1/|V|
                uniform_prob = 1.0 / len(vocab)
                return {word: uniform_prob for word in vocab}
            total = len(self.words)
            return {word: count / total for word, count in self.word_counts.items()}

        def get_prefix_probabilities(self):
            """
            Restituisce le probabilità di occorrenza dei prefissi.

            Calcola P(prefisso) per ogni prefisso di k parole.

            Returns:
                dict: Dizionario con prefissi (tuple) come chiavi e probabilità come valori
                      {prefisso_tupla: probabilità}
            """
            total = sum(self.prefix_counts.values())
            return {prefix: count / total for prefix, count in self.prefix_counts.items()}

        def get_conditional_probabilities(self, prefix):
            """
            Restituisce le probabilità condizionate per un prefisso.

            Calcola P(W = w | prefisso) per ogni parola w che è stata osservata
            dopo il prefisso specificato.

            Args:
                prefix (tuple): Il prefisso (tupla di k parole) da condizionare

            Returns:
                dict: Dizionario con parole come chiavi e probabilità condizionate come valori
                      {parola: P(parola|prefisso)}
                      Ritorna dizionario vuoto se il prefisso non è mai stato osservato
            """
            total = self.prefix_counts.get(prefix, 0)
            if total == 0:
                return {}
            return {word: count / total
                    for word, count in self.conditional_counts[prefix].items()}

        def calculate_entropy(self, probabilities):
            """
            Calcola l'entropia da un dizionario di probabilità.

            L'entropia di Shannon è definita come:
            H(X) = -Σ P(x) * log₂(P(x))

            Args:
                probabilities (dict): Dizionario di probabilità {evento: probabilità}

            Returns:
                float: Entropia in bit (base 2 del logaritmo)
            """
            entropy = 0.0
            for prob in probabilities.values():
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            return entropy

        def get_entropy(self):
            """
            Calcola l'entropia del testo a livello di parola.

            Con k=-1 (modello uniforme) l'entropia è massima: H(W) = log₂(|V|),
            dove |V| è la dimensione del lessico.
            Altrimenti calcola H(W) empiricamente dalle frequenze osservate.

            Returns:
                float: Entropia in bit
            """
            if self.k == -1:
                # Distribuzione uniforme: H = log2(|V|)
                vocab_size = len(self.word_counts)
                return math.log2(vocab_size) if vocab_size > 0 else 0.0
            probs = self.get_word_probabilities()
            return self.calculate_entropy(probs)

        def get_conditional_entropy(self, prefix):
            """
            Calcola l'entropia condizionata per un prefisso.

            Calcola H(W | prefisso), che misura l'incertezza residua nel predire
            la prossima parola quando si conosce il prefisso di k parole.

            Args:
                prefix (tuple): Il prefisso (tupla di k parole) da condizionare

            Returns:
                float: Entropia condizionata in bit
            """
            probs = self.get_conditional_probabilities(prefix)
            return self.calculate_entropy(probs)

        def get_average_conditional_entropy(self):
            """
            Calcola l'entropia condizionata media pesata sui prefissi.

            Con k=-1 (modello uniforme) non esiste contesto: l'entropia condizionata
            coincide con l'entropia assoluta H(W) = log₂(|V|), ovvero il contesto
            non apporta alcuna riduzione di incertezza.
            Altrimenti calcola H(W | W^k) = Σ P(prefisso) * H(W | prefisso).

            Returns:
                float: Entropia condizionata media in bit

            Note:
                Un valore più basso indica che il contesto è più informativo.
                H(W | W^k) ≤ H(W) per ogni k ≥ 0
            """
            if self.k == -1:
                # Senza contesto, l'incertezza residua è quella del modello uniforme
                return self.get_entropy()
            prefix_probs = self.get_prefix_probabilities()
            avg_entropy = 0.0
            for prefix, prob_prefix in prefix_probs.items():
                cond_entropy = self.get_conditional_entropy(prefix)
                avg_entropy += prob_prefix * cond_entropy
            return avg_entropy

        def print_report(self):
            """
            Stampa un rapporto dell'analisi entropica a livello di parola.

            Il rapporto include solo valori derivati (non distribuzioni per singole parole):
            - Statistiche generali (numero di parole, dimensione lessico, prefissi unici)
            - Entropia assoluta H(W)
            - Entropia condizionata media H(W|W^k)  [non applicabile con k=-1]
            - Riduzione di entropia dovuta al contesto
            """
            print("=" * 70)
            if self.k == -1:
                print("ANALISI ENTROPICA DEL TESTO (modello uniforme, k=-1)")
            else:
                print("ANALISI ENTROPICA DEL TESTO (modello a parole)")
            print("=" * 70)

            # Sezione informazioni generali
            print(f"\nNumero di parole nel testo:   {len(self.words)}")
            print(f"Dimensione del lessico:        {len(self.word_counts)} parole uniche")
            if self.k == -1:
                print(f"Modello:                       uniforme (k=-1, nessun contesto)")
            else:
                print(f"Lunghezza prefissi (k):        {self.k} {'parola' if self.k == 1 else 'parole'}")
                print(f"Numero di prefissi unici:      {len(self.prefix_counts)}")

            # Sezione misure di entropia
            print("\n" + "-" * 70)
            print("MISURE DI ENTROPIA")
            print("-" * 70)

            entropy = self.get_entropy()

            if self.k == -1:
                print(f"\nEntropia uniforme   H(W):     {entropy:.6f} bit")
                print(f"  (= log₂({len(self.word_counts)}) = log₂(|lessico|))")
                print(f"\nEntropia condizionata:        non applicabile (nessun contesto)")
                print(f"Riduzione di entropia:         0.000000 bit  (0.00%)")
            else:
                avg_cond_entropy = self.get_average_conditional_entropy()
                reduction = entropy - avg_cond_entropy
                reduction_pct = (reduction / entropy * 100) if entropy > 0 else 0.0

                print(f"\nEntropia assoluta   H(W):         {entropy:.6f} bit")
                print(f"Entropia condizionata media")
                print(f"  H(W|W^{self.k}):                   {avg_cond_entropy:.6f} bit")
                print(f"\nRiduzione di entropia:            {reduction:.6f} bit  ({reduction_pct:.2f}%)")
                print(f"  (informazione apportata dal contesto di {self.k} {'parola' if self.k == 1 else 'parole'})")

            print("\n" + "=" * 70)

        def print_latex_report(self):
            """
            Stampa un rapporto dell'analisi entropica in formato LaTeX.

            Produce una tabella con le statistiche generali e le misure di entropia.
            Non vengono riportate distribuzioni per singole parole.
            """
            if self.k == -1:
                print("ANALISI ENTROPICA DEL TESTO (modello uniforme, k=-1)")
            else:
                print("ANALISI ENTROPICA DEL TESTO (modello a parole)")
            print("\\begin{tabular}{|l|c|}")
            print("\\hline")
            print(f"Numero di parole & {len(self.words)} \\\\")
            print(f"Dimensione del lessico & {len(self.word_counts)} \\\\")
            if self.k == -1:
                print(f"Modello & uniforme (k=-1) \\\\")
            else:
                print(f"Lunghezza prefissi (k) & {self.k} \\\\")
                print(f"Numero di prefissi unici & {len(self.prefix_counts)} \\\\")
            print("\\hline")
            print("\\end{tabular}")

            entropy = self.get_entropy()

            print("\nMISURE DI ENTROPIA")
            print("\\begin{tabular}{|l|c|}")
            print("\\hline")
            if self.k == -1:
                print(f"$H(W)$ (uniforme) & {entropy:.6f} \\\\")
                print(f"Riduzione di entropia & 0.000000 (0.00\\%) \\\\")
            else:
                avg_cond_entropy = self.get_average_conditional_entropy()
                reduction = entropy - avg_cond_entropy
                reduction_pct = (reduction / entropy * 100) if entropy > 0 else 0.0
                print(f"$H(W)$ & {entropy:.6f} \\\\")
                print(f"$H(W|W^{{{self.k}}})$ & {avg_cond_entropy:.6f} \\\\")
                print(f"Riduzione di entropia & {reduction:.6f} ({reduction_pct:.2f}\\%) \\\\")
            print("\\hline")
            print("\\end{tabular}")

    # Fine della classe interna TextEntropyAnalyzer

    def __init__(self, filepath, k=1):
        """
        Inizializza il modello di linguaggio a livello di parola.

        Crea un'istanza embedded di TextEntropyAnalyzer che analizza il file
        di testo e calcola tutte le statistiche necessarie per la generazione.

        Args:
            filepath (str o Path): Percorso del file di testo da analizzare
            k (int, optional): Lunghezza del contesto in parole (ordine del modello).
                               Usare -1 per il modello uniforme (lessico equiprobabile).
                               Default: 1

        Esempio:
            >>> model = LanguageModel("divina_commedia.txt", k=3)
            >>> print(f"Entropia: {model.get_entropy():.3f}")
            >>> text = model.generate_text(50, temperature=0.2)
            >>> # Modello uniforme:
            >>> model_uni = LanguageModel("divina_commedia.txt", k=-1)
        """
        self.k = k
        self.analyzer = self.TextEntropyAnalyzer(filepath, k)
        # Lista di tutte le parole del lessico (per campionamento uniforme)
        self.all_words = list(self.analyzer.word_counts.keys())

    # ========================================================================
    # METODI WRAPPER - Delegano all'istanza embedded di TextEntropyAnalyzer
    # ========================================================================

    def get_word_probabilities(self):
        """
        Restituisce le probabilità di occorrenza delle parole.

        Returns:
            dict: Dizionario {parola: probabilità}
        """
        return self.analyzer.get_word_probabilities()

    def get_prefix_probabilities(self):
        """
        Restituisce le probabilità di occorrenza dei prefissi.

        Returns:
            dict: Dizionario {prefisso_tupla: probabilità}
        """
        return self.analyzer.get_prefix_probabilities()

    def get_conditional_probabilities(self, prefix):
        """
        Restituisce le probabilità condizionate per un prefisso.

        Args:
            prefix (tuple): Il prefisso (tupla di k parole) da condizionare

        Returns:
            dict: Dizionario {parola: P(parola|prefisso)}
        """
        return self.analyzer.get_conditional_probabilities(prefix)

    def calculate_entropy(self, probabilities):
        """
        Calcola l'entropia da un dizionario di probabilità.

        Args:
            probabilities (dict): Dizionario di probabilità

        Returns:
            float: Entropia in bit
        """
        return self.analyzer.calculate_entropy(probabilities)

    def get_entropy(self):
        """
        Calcola l'entropia del testo H(W).

        Returns:
            float: Entropia in bit
        """
        return self.analyzer.get_entropy()

    def get_conditional_entropy(self, prefix):
        """
        Calcola l'entropia condizionata per un prefisso.

        Args:
            prefix (tuple): Il prefisso (tupla di k parole) da condizionare

        Returns:
            float: Entropia condizionata in bit
        """
        return self.analyzer.get_conditional_entropy(prefix)

    def get_average_conditional_entropy(self):
        """
        Calcola l'entropia condizionata media H(W|W^k).

        Returns:
            float: Entropia condizionata media in bit
        """
        return self.analyzer.get_average_conditional_entropy()

    def print_report(self):
        """
        Stampa un rapporto completo dell'analisi entropica.

        Delega al metodo corrispondente dell'analizzatore embedded.
        """
        self.analyzer.print_report()

    def get_text(self):
        """
        Restituisce il testo pulito analizzato.

        Returns:
            str: Il testo processato e pulito
        """
        return self.analyzer.text

    def get_words(self):
        """
        Restituisce la lista di parole del testo analizzato.

        Returns:
            list: Lista di parole pulite
        """
        return self.analyzer.words

    def get_text_length(self):
        """
        Restituisce il numero di parole nel testo analizzato.

        Returns:
            int: Numero di parole nel testo pulito
        """
        return len(self.analyzer.words)

    # ========================================================================
    # METODI PROPRI DI LANGUAGEMODEL - Generazione di testo
    # ========================================================================

    def _get_prefix(self, word_list):
        """
        Estrae il prefisso rilevante da una lista di parole.

        Se la lista ha almeno k parole, prende le ultime k.
        Se è più corta, aggiunge padding di stringhe vuote "" all'inizio.

        Args:
            word_list (list): Lista di parole dal testo corrente

        Returns:
            tuple: Prefisso di esattamente k parole (come tupla)

        Esempio:
            Se k=2 e word_list=["nel","mezzo","del"], ritorna ("mezzo","del")
            Se k=2 e word_list=["nel"], ritorna ("","nel")
        """
        if len(word_list) >= self.k:
            return tuple(word_list[-self.k:])
        else:
            padding = [""] * (self.k - len(word_list))
            return tuple(padding + word_list)

    def _sample_from_distribution(self, probs):
        """
        Campiona una parola da una distribuzione di probabilità.

        Usa il metodo della ruota della fortuna (roulette wheel selection):
        genera un numero casuale e trova in quale intervallo cade.

        Args:
            probs (dict): Dizionario {parola: probabilità}

        Returns:
            str: Parola campionata
        """
        r = random.random()
        cumulative = 0.0
        for word, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return word
        return list(probs.keys())[-1]

    def next_word(self, word_list, temperature=0.0):
        """
        Campiona la prossima parola data una lista di parole contestuali.

        Con k=-1 (modello uniforme) campiona sempre uniformemente dal lessico,
        ignorando sia il contesto che la temperatura.

        Altrimenti:
        1. Estrae il prefisso rilevante (ultime k parole, con padding se necessario)
        2. Con probabilità 'temperature', campiona uniformemente da tutto il lessico
        3. Altrimenti, campiona dalla distribuzione condizionata P(W | prefisso)
        4. Se il prefisso non è mai stato visto, usa la distribuzione assoluta P(W)

        Args:
            word_list (list): Lista delle parole generate finora (contesto)
            temperature (float): Parametro di randomizzazione, deve essere in [0, 1]
                                - 0.0: completamente deterministico
                                - 1.0: completamente casuale (distribuzione uniforme)
                                - valori intermedi: mix tra distribuzione e casualità
                                (ignorato con k=-1)

        Returns:
            str: La prossima parola generata

        Raises:
            ValueError: Se temperature non è in [0, 1] (solo per k ≥ 1)
        """
        # Modello uniforme: campionamento diretto dal lessico, nessun contesto
        if self.k == -1:
            return random.choice(self.all_words)

        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature deve essere compreso tra 0 e 1")

        prefix = self._get_prefix(word_list)

        # Con probabilità 'temperature', campionamento uniforme sul lessico
        if random.random() < temperature:
            return random.choice(self.all_words)

        # Altrimenti usa la distribuzione condizionata
        if prefix in self.analyzer.conditional_counts and self.analyzer.prefix_counts[prefix] > 0:
            cond_probs = self.get_conditional_probabilities(prefix)
            return self._sample_from_distribution(cond_probs)
        else:
            # Fallback alla distribuzione assoluta
            word_probs = self.get_word_probabilities()
            return self._sample_from_distribution(word_probs)

    def generate_text(self, length=50, temperature=0.0, seed_text=""):
        """
        Genera una sequenza di testo di lunghezza specificata (in parole).

        Genera parola per parola usando next_word(), partendo da un testo
        iniziale (seed) o da una lista vuota.

        Args:
            length (int): Numero di parole da generare
            temperature (float): Parametro di randomizzazione in [0, 1]
                                - 0.0: generazione deterministica
                                - 1.0: generazione completamente casuale
            seed_text (str): Testo iniziale da cui partire (opzionale);
                             viene pulito con le stesse regole del training

        Returns:
            str: Testo generato come stringa di parole separate da spazio

        Esempio:
            >>> model = LanguageModel("divina_commedia.txt", k=3)
            >>> text = model.generate_text(length=30, temperature=0.2, seed_text="nel mezzo")
        """
        # Pulisce e tokenizza il seed con le stesse regole del training
        if seed_text:
            seed_text_lower = seed_text.lower()
            import unicodedata
            cleaned_chars = []
            for c in seed_text_lower:
                cat = unicodedata.category(c)
                if cat.startswith('L'):
                    cleaned_chars.append(c)
                else:
                    cleaned_chars.append(' ')
            cleaned_seed = re.sub(r'\s+', ' ', ''.join(cleaned_chars)).strip()
            generated_words = cleaned_seed.split() if cleaned_seed else []
        else:
            generated_words = []

        # Genera parola per parola
        for _ in range(length):
            next_w = self.next_word(generated_words, temperature=temperature)
            generated_words.append(next_w)

        return ' '.join(generated_words)


def print_menu():
    """
    Stampa il menu principale dei comandi disponibili.
    """
    print("\n" + "=" * 70)
    print("MENU PRINCIPALE - LANGUAGE MODEL (a parole)")
    print("=" * 70)
    print("Comandi disponibili:")
    print("  1. genera    - Genera testo con il modello")
    print("  2. report    - Mostra il rapporto di analisi entropica")
    print("  3. info      - Mostra informazioni base sul modello")
    print("  4. esci      - Termina il programma")
    print("=" * 70)


def get_user_input(prompt, input_type=str, default=None, validate=None):
    """
    Richiede input dall'utente con validazione e valore di default.

    Args:
        prompt (str): Messaggio da mostrare all'utente
        input_type (type): Tipo di dato atteso (str, int, float)
        default: Valore di default se l'utente preme solo Invio
        validate (callable): Funzione di validazione opzionale che ritorna True se valido

    Returns:
        Il valore inserito dall'utente, convertito al tipo corretto
    """
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} [default: {default}]: ").strip()
                if user_input == "":
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()

            value = input_type(user_input)

            if validate is not None and not validate(value):
                print("Valore non valido. Riprova.")
                continue

            return value

        except ValueError:
            print(f"Errore: inserire un valore di tipo {input_type.__name__}. Riprova.")


def command_generate(model):
    """
    Esegue il comando di generazione testo.

    Richiede all'utente i parametri necessari (numero di parole, temperatura, seed)
    e genera una sequenza di testo usando il modello di linguaggio.
    Con k=-1 (modello uniforme) il parametro temperatura non viene richiesto,
    poiché la generazione è intrinsecamente uniforme.

    Args:
        model (LanguageModel): Il modello di linguaggio da usare per la generazione
    """
    print("\n" + "-" * 70)
    print("GENERAZIONE DI TESTO")
    if model.k == -1:
        print("  (modello uniforme: campionamento equiprobabile sul lessico)")
    print("-" * 70)

    length = get_user_input(
        "Lunghezza del testo (numero di parole)",
        input_type=int,
        default=50,
        validate=lambda x: x > 0
    )

    # La temperatura non ha significato per il modello uniforme
    if model.k == -1:
        temperature = 0.0
    else:
        temperature = get_user_input(
            "Temperatura (0.0 = deterministico, 1.0 = casuale)",
            input_type=float,
            default=0.0,
            validate=lambda x: 0.0 <= x <= 1.0
        )

    seed_text = get_user_input(
        "Testo iniziale (seed, premi Invio per nessun seed)",
        input_type=str,
        default=""
    )

    print("\n" + "-" * 70)
    print("Parametri di generazione:")
    print(f"  Lunghezza: {length} parole")
    if model.k != -1:
        print(f"  Temperatura: {temperature}")
    if seed_text:
        print(f"  Seed: \"{seed_text}\"")
    else:
        print(f"  Seed: (nessuno)")
    print("-" * 70)
    print("Generazione in corso...\n")

    try:
        generated = model.generate_text(
            length=length,
            temperature=temperature,
            seed_text=seed_text
        )

        print("TESTO GENERATO:")
        print("-" * 70)
        print(generated)
        print("-" * 70)

    except Exception as e:
        print(f"Errore durante la generazione: {e}")


def command_report(model):
    """
    Esegue il comando di stampa del rapporto di analisi.

    Mostra il rapporto con le statistiche entropiche a livello di parola.
    Non vengono mostrate distribuzioni per singole parole del lessico.

    Args:
        model (LanguageModel): Il modello di linguaggio con i dati da analizzare
    """
    print()
    model.print_report()


def command_info(model):
    """
    Esegue il comando di visualizzazione informazioni base.

    Mostra un sommario rapido delle informazioni principali sul modello.

    Args:
        model (LanguageModel): Il modello di linguaggio da ispezionare
    """
    print("\n" + "=" * 70)
    print("INFORMAZIONI SUL MODELLO")
    print("=" * 70)

    if model.k == -1:
        print(f"\nOrdine del modello (k): -1  (modello uniforme)")
    else:
        print(f"\nOrdine del modello (k): {model.k} {'parola' if model.k == 1 else 'parole'}")
    print(f"Parole nel testo analizzato: {model.get_text_length()}")
    print(f"Dimensione del lessico: {len(model.analyzer.word_counts)} parole uniche")
    if model.k != -1:
        print(f"Prefissi unici: {len(model.analyzer.prefix_counts)}")

    print("\nMisure di entropia:")
    entropy = model.get_entropy()

    if model.k == -1:
        print(f"  H(W) uniforme = {entropy:.4f} bit  (= log₂({len(model.analyzer.word_counts)}))")
        print(f"  Nessun contesto: riduzione = 0.0000 bit (0.0%)")
    else:
        avg_cond_entropy = model.get_average_conditional_entropy()
        reduction = entropy - avg_cond_entropy
        reduction_pct = (reduction / entropy * 100) if entropy > 0 else 0

        print(f"  H(W)       = {entropy:.4f} bit")
        print(f"  H(W|W^{model.k})   = {avg_cond_entropy:.4f} bit")
        print(f"  Riduzione  = {reduction:.4f} bit ({reduction_pct:.1f}%)")

    print("=" * 70)


def main():
    """
    Funzione principale per l'esecuzione interattiva del programma.

    Il programma funziona in due fasi:
    1. Carica il modello di linguaggio da file
    2. Entra in modalità interattiva con menu di comandi

    Uso:
        python language_model_words.py <filepath> [k]

    Argomenti:
        filepath: percorso del file di testo da analizzare
        k: lunghezza del contesto in parole (ordine del modello, default 1)
           Usare -1 per il modello uniforme (lessico equiprobabile)

    Esempi:
        python language_model_words.py testo.txt
        python language_model_words.py divina_commedia.txt 3
        python language_model_words.py divina_commedia.txt -1
    """
    import sys

    if len(sys.argv) < 2:
        print("=" * 70)
        print("LANGUAGE MODEL (a parole) - Analisi e Generazione di Testo")
        print("=" * 70)
        print("\nUtilizzo: python language_model_words.py <filepath> [k]")
        print("\nArgomenti:")
        print("  filepath: percorso del file di testo da analizzare")
        print("  k: lunghezza del contesto in parole (ordine del modello, default 0)")
        print("     Usare k=-1 per il modello uniforme (lessico equiprobabile)")
        print("\nEsempi:")
        print("  python language_model_words.py testo.txt")
        print("  python language_model_words.py divina_commedia.txt 3")
        print("  python language_model_words.py divina_commedia.txt -1")
        print("\nDopo il caricamento, il programma entrerà in modalità interattiva")
        print("dove potrai generare testo e visualizzare report di analisi.")
        print("=" * 70)
        sys.exit(1)

    filepath = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Valida k: deve essere -1 oppure un intero positivo
    if k < -1:
        print(f"✗ Errore: k deve essere un intero non negativo oppure -1 (modello uniforme).")
        sys.exit(1)

    print("=" * 70)
    if k == -1:
        print(f"LANGUAGE MODEL: File {filepath}, distribuzione uniforme delle parole")
    else:
        print(f"LANGUAGE MODEL: File {filepath}, modello di ordine {k}")
    print("=" * 70)
    print("Analisi del testo in corso...")

    try:
        model = LanguageModel(filepath, k)
        print(f"✓ Lunghezza testo: {model.get_text_length()} parole")
        print(f"✓ {len(model.analyzer.word_counts)} parole distinte")
        if k == -1:
            print(f"✓ Probabilità p uniforme pari a 1/{len(model.analyzer.word_counts)}={1/len(model.analyzer.word_counts):.6f} per ogni parola")
        elif k == 0:
            print(f"✓ Probabilità p(x) non basata su prefissi (k=0), ma su frequenze assolute")
        elif k==1:
            print(f"✓ Probabilità p(x|x_1) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.word_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.word_counts)**k*100:.2f}% del totale")
        elif k==2:
            print(f"✓ Probabilità p(x|x_1x_2) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.word_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.word_counts)**k*100:.2f}% del totale")
        elif k==3:
            print(f"✓ Probabilità p(x|x_1x_2x_3) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.word_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.word_counts)**k*100:.2f}% del totale")
        else:
            print(f"✓ Probabilità p(x|x_1,...,x_{k}) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.word_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.word_counts)**k*100:.2f}% del totale")

    except FileNotFoundError:
        print(f"\n✗ Errore: file '{filepath}' non trovato")
        print("Verifica che il percorso sia corretto e riprova.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Errore durante il caricamento: {e}")
        sys.exit(1)

    commands = {
        '1': ('genera', command_generate),
        'genera': ('genera', command_generate),
        '2': ('report', command_report),
        'report': ('report', command_report),
        '3': ('info', command_info),
        'info': ('info', command_info),
        '4': ('esci', None),
        'esci': ('esci', None),
        'exit': ('esci', None),
        'quit': ('esci', None),
        'q': ('esci', None),
    }

    while True:
        print_menu()
        user_input = input("\nInserisci un comando: ").strip().lower()

        if user_input in commands:
            command_name, command_func = commands[user_input]

            if command_name == 'esci':
                print("\n" + "=" * 70)
                print("Chiusura del programma...")
                print("Arrivederci!")
                print("=" * 70)
                break

            try:
                command_func(model)
            except KeyboardInterrupt:
                print("\n\nComando interrotto dall'utente.")
            except Exception as e:
                print(f"\n✗ Errore durante l'esecuzione del comando: {e}")

        else:
            print(f"\n✗ Comando '{user_input}' non riconosciuto.")
            print("Usa uno dei comandi del menu (1-4, oppure il nome completo).")


if __name__ == "__main__":
    main()
