"""
Modulo per l'analisi dell'entropia di testi e la generazione di testo basata su modelli probabilistici.

Questo modulo fornisce strumenti per:
- Analizzare l'entropia di un testo basandosi su statistiche di caratteri e prefissi
- Calcolare probabilità condizionate di caratteri dato un contesto
- Generare testo usando un modello probabilistico basato sui dati analizzati

Modalità speciale k=-1 (modello uniforme):
    Tutti i caratteri dell'alfabeto osservato hanno probabilità equiprobabile 1/|Σ|.
    L'entropia è massima: H(X) = log₂(|Σ|). Il contesto non viene usato;
    la generazione campiona uniformemente sull'alfabeto.
    Rappresenta il caso di riferimento (baseline) con massima incertezza.

Classe principale:
    LanguageModel: Modello di linguaggio che include analisi entropica e generazione di testo
    
Classe interna:
    LanguageModel.TextEntropyAnalyzer: Analizzatore di entropia (embedded in LanguageModel)
"""

import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path


class LanguageModel:
    """
    Modello di linguaggio basato su distribuzioni di probabilità condizionate.
    
    Questa classe integra l'analisi entropica di un testo e la generazione di nuovo testo.
    Quando si crea un'istanza, viene automaticamente creato un analizzatore interno
    (TextEntropyAnalyzer) che processa il file di testo e calcola tutte le statistiche
    necessarie per la generazione.
    
    Il modello può:
    - Analizzare le proprietà entropiche di un testo
    - Calcolare probabilità assolute e condizionate
    - Generare nuovo testo campionando dalle distribuzioni apprese
    
    Attributes:
        k (int): Lunghezza del contesto (prefisso) considerato;
                 -1 indica il modello uniforme (alfabeto equiprobabile)
        analyzer (TextEntropyAnalyzer): Istanza embedded dell'analizzatore
        all_chars (list): Lista di tutti i caratteri possibili
    
    Esempio:
        >>> model = LanguageModel("testo.txt", k=2)
        >>> print(f"Entropia: {model.get_entropy():.3f} bit")
        >>> text = model.generate_text(length=100, temperature=0.3)
        >>> # Modello uniforme (baseline):
        >>> model_uni = LanguageModel("testo.txt", k=-1)
        >>> print(f"Entropia uniforme: {model_uni.get_entropy():.3f} bit")
    """
    
    class TextEntropyAnalyzer:
        """
        Analizzatore di entropia per testi (classe interna di LanguageModel).
        
        Questa classe legge un file di testo, lo pulisce (rimuovendo accenti e caratteri speciali),
        e calcola varie statistiche probabilistiche ed entropiche basate su caratteri e prefissi
        di lunghezza k.
        
        Attributes:
            k (int): Lunghezza dei prefissi da considerare per l'analisi condizionata;
                     -1 indica il modello uniforme (nessun prefisso calcolato)
            text (str): Testo pulito e processato
            char_counts (defaultdict): Conteggio delle occorrenze di ogni carattere
            prefix_counts (defaultdict): Conteggio delle occorrenze di ogni prefisso di lunghezza k
                                         (vuoto se k=-1)
            conditional_counts (defaultdict): Conteggio delle occorrenze di caratteri dato un prefisso
                                              (vuoto se k=-1)
        """
        
        def __init__(self, filepath, k=1):
            """
            Inizializza l'analizzatore di testo.
            
            Legge il file specificato, lo pulisce rimuovendo accenti e caratteri non alfabetici,
            e calcola tutte le statistiche necessarie per l'analisi entropica.
            
            Con k=-1 viene istanziato il modello uniforme: si calcola solo l'alfabeto
            (char_counts), mentre prefissi e conteggi condizionati non vengono popolati.
            
            Args:
                filepath (str o Path): Percorso del file di testo da analizzare
                k (int, optional): Lunghezza dei prefissi per l'analisi condizionata.
                                   Usare -1 per il modello uniforme. Default: 1
            
            Raises:
                FileNotFoundError: Se il file specificato non esiste
            """
            # Memorizza la lunghezza dei prefissi
            self.k = k
            
            # Legge e pulisce il testo dal file
            self.text = self._read_and_clean(filepath)
            
            # Inizializza le strutture dati per le statistiche
            # char_counts: conta quante volte appare ogni carattere
            self.char_counts = defaultdict(int)
            
            # prefix_counts: conta quante volte appare ogni prefisso di lunghezza k
            self.prefix_counts = defaultdict(int)
            
            # conditional_counts: per ogni prefisso, conta quante volte appare ogni carattere successivo
            # Struttura: conditional_counts[prefisso][carattere] = conteggio
            self.conditional_counts = defaultdict(lambda: defaultdict(int))
            
            # Calcola tutte le statistiche dal testo
            self._calculate_statistics()
        
        def _remove_accents(self, text):
            """
            Rimuove gli accenti dai caratteri.
            
            Utilizza la normalizzazione Unicode NFD (Normalization Form Decomposition) per
            separare i caratteri base dai loro segni diacritici, poi filtra via i segni diacritici.
            
            Args:
                text (str): Testo da cui rimuovere gli accenti
            
            Returns:
                str: Testo senza accenti
            
            Esempio:
                "àèìòù" -> "aeiou"
                "café" -> "cafe"
            """
            # Decompone i caratteri Unicode: separa lettere base da accenti
            # NFD = Normalization Form Canonical Decomposition
            nfd = unicodedata.normalize('NFD', text)
            
            # Filtra i caratteri mantenendo solo quelli che non sono "Mark, Nonspacing" (accenti)
            # Category 'Mn' identifica i segni diacritici combinatori
            return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        
        def _read_and_clean(self, filepath):
            """
            Legge il file e pulisce il testo.
            
            Operazioni eseguite:
            1. Legge il file con encoding UTF-8
            2. Rimuove tutti gli accenti
            3. Converte tutto in minuscolo
            4. Mantiene solo lettere alfabetiche e spazi
            5. Normalizza sequenze multiple di spazi in spazi singoli
            
            Args:
                filepath (str o Path): Percorso del file da leggere
            
            Returns:
                str: Testo pulito e normalizzato
            
            Note:
                In caso di errore nella lettura, stampa un messaggio e ritorna stringa vuota
            """
            try:
                # Apre il file in modalità lettura con encoding UTF-8
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()  # Legge tutto il contenuto del file
            except FileNotFoundError:
                # Gestisce il caso in cui il file non esista
                print(f"Errore: file '{filepath}' non trovato")
                return ""  # Ritorna stringa vuota per permettere la continuazione
            
            # Rimuove gli accenti dal testo usando la funzione helper
            text = self._remove_accents(text)
            
            # Converte tutti i caratteri in minuscolo per normalizzazione
            text = text.lower()
            
            # Filtra il testo: mantiene solo caratteri alfabetici minuscoli e spazi
            # Tutti gli altri caratteri (punteggiatura, numeri, ecc.) vengono rimossi
            cleaned = ''.join(c if (c.isalpha()) or c == ' ' else '' 
                             for c in text)
            
            # Riduce sequenze multiple di whitespace (spazi, tab, newline) a spazi singoli
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Regex: uno o più whitespace -> un solo spazio
            
            # Rimuove spazi iniziali e finali
            cleaned = cleaned.strip()
            
            return cleaned
        
        def _calculate_statistics(self):
            """
            Calcola le statistiche del testo.
            
            Popola le strutture dati con:
            1. Conteggio di ogni carattere nel testo
            2. Conteggio di ogni prefisso di lunghezza k      [saltato se k=-1]
            3. Conteggio di caratteri che seguono ogni prefisso [saltato se k=-1]
            
            Complessità temporale: O(n) dove n è la lunghezza del testo
            """
            # Prima passata: conta le occorrenze di ogni singolo carattere
            for char in self.text:
                self.char_counts[char] += 1  # Incrementa il contatore per questo carattere
            
            # Con k=-1 (modello uniforme) non servono prefissi né conteggi condizionati
            if self.k == -1:
                return
            
            # Seconda passata: conta prefissi e caratteri condizionati
            # Itera fino a len(text) - k per avere sempre un carattere successivo
            for i in range(len(self.text) - self.k):
                # Estrae il prefisso di lunghezza k che inizia alla posizione i
                prefix = self.text[i:i+self.k]
                
                # Prende il carattere che segue il prefisso
                next_char = self.text[i+self.k]
                
                # Incrementa il contatore del prefisso
                self.prefix_counts[prefix] += 1
                
                # Incrementa il contatore per "next_char dato prefix"
                # Questa è la statistica fondamentale per P(X | prefisso)
                self.conditional_counts[prefix][next_char] += 1
        
        def get_char_probabilities(self):
            """
            Restituisce le probabilità di occorrenza dei caratteri.
            
            Con k=-1 (modello uniforme) ritorna distribuzione equiprobabile 1/|Σ|.
            Altrimenti calcola P(X = c) come frequenza relativa osservata:
            count(c) / lunghezza_totale
            
            Returns:
                dict: Dizionario con caratteri come chiavi e probabilità come valori
                      {carattere: probabilità}
            
            Esempio:
                Se il testo è "aabac", ritorna {'a': 0.6, 'b': 0.2, 'c': 0.2}
            """
            alphabet = list(self.char_counts.keys())
            if self.k == -1:
                # Distribuzione uniforme: ogni carattere dell'alfabeto ha prob 1/|Σ|
                uniform_prob = 1.0 / len(alphabet)
                return {char: uniform_prob for char in alphabet}
            # Lunghezza totale del testo (numero di osservazioni)
            total = len(self.text)
            
            # Calcola la probabilità dividendo ogni conteggio per il totale
            return {char: count / total for char, count in self.char_counts.items()}
        
        def get_prefix_probabilities(self):
            """
            Restituisce le probabilità di occorrenza dei prefissi.
            
            Calcola P(prefisso) per ogni prefisso di lunghezza k.
            
            Returns:
                dict: Dizionario con prefissi come chiavi e probabilità come valori
                      {prefisso: probabilità}
            """
            # Numero totale di prefissi osservati
            total = sum(self.prefix_counts.values())
            
            # Calcola la probabilità di ogni prefisso
            return {prefix: count / total for prefix, count in self.prefix_counts.items()}
        
        def get_conditional_probabilities(self, prefix):
            """
            Restituisce le probabilità condizionate per un prefisso.
            
            Calcola P(X = c | prefisso) per ogni carattere c che è stato osservato
            dopo il prefisso specificato.
            
            Args:
                prefix (str): Il prefisso da condizionare (deve avere lunghezza k)
            
            Returns:
                dict: Dizionario con caratteri come chiavi e probabilità condizionate come valori
                      {carattere: P(carattere|prefisso)}
                      Ritorna dizionario vuoto se il prefisso non è mai stato osservato
            
            Esempio:
                Se dopo "ab" è seguito 3 volte "c" e 1 volta "d", ritorna:
                {'c': 0.75, 'd': 0.25}
            """
            # Ottiene il numero totale di volte che questo prefisso è apparso
            total = self.prefix_counts.get(prefix, 0)
            
            # Se il prefisso non è mai stato osservato, ritorna dizionario vuoto
            if total == 0:
                return {}
            
            # Calcola P(char | prefix) = count(prefix -> char) / count(prefix)
            return {char: count / total 
                    for char, count in self.conditional_counts[prefix].items()}
        
        def calculate_entropy(self, probabilities):
            """
            Calcola l'entropia da un dizionario di probabilità.
            
            L'entropia di Shannon è definita come:
            H(X) = -Σ P(x) * log₂(P(x))
            
            Misura l'incertezza media o il contenuto informativo medio.
            
            Args:
                probabilities (dict): Dizionario di probabilità {evento: probabilità}
            
            Returns:
                float: Entropia in bit (base 2 del logaritmo)
            
            Note:
                - Entropia massima quando tutti gli eventi sono equiprobabili
                - Entropia minima (0) quando un evento ha probabilità 1
            """
            entropy = 0.0
            
            # Somma i contributi di ogni evento
            for prob in probabilities.values():
                if prob > 0:  # Salta probabilità zero (log(0) non definito)
                    # Aggiunge -p * log₂(p) all'entropia
                    entropy -= prob * math.log2(prob)
            
            return entropy
        
        def get_entropy(self):
            """
            Calcola l'entropia del testo.
            
            Con k=-1 (modello uniforme) l'entropia è massima: H(X) = log₂(|Σ|),
            dove |Σ| è la dimensione dell'alfabeto osservato.
            Altrimenti calcola H(X) empiricamente dalle frequenze osservate.
            
            Returns:
                float: Entropia in bit
            """
            if self.k == -1:
                # Distribuzione uniforme: H = log2(|Σ|)
                alphabet_size = len(self.char_counts)
                return math.log2(alphabet_size) if alphabet_size > 0 else 0.0
            # Ottiene le probabilità dei caratteri
            probs = self.get_char_probabilities()
            
            # Calcola l'entropia usando la formula di Shannon
            return self.calculate_entropy(probs)
        
        def get_conditional_entropy(self, prefix):
            """
            Calcola l'entropia condizionata per un prefisso.
            
            Calcola H(X | prefisso), che misura l'incertezza residua nel predire
            il prossimo carattere quando si conosce il prefisso.
            
            Args:
                prefix (str): Il prefisso da condizionare
            
            Returns:
                float: Entropia condizionata in bit
            """
            # Ottiene le probabilità condizionate per questo prefisso
            probs = self.get_conditional_probabilities(prefix)
            
            # Calcola l'entropia di questa distribuzione condizionata
            return self.calculate_entropy(probs)
        
        def get_average_conditional_entropy(self):
            """
            Calcola l'entropia condizionata media pesata sui prefissi.
            
            Con k=-1 (modello uniforme) non esiste contesto: l'entropia condizionata
            coincide con l'entropia assoluta H(X) = log₂(|Σ|), ovvero il contesto
            non apporta alcuna riduzione di incertezza.
            Altrimenti calcola H(X | X^k) = Σ P(prefisso) * H(X | prefisso).
            
            Returns:
                float: Entropia condizionata media in bit
            
            Note:
                Un valore più basso indica che il contesto è più informativo.
                H(X | X^k) ≤ H(X) per ogni k ≥ 0
            """
            if self.k == -1:
                # Senza contesto, l'incertezza residua è quella del modello uniforme
                return self.get_entropy()
            # Ottiene le probabilità di ogni prefisso
            prefix_probs = self.get_prefix_probabilities()
            
            avg_entropy = 0.0
            
            # Somma pesata delle entropie condizionate
            for prefix, prob_prefix in prefix_probs.items():
                # Calcola H(X | questo prefisso)
                cond_entropy = self.get_conditional_entropy(prefix)
                
                # Aggiunge il contributo pesato: P(prefisso) * H(X | prefisso)
                avg_entropy += prob_prefix * cond_entropy
            
            return avg_entropy
        
        def print_report(self):
            """
            Stampa un rapporto completo dell'analisi.
            
            Il rapporto include:
            - Statistiche generali (lunghezza, caratteri unici, prefissi unici)
            - Probabilità di tutti i caratteri
            - Entropia assoluta H(X)
            - Entropia condizionata media H(X|X^k)  [non applicabile con k=-1]
            - Dettagli sui 20 prefissi più frequenti con relative entropie condizionate
              [non applicabile con k=-1]
            """
            print("=" * 70)
            if self.k == -1:
                print("ANALISI STATISTICA DEL TESTO (modello uniforme, k=-1)")
            else:
                print("ANALISI STATISTICA DEL TESTO")
            print("=" * 70)
            
            # Sezione informazioni generali
            print(f"\nLunghezza testo: {len(self.text)} caratteri")
            if self.k == -1:
                print(f"Modello:         uniforme (k=-1, nessun contesto)")
            else:
                print(f"Lunghezza prefissi (k): {self.k}")
            print(f"Numero caratteri: {len(self.char_counts)}")
            print(f"Numero prefissi: {len(self.char_counts)**self.k}")
            if self.k != -1:
                print(f"Numero prefissi osservati: {len(self.prefix_counts)}")
                print(f"Frazione dei prefissi osservati: {len(self.prefix_counts)/len(self.char_counts)**self.k:.4f}")
            
            # Sezione probabilità dei caratteri
            print("\n" + "-" * 70)
            print("PROBABILITÀ E INFORMAZIONE DEI CARATTERI")
            print("-" * 70)
            char_probs = self.get_char_probabilities()
            
            # Ordina alfabeticamente per migliore leggibilità
            for char in sorted(char_probs.keys()):
                # Rappresenta lo spazio con 'repr' per visibilità
                display_char = repr(char) if char == ' ' else char
                information = -math.log2(char_probs[char]) if char_probs[char] > 0 else 0
                print(f" {display_char}: p={char_probs[char]:.6f}, I={information:.6f} bit")
            
            # Sezione misure di entropia
            print("\n" + "-" * 70)
            print("MISURE DI ENTROPIA")
            print("-" * 70)
            
            entropy = self.get_entropy()
            
            if self.k == -1:
                print(f"Entropia uniforme H(X): {entropy:.6f} bit")
                print(f"  (= log₂({len(self.char_counts)}) = log₂(|Σ|))")
                print(f"Entropia condizionata:  non applicabile (nessun contesto)")
                print(f"Riduzione di entropia:  0.000000 bit  (0.00%)")
            else:
                # Entropia assoluta (senza condizionamento)
                print(f"Entropia H(X): {entropy:.6f} bit")
                
                # Entropia condizionata media
                avg_cond_entropy = self.get_average_conditional_entropy()
                print(f"Entropia condizionata media H(X|X^{self.k}): {avg_cond_entropy:.6f} bit")
                
                # Sezione prefissi e probabilità condizionate
                print("\n" + "-" * 70)
                print(f"PROBABILITÀ PREFISSI E ENTROPIE CONDIZIONATE")
                print("-" * 70)
                
                prefix_probs = self.get_prefix_probabilities()
                
                # Ordina i prefissi per frequenza decrescente
                sorted_prefixes = sorted(prefix_probs.items(), 
                                        key=lambda x: x[1], reverse=True)[:20]
                
                # Per ogni prefisso frequente, mostra statistiche dettagliate
                for prefix, prob_prefix in sorted_prefixes:
                    # Rappresenta il prefisso con quotes per visibilità
                    display_prefix = repr(prefix)
                    
                    # Calcola entropia condizionata per questo prefisso
                    cond_entropy = self.get_conditional_entropy(prefix)
                    
                    print(f"\nPrefisso {display_prefix}:")
                    print(f"  P({display_prefix}) = {prob_prefix:.6f}")
                    print(f"  H(X|{display_prefix}) = {cond_entropy:.6f} bit")

        def print_latex_report(self):
            """
            Stampa un rapporto completo dell'analisi in formato latex.
            
            Il rapporto include:
            - Statistiche generali (lunghezza, caratteri unici, prefissi unici)
            - Probabilità di tutti i caratteri
            - Entropia assoluta H(X)
            - Entropia condizionata media H(X|X^k)  [non applicabile con k=-1]
            - Dettagli sui prefissi con relative entropie condizionate [non con k=-1]
            """
            # Sezione informazioni generali
            if self.k == -1:
                print("ANALISI ENTROPICA DEL TESTO (modello uniforme, k=-1)")
            else:
                print("ANALISI ENTROPICA DEL TESTO")
            print("\\begin{tabular}{|l|c|}")
            print(f"Lunghezza testo & {len(self.text)} \\\\")
            if self.k == -1:
                print(f"Modello & uniforme (k=-1) \\\\")
            else:
                print(f"Lunghezza prefissi & {self.k} \\\\")
            print(f"Numero caratteri unici & {len(self.char_counts)}\\\\")
            if self.k != -1:
                print(f"Numero prefissi unici & {len(self.prefix_counts)}")
            print("\\end{tabular}")
            
            # Sezione probabilità dei caratteri
            print("PROBABILITÀ DEI CARATTERI")
            char_probs = self.get_char_probabilities()
            print("\\begin{tabular}{|c|c|c|c}")
            print("carattere & occorrenze & probabilità & informazione\\\\")
            print("\\hline")
            # Ordina per frequenza decrescente per migliore leggibilità
            sorted_chars = sorted(char_probs.items(), 
                                    key=lambda x: x[1], reverse=True)
            for char, prob_char in sorted_chars:
                display_char = repr(char)
                information = -math.log2(prob_char) if prob_char > 0 else 0
                print(f"'{char}' & {self.char_counts[char]} & {prob_char:.6f} & {information:.6f}\\\\")
            print("\\end{tabular}")
            
            # Sezione misure di entropia
            print("MISURE DI ENTROPIA")
            entropy = self.get_entropy()
            
            if self.k == -1:
                print(f"Entropia uniforme $H(X)$: {entropy:.6f} bit")
                print(f"Riduzione di entropia: 0.000000 (0.00\\%)")
            else:
                print(f"Entropia $H(X)$: {entropy:.6f} bit")
                
                avg_cond_entropy = self.get_average_conditional_entropy()
                
                # Sezione prefissi e probabilità condizionate
                print("\n" + "-" * 70)
                print(f"PROBABILITÀ PREFISSI E ENTROPIE CONDIZIONATE")
                print("-" * 70)
                print("\\begin{tabular}{|c|c}")
                print("prefisso & entropia condizionata\\\\")
                print("\\hline")
                prefix_probs = self.get_prefix_probabilities()
                
                # Ordina i prefissi per frequenza decrescente e prende i primi 20
                sorted_prefixes = sorted(prefix_probs.items(), 
                                        key=lambda x: x[1], reverse=True)
                
                # Per ogni prefisso frequente, mostra statistiche dettagliate
                for prefix, prob_prefix in sorted_prefixes:
                    # Rappresenta il prefisso con quotes per visibilità
                    display_prefix = repr(prefix)
                    
                    # Calcola entropia condizionata per questo prefisso
                    cond_entropy = self.get_conditional_entropy(prefix)
                    
                    print(f"{display_prefix} & {cond_entropy:.6f}\\\\")
                print("\\end{tabular}")
                print(f"Entropia condizionata media $H(X|X^{self.k}$): {avg_cond_entropy:.6f}")
    # Fine della classe interna TextEntropyAnalyzer
    
    def __init__(self, filepath, k=1):
        """
        Inizializza il modello di linguaggio.
        
        Crea un'istanza embedded di TextEntropyAnalyzer che analizza il file
        di testo e calcola tutte le statistiche necessarie per la generazione.
        
        Args:
            filepath (str o Path): Percorso del file di testo da analizzare
            k (int, optional): Lunghezza del contesto (prefisso).
                               Usare -1 per il modello uniforme (alfabeto equiprobabile).
                               Default: 1
        
        Esempio:
            >>> model = LanguageModel("divina_commedia.txt", k=3)
            >>> print(f"Entropia: {model.get_entropy():.3f}")
            >>> text = model.generate_text(100, temperature=0.2)
            >>> # Modello uniforme:
            >>> model_uni = LanguageModel("divina_commedia.txt", k=-1)
        """
        # Memorizza la lunghezza del contesto
        self.k = k
        
        # Crea l'istanza embedded dell'analizzatore
        # Questo analizza il testo e calcola tutte le statistiche
        self.analyzer = self.TextEntropyAnalyzer(filepath, k)
        
        # Estrae la lista di tutti i caratteri possibili (per campionamento uniforme)
        self.all_chars = list(self.analyzer.char_counts.keys())
    
    # ========================================================================
    # METODI WRAPPER - Delegano all'istanza embedded di TextEntropyAnalyzer
    # ========================================================================
    
    def get_char_probabilities(self):
        """
        Restituisce le probabilità di occorrenza dei caratteri.
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
        Returns:
            dict: Dizionario {carattere: probabilità}
        """
        return self.analyzer.get_char_probabilities()
    
    def get_prefix_probabilities(self):
        """
        Restituisce le probabilità di occorrenza dei prefissi.
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
        Returns:
            dict: Dizionario {prefisso: probabilità}
        """
        return self.analyzer.get_prefix_probabilities()
    
    def get_conditional_probabilities(self, prefix):
        """
        Restituisce le probabilità condizionate per un prefisso.
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
        Args:
            prefix (str): Il prefisso da condizionare
        
        Returns:
            dict: Dizionario {carattere: P(carattere|prefisso)}
        """
        return self.analyzer.get_conditional_probabilities(prefix)
    
    def calculate_entropy(self, probabilities):
        """
        Calcola l'entropia da un dizionario di probabilità.
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
        Args:
            probabilities (dict): Dizionario di probabilità
        
        Returns:
            float: Entropia in bit
        """
        return self.analyzer.calculate_entropy(probabilities)
    
    def get_entropy(self):
        """
        Calcola l'entropia del testo H(X).
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
        Returns:
            float: Entropia in bit
        """
        return self.analyzer.get_entropy()
    
    def get_conditional_entropy(self, prefix):
        """
        Calcola l'entropia condizionata per un prefisso.
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
        Args:
            prefix (str): Il prefisso da condizionare
        
        Returns:
            float: Entropia condizionata in bit
        """
        return self.analyzer.get_conditional_entropy(prefix)
    
    def get_average_conditional_entropy(self):
        """
        Calcola l'entropia condizionata media H(X|X^k).
        
        Delega al metodo corrispondente dell'analizzatore embedded.
        
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
    
    def get_text_length(self):
        """
        Restituisce la lunghezza del testo analizzato.
        
        Returns:
            int: Numero di caratteri nel testo pulito
        """
        return len(self.analyzer.text)
    
    # ========================================================================
    # METODI PROPRI DI LANGUAGEMODEL - Generazione di testo
    # ========================================================================
    
    def _get_prefix(self, text):
        """
        Estrae il prefisso rilevante da un testo.
        
        Se il testo è più lungo di k, prende gli ultimi k caratteri.
        Se è più corto, aggiunge padding di spazi all'inizio.
        
        Args:
            text (str): Testo da cui estrarre il prefisso
        
        Returns:
            str: Prefisso di lunghezza esattamente k
        
        Esempio:
            Se k=3 e text="hello", ritorna "llo"
            Se k=3 e text="hi", ritorna " hi"
        """
        if len(text) >= self.k:
            # Prende gli ultimi k caratteri
            return text[-self.k:]
        else:
            # Aggiunge spazi all'inizio per raggiungere lunghezza k
            # Questo permette di generare anche i primi caratteri
            padding = ' ' * (self.k - len(text))
            return padding + text
    
    def _sample_from_distribution(self, probs):
        """
        Campiona un carattere da una distribuzione di probabilità.
        
        Usa il metodo della ruota della fortuna (roulette wheel selection):
        genera un numero casuale e trova in quale intervallo cade.
        
        Args:
            probs (dict): Dizionario {carattere: probabilità}
        
        Returns:
            str: Carattere campionato
        
        Note:
            Assume che sum(probs.values()) ≈ 1.0
        """
        # Genera un numero casuale tra 0 e 1
        r = random.random()
        
        # Accumula probabilità finché non supera il valore random
        cumulative = 0.0
        for char, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return char
        
        # Fallback: ritorna l'ultimo carattere (dovrebbe essere raro)
        # Può succedere a causa di errori di arrotondamento
        return list(probs.keys())[-1]
    
    def next_char(self, text, temperature=0.0):
        """
        Campiona il prossimo carattere data una stringa.
        
        Con k=-1 (modello uniforme) campiona sempre uniformemente dall'alfabeto,
        ignorando sia il contesto che la temperatura.
        
        Altrimenti:
        1. Estrae il prefisso rilevante dal testo (ultimi k caratteri o con padding)
        2. Con probabilità 'temperature', campiona uniformemente da tutti i caratteri
        3. Altrimenti, campiona dalla distribuzione condizionata P(X | prefisso)
        4. Se il prefisso non è mai stato visto, usa la distribuzione assoluta P(X)
        
        Args:
            text (str): Testo corrente (contesto)
            temperature (float): Parametro di randomizzazione, deve essere in [0, 1]
                                - 0.0: completamente deterministico (usa solo P(X|prefix))
                                - 1.0: completamente casuale (distribuzione uniforme)
                                - valori intermedi: mix tra distribuzione e casualità
                                (ignorato con k=-1)
        
        Returns:
            str: Il prossimo carattere generato
        
        Raises:
            ValueError: Se temperature non è in [0, 1] (solo per k ≥ 1)
        
        Esempio:
            >>> model = LanguageModel("testo.txt", k=2)
            >>> model.next_char("hel", temperature=0.0)  # Potrebbe ritornare 'l'
            >>> model.next_char("hel", temperature=1.0)  # Potrebbe ritornare qualsiasi carattere
        """
        no_match = 0
        # Modello uniforme: campionamento diretto dall'alfabeto, nessun contesto
        if self.k == -1:
            return random.choice(self.all_chars), no_match
        
        # Valida il parametro temperatura
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature deve essere compreso tra 0 e 1")
        
        # Estrae il prefisso rilevante (ultimi k caratteri o con padding)
        prefix = self._get_prefix(text)
        
        # Decide se usare campionamento casuale basandosi sulla temperatura
        # random() genera un numero in [0,1), quindi se temperature=0.3,
        # il 30% delle volte userà campionamento uniforme
        if random.random() < temperature:
            # Campionamento completamente casuale (distribuzione uniforme)
            no_match = 1
            return random.choice(self.all_chars), no_match
        
        # Altrimenti, usa la distribuzione condizionata
        # Controlla se abbiamo mai visto questo prefisso nei dati di training
        if prefix in self.analyzer.conditional_counts and self.analyzer.prefix_counts[prefix] > 0:
            # Il prefisso è conosciuto: usa P(X | prefisso)
            
            # Calcola le probabilità condizionate usando l'analizzatore
            cond_probs = self.get_conditional_probabilities(prefix)
            
            # Campiona dalla distribuzione condizionata
            return self._sample_from_distribution(cond_probs), no_match
        else:
            # Il prefisso non è mai stato visto: fallback alla distribuzione assoluta
            # Usa P(X) ignorando il contesto
            if self.k > 0:
                no_match = 1
            char_probs = self.get_char_probabilities()
            return self._sample_from_distribution(char_probs), no_match
    
    def generate_text(self, length=100, temperature=0.0, seed_text=""):
        """
        Genera una sequenza di testo di lunghezza specificata.
        
        Genera carattere per carattere usando next_char(), partendo da un testo
        iniziale (seed) o da una stringa vuota.
        
        Args:
            length (int): Numero di caratteri da generare
            temperature (float): Parametro di randomizzazione in [0, 1]
                                - 0.0: generazione deterministica
                                - 1.0: generazione completamente casuale
                                - valori intermedi: bilanciano coerenza e varietà
            seed_text (str): Testo iniziale da cui partire (opzionale)
        
        Returns:
            str: Testo generato
        
        Esempio:
            >>> model = LanguageModel("divina_commedia.txt", k=3)
            >>> text = model.generate_text(length=200, temperature=0.2, seed_text="nel ")
            >>> print(text)
            "nel mezzo del cammin di nostra vita..."
        """
        no_match_counter = 0
        # Inizializza il testo con il seed fornito
        generated = seed_text
        
        # Genera carattere per carattere
        for _ in range(length):
            # Ottiene il prossimo carattere basandosi sul testo corrente
            next_ch, no_match = self.next_char(generated, temperature=temperature)
            no_match_counter += no_match
            # Aggiunge il carattere al testo generato
            generated += next_ch
        
        return generated, no_match_counter


def print_menu():
    """
    Stampa il menu principale dei comandi disponibili.
    
    Il menu viene mostrato ad ogni iterazione del ciclo principale,
    permettendo all'utente di scegliere quale operazione eseguire.
    """
    print("\n" + "=" * 70)
    print("MENU PRINCIPALE - LANGUAGE MODEL")
    print("=" * 70)
    print("Comandi disponibili:")
    print("  1. genera    - Genera testo con il modello")
    print("  2. report    - Mostra il rapporto di analisi entropica completo")
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
    
    Raises:
        ValueError: Se la conversione al tipo fallisce
    """
    while True:
        try:
            # Mostra il prompt con indicazione del default se presente
            if default is not None:
                user_input = input(f"{prompt} [default: {default}]: ").strip()
                # Se l'utente preme solo Invio, usa il default
                if user_input == "":
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            # Converte al tipo richiesto
            value = input_type(user_input)
            
            # Applica validazione personalizzata se fornita
            if validate is not None and not validate(value):
                print("Valore non valido. Riprova.")
                continue
            
            return value
            
        except ValueError:
            print(f"Errore: inserire un valore di tipo {input_type.__name__}. Riprova.")


def command_generate(model):
    """
    Esegue il comando di generazione testo.
    
    Richiede all'utente i parametri necessari (lunghezza, temperatura, seed)
    e genera una sequenza di testo usando il modello di linguaggio.
    Con k=-1 (modello uniforme) il parametro temperatura non viene richiesto,
    poiché la generazione è intrinsecamente uniforme.
    
    Args:
        model (LanguageModel): Il modello di linguaggio da usare per la generazione
    """
    print("\n" + "-" * 70)
    print("GENERAZIONE DI TESTO")
    if model.k == -1:
        print("  (modello uniforme: campionamento equiprobabile sull'alfabeto)")
    print("-" * 70)
    
    # Richiede la lunghezza del testo da generare
    length = get_user_input(
        "Lunghezza del testo (numero di caratteri)",
        input_type=int,
        default=100,
        validate=lambda x: x > 0  # Deve essere positivo
    )
    
    # La temperatura non ha significato per il modello uniforme
    if model.k == -1:
        temperature = 0.0
    else:
        # Richiede la temperatura (parametro di randomizzazione)
        temperature = get_user_input(
            "Temperatura (0.0 = deterministico, 1.0 = casuale)",
            input_type=float,
            default=0.0,
            validate=lambda x: 0.0 <= x <= 1.0  # Deve essere in [0, 1]
        )
    
    # Richiede il testo seed (opzionale)
    seed_text = get_user_input(
        "Testo iniziale (seed, premi Invio per nessun seed)",
        input_type=str,
        default=""
    )
    
    # Mostra i parametri scelti
    print("\n" + "-" * 70)
    print("Parametri di generazione:")
    print(f"  Lunghezza: {length} caratteri")
    if model.k != -1:
        print(f"  Temperatura: {temperature}")
    if seed_text:
        print(f"  Seed: \"{seed_text}\"")
    else:
        print(f"  Seed: (nessuno)")
    print("-" * 70)
    print("Generazione in corso...\n")
    
    # Genera il testo usando il modello
    try:
        generated, no_match_counter = model.generate_text(
            length=length,
            temperature=temperature,
            seed_text=seed_text
        )
        
        # Stampa il risultato
        print("TESTO GENERATO:")
        print("-" * 70)
        print(generated)
        print("-" * 70)
        print(f"Numero di caratteri generati con prefisso non osservato: {no_match_counter} (usati P(X) invece di P(X|prefix))")
        print(f"Frazione di caratteri generati con prefisso non osservato: {no_match_counter/len(generated)} (usati P(X) invece di P(X|prefix))")
        
    except Exception as e:
        print(f"Errore durante la generazione: {e}")


def command_report(model):
    """
    Esegue il comando di stampa del rapporto di analisi.
    
    Mostra il rapporto completo con tutte le statistiche entropiche,
    probabilità dei caratteri, e analisi dei prefissi più frequenti.
    
    Args:
        model (LanguageModel): Il modello di linguaggio con i dati da analizzare
    """
    print()
    # Delega al metodo print_report del modello
    model.print_report()


def command_info(model):
    """
    Esegue il comando di visualizzazione informazioni base.
    
    Mostra un sommario rapido delle informazioni principali sul modello
    senza entrare nei dettagli completi del rapporto.
    
    Args:
        model (LanguageModel): Il modello di linguaggio da ispezionare
    """
    print("\n" + "=" * 70)
    print("INFORMAZIONI SUL MODELLO")
    print("=" * 70)
    
    # Informazioni generali
    if model.k == -1:
        print(f"\nOrdine del modello (k): -1  (modello uniforme)")
    else:
        print(f"\nOrdine del modello (k): {model.k}")
    print(f"Lunghezza del testo analizzato: {model.get_text_length()} caratteri")
    print(f"Numero di caratteri unici: {len(model.analyzer.char_counts)}")
    if model.k != -1:
        print(f"Numero di prefissi unici: {len(model.analyzer.prefix_counts)}")
    
    # Misure di entropia
    print("\nMisure di entropia:")
    entropy = model.get_entropy()
    
    if model.k == -1:
        print(f"  H(X) uniforme = {entropy:.4f} bit  (= log₂({len(model.analyzer.char_counts)}))")
        print(f"  Nessun contesto: riduzione = 0.0000 bit (0.0%)")
    else:
        avg_cond_entropy = model.get_average_conditional_entropy()
        print(f"  H(X) = {entropy:.4f} bit")
        print(f"  H(X|X^{model.k}) = {avg_cond_entropy:.4f} bit")
        
        # Calcola la riduzione di entropia dovuta al contesto
        reduction = entropy - avg_cond_entropy
        reduction_pct = (reduction / entropy * 100) if entropy > 0 else 0
        print(f"  Riduzione entropia: {reduction:.4f} bit ({reduction_pct:.1f}%)")
    
    print("=" * 70)


def main():
    """
    Funzione principale per l'esecuzione interattiva del programma.
    
    Il programma funziona in due modalità:
    1. Modalità iniziale: carica il modello di linguaggio da file
    2. Modalità interattiva: mostra un menu e accetta comandi dall'utente
    
    Uso:
        python language_model.py <filepath> [k]
    
    Argomenti:
        filepath: percorso del file di testo da analizzare
        k: lunghezza dei prefissi (ordine del modello, default 1)
           Usare -1 per il modello uniforme (alfabeto equiprobabile)
    
    Esempi:
        python language_model.py testo.txt
        python language_model.py divina_commedia.txt 3
        python language_model.py divina_commedia.txt -1
    
    Dopo il caricamento, il programma entra in modalità interattiva dove
    l'utente può eseguire vari comandi fino a quando sceglie di uscire.
    """
    import sys
    
    # Controlla gli argomenti minimi
    if len(sys.argv) < 2:
        print("=" * 70)
        print("LANGUAGE MODEL - Analisi e Generazione di Testo")
        print("=" * 70)
        print("\nUtilizzo: python language_model.py <filepath> [k]")
        print("\nArgomenti:")
        print("  filepath: percorso del file di testo da analizzare")
        print("  k: lunghezza dei prefissi (ordine del modello, default 1)")
        print("     Usare k=-1 per il modello uniforme (alfabeto equiprobabile)")
        print("\nEsempi:")
        print("  python language_model.py testo.txt")
        print("  python language_model.py divina_commedia.txt 3")
        print("  python language_model.py divina_commedia.txt -1")
        print("\nDopo il caricamento, il programma entrerà in modalità interattiva")
        print("dove potrai eseguire vari comandi come generare testo, visualizzare")
        print("report di analisi, e altro.")
        print("=" * 70)
        sys.exit(1)
    
    # Legge gli argomenti dalla linea di comando
    filepath = sys.argv[1]
    # Se viene fornito k come secondo argomento, lo usa; altrimenti default a 1
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    # Valida k: deve essere -1 oppure un intero positivo
    if k < -1:
        print(f"✗ Errore: k deve essere un intero non negativo oppure -1 (modello uniforme).")
        sys.exit(1)
    
    # ========================================================================
    # FASE 1: CARICAMENTO E INIZIALIZZAZIONE DEL MODELLO
    # ========================================================================
    
    print("=" * 70)
    if k == -1:
        print(f"LANGUAGE MODEL: File {filepath}, distribuzione uniforme dei simboli")
    else:
        print(f"LANGUAGE MODEL: File {filepath}, modello di ordine {k}")
    print("=" * 70)
    print("Analisi del testo in corso...")
    
    try:
        # Crea il modello di linguaggio
        # Questo crea automaticamente l'analizzatore embedded e calcola le statistiche
        model = LanguageModel(filepath, k)
        
        print(f"✓ Lunghezza testo: {model.get_text_length()} caratteri")
        print(f"✓ {len(model.analyzer.char_counts)} caratteri distinti")
        if k == -1:
            print(f"✓ Probabilità p uniforme pari a 1/{len(model.analyzer.char_counts)}={1/len(model.analyzer.char_counts):.6f} per ogni simbolo")
        elif k == 0:
            print(f"✓ Probabilità p(x) non basata su prefissi (k=0), ma su frequenze assolute")
        elif k==1:
            print(f"✓ Probabilità p(x|x_1) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.char_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.char_counts)**k*100:.2f}% del totale")
        elif k==2:
            print(f"✓ Probabilità p(x|x_1x_2) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.char_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.char_counts)**k*100:.2f}% del totale")
        elif k==3:
            print(f"✓ Probabilità p(x|x_1x_2x_3) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.char_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.char_counts)**k*100:.2f}% del totale")
        else:
            print(f"✓ Probabilità p(x|x_1,...,x_{k}) basata su prefissi di lunghezza {k}")
            print(f"✓ {len(model.analyzer.char_counts)**k} prefissi possibili")
            print(f"✓ {len(model.analyzer.conditional_counts)} prefissi osservati, " 
                f"{len(model.analyzer.conditional_counts)/len(model.analyzer.char_counts)**k*100:.2f}% del totale")
        
    except FileNotFoundError:
        print(f"\n✗ Errore: file '{filepath}' non trovato")
        print("Verifica che il percorso sia corretto e riprova.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Errore durante il caricamento: {e}")
        sys.exit(1)
    
    # ========================================================================
    # FASE 2: CICLO INTERATTIVO PRINCIPALE
    # ========================================================================
    
    # Dizionario che mappa i comandi alle funzioni corrispondenti
    commands = {
        '1': ('genera', command_generate),
        '2': ('report', command_report),
        '3': ('info', command_info),
        '4': ('esci', None)
    }
    
    # Ciclo principale del programma
    while True:
        # Mostra il menu
        print_menu()
        
        # Richiede un comando all'utente
        user_input = input("\nInserisci un comando: ").strip().lower()
        
        # Controlla se il comando è valido
        if user_input in commands:
            command_name, command_func = commands[user_input]
            
            # Se il comando è 'esci', termina il programma
            if command_name == 'esci':
                print("\n" + "=" * 70)
                print("Chiusura del programma...")
                print("Arrivederci!")
                print("=" * 70)
                break
            
            # Altrimenti esegue la funzione associata al comando
            try:
                command_func(model)
            except KeyboardInterrupt:
                # Permette all'utente di interrompere un comando con Ctrl+C
                print("\n\nComando interrotto dall'utente.")
            except Exception as e:
                # Gestisce errori imprevisti senza far crashare il programma
                print(f"\n✗ Errore durante l'esecuzione del comando: {e}")
        
        else:
            # Comando non riconosciuto
            print(f"\n✗ Comando '{user_input}' non riconosciuto.")
            print("Usa uno dei comandi del menu (1-4, oppure il nome completo).")


if __name__ == "__main__":
    main()