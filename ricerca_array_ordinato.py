#%%
"""
ricerca_array_ordinato.py
================
Ricerca binaria ottimale in un array ordinato usando la teoria dell'informazione.

ENUNCIATO DEL PROBLEMA
----------------------
Un array di N interi distinti **ordinati in modo crescente** contiene un valore 
bersaglio T la cui posizione è ignota. Usando confronti a tre vie 
(minore / uguale / maggiore) tra T e un elemento scelto dell'array, trovare 
l'indice di T nel minor numero di confronti possibile.

ANALOGIA CON IL PROBLEMA DELLE 12 PALLINE
------------------------------------------
  Problema delle palline          Ricerca nell'array ordinato
  ─────────────────────────────   ─────────────────────────────────────
  Pallina anomala (indice, tipo)  Posizione del bersaglio (indice)
  Pesata su bilancia (sx, dx)     Confronto con arr[i]
  Esiti: SX+ / Pari / DX+        Esiti: T < arr[i] / T = arr[i] / T > arr[i]
  24 stati iniziali               N stati iniziali

APPROCCIO
---------
Ad ogni passo viene calcolato l'*information gain* (guadagno informativo) di 
ogni confronto candidato tramite l'entropia di Shannon. Il confronto con il 
massimo guadagno informativo viene preferito.

Per **array ordinati**, la strategia ottimale coincide con la **ricerca binaria**:
confronto con l'elemento centrale, garantendo al massimo ⌈log₂ N⌉ confronti.

L'algoritmo permette di verificare questa proprietà in modalità interattiva,
mostrando che il confronto ottimale (secondo la teoria dell'informazione) 
corrisponde sempre all'elemento centrale per array ordinati.

STRUTTURA DEL MODULO
--------------------
  1. Costanti globali
  2. Funzioni di generazione (array, stati, istanza, confronti)
  3. Funzioni di calcolo (entropia, simulazione confronto, analisi, valutazione)
  4. Funzioni di formattazione output
  5. Funzioni di I/O e interfaccia utente
  6. Funzione principale `risolvi()`

Utilizzo da riga di comando::

    python ricerca_array_ordinato.py
"""

# ---------------------------------------------------------------------------
# Import della libreria standard
# ---------------------------------------------------------------------------
import math    # Funzioni matematiche (log2 per calcolo entropia)
import random  # Generazione casuale dell'array e dell'istanza
from collections import defaultdict        # Dizionario con valore di default per raggruppamenti
from typing import Dict, List, Optional, Tuple  # Type hints per documentazione


# ===========================================================================
# 1. COSTANTI GLOBALI
# ===========================================================================

# Dimensione predefinita dell'array se l'utente non la specifica
DIM_DEFAULT: int = 16

# Intervallo di campionamento dei valori interi distinti dell'array.
# I valori vengono scelti da [1, FATTORE_RANGE * DIM_DEFAULT], assicurando
# che ci siano sempre abbastanza interi distinti da campionare.
FATTORE_RANGE: int = 10

# Quanti confronti "migliori per IG medio" mostrare in modalità interattiva
TOP_K_MEDIA: int = 3

# Quanti confronti "migliori per worst-case" mostrare in modalità interattiva
TOP_M_WORST: int = 3

# Esiti possibili di un confronto tra il bersaglio T e arr[i]
ESITO_MINORE_UGUALE:   str = "<="  # T  <  arr[i]  → cercare a sinistra
ESITO_MAGGIORE: str = ">"  # T  >  arr[i]  → cercare a destra


# ===========================================================================
# 2. FUNZIONI DI GENERAZIONE
# ===========================================================================

def genera_array(n: int, ordinato: bool = False) -> List[int]:
    """
    Genera un array di ``n`` interi distinti scelti casualmente.

    Campiona n valori distinti dall'intervallo [1, FATTORE_RANGE * n] e
    opzionalmente li ordina in modo crescente.

    Parameters
    ----------
    n : int
        Numero di elementi dell'array.
    ordinato : bool, optional
        Se ``True`` l'array viene restituito in ordine crescente,
        altrimenti in ordine casuale. Default: ``False``.
        **Nota**: Questo script usa sempre ``ordinato=True``.

    Returns
    -------
    List[int]
        Lista di ``n`` interi distinti nell'intervallo
        ``[1, FATTORE_RANGE * n]``.

    Examples
    --------
    >>> random.seed(0)
    >>> genera_array(6)
    [37, 12, 72, 9, 75, 5]            # esempio: ordine casuale

    >>> random.seed(0)
    >>> genera_array(6, ordinato=True)
    [5, 9, 12, 37, 72, 75]            # stesso campione, ordinato
    """
    # Campiona n interi distinti dall'intervallo [1, FATTORE_RANGE * n]
    # random.sample garantisce che tutti i valori siano distinti
    valori = random.sample(range(1, FATTORE_RANGE * n + 1), n)

    # Ordina i valori se richiesto, altrimenti li lascia nell'ordine casuale
    if ordinato:
        valori.sort()  # Ordine crescente in-place

    return valori


def genera_stati(n: int) -> List[int]:
    """
    Genera la lista di tutti gli stati iniziali possibili.

    Uno *stato* è l'indice che il bersaglio T potrebbe occupare nell'array.
    All'inizio tutti gli N indici sono candidati equiprobabili.

    Parameters
    ----------
    n : int
        Numero di elementi dell'array (= numero di stati iniziali).

    Returns
    -------
    List[int]
        Lista ``[0, 1, 2, ..., n-1]`` che rappresenta tutti i possibili
        indici dell'array.

    Examples
    --------
    >>> genera_stati(5)
    [0, 1, 2, 3, 4]
    >>> len(genera_stati(16))
    16
    """
    # Crea la lista degli indici da 0 a n-1 (un indice per ogni cella dell'array)
    return list(range(n))


def genera_istanza(n: int) -> int:
    """
    Genera casualmente la "soluzione vera": l'indice reale del bersaglio.

    Simula la scelta casuale della posizione dove si trova effettivamente
    il valore cercato nell'array. Questa posizione rimane nascosta durante
    la ricerca e viene usata solo per simulare gli esiti dei confronti.

    Parameters
    ----------
    n : int
        Numero di elementi dell'array.

    Returns
    -------
    int
        Indice intero nell'intervallo ``[0, n-1]`` scelto uniformemente.

    Examples
    --------
    >>> random.seed(7)
    >>> genera_istanza(16)
    11                    # il bersaglio si trova in posizione 11
    """
    # Sceglie uniformemente a caso un indice valido nell'array [0, n-1]
    return random.randint(0, n - 1)


def genera_confronti(n: int) -> List[int]:
    """
    Genera la lista di tutti i confronti disponibili.

    Un *confronto* è un indice ``i`` tale che l'algoritmo può chiedere
    "T è minore, uguale o maggiore di arr[i]?". Ogni indice dell'array
    è un confronto valido, indipendentemente dal fatto che sia ancora
    uno stato candidato.

    In teoria dell'informazione, ogni confronto è un "esperimento" che
    fornisce informazione per ridurre l'incertezza.

    Parameters
    ----------
    n : int
        Numero di elementi dell'array.

    Returns
    -------
    List[int]
        Lista ``[0, 1, 2, ..., n-1]`` contenente tutti gli indici disponibili.

    Examples
    --------
    >>> genera_confronti(4)
    [0, 1, 2, 3]
    """
    # Ogni indice dell'array può essere scelto come punto di confronto
    return list(range(n))


# ===========================================================================
# 3. FUNZIONI DI CALCOLO E VALUTAZIONE
# ===========================================================================

def entropia(stati: List[int]) -> float:
    """
    Calcola l'entropia di Shannon di un insieme di stati equiprobabili.

    L'entropia misura l'incertezza: quanti bit di informazione servono
    per identificare univocamente uno stato tra tutti i candidati.
    
    Con distribuzione uniforme su ``n`` stati::

        H = log₂(n)

    Parameters
    ----------
    stati : List[int]
        Lista degli indici candidati.

    Returns
    -------
    float
        Entropia in bit. ``0.0`` se la lista è vuota o ha un solo elemento
        (problema risolto), ``log2(len(stati))`` altrimenti.

    Examples
    --------
    >>> entropia([0, 1, 2, 3])
    2.0                        # log₂(4) = 2 bit
    >>> entropia([5])
    0.0                        # un solo candidato = nessuna incertezza
    >>> entropia([])
    0.0                        # lista vuota = nessuna incertezza
    """
    # Numero di stati candidati
    n = len(stati)
    
    # Se non ci sono stati o c'è un solo candidato, l'incertezza è zero
    if n <= 1:
        return 0.0
    
    # Formula dell'entropia per distribuzione uniforme: H = log₂(n)
    return math.log2(n)


def confronta(idx_vero: int, arr: List[int], idx_test: int) -> str:
    """
    Simula il confronto tra il valore bersaglio T e arr[idx_test].

    Dato che l'indice vero del bersaglio è ``idx_vero``, confronta
    ``arr[idx_vero]`` con ``arr[idx_test]`` e restituisce l'esito
    come una delle costanti ESITO_*.

    **Nota**: Questa funzione "bara" guardando la soluzione vera, ma
    simula ciò che accadrebbe in un confronto reale con il valore T.

    Parameters
    ----------
    idx_vero : int
        Indice reale (nascosto) dove si trova il bersaglio nell'array.
    arr : List[int]
        Array ordinato di valori.
    idx_test : int
        Indice dell'elemento con cui confrontare il bersaglio.

    Returns
    -------
    str
        Una delle costanti: ESITO_MINORE, ESITO_UGUALE, ESITO_MAGGIORE.

    Examples
    --------
    >>> arr = [10, 20, 30, 40, 50]
    >>> confronta(2, arr, 1)  # T=30, arr[1]=20 → T > arr[1]
    '>'
    >>> confronta(2, arr, 2)  # T=30, arr[2]=30 → T = arr[2]
    '='
    >>> confronta(2, arr, 4)  # T=30, arr[4]=50 → T < arr[4]
    '<'
    """
    # Valore del bersaglio (posizione vera)
    val_vero = arr[idx_vero]
    
    # Valore dell'elemento testato
    val_test = arr[idx_test]

    # Confronto a tre vie
    if val_vero <= val_test:
        return ESITO_MINORE_UGUALE      # Bersaglio a sinistra o trovato
    else:
        return ESITO_MAGGIORE    # Bersaglio a destra


def analizza(
    stati: List[int], 
    arr: List[int], 
    idx_confronto: int
) -> Tuple[Dict[str, List[int]], Dict[str, int], float, float]:
    """
    Analizza gli esiti possibili di un confronto con arr[idx_confronto].

    Per ogni stato candidato, determina quale sarebbe l'esito del confronto
    (< = >) e raggruppa gli stati per esito. Calcola poi l'entropia media
    e l'entropia worst-case risultanti.

    **Per array ordinati**: 
    - Esito "<" → stati a sinistra di idx_confronto
    - Esito "=" → solo idx_confronto (se è candidato)
    - Esito ">" → stati a destra di idx_confronto

    Parameters
    ----------
    stati : List[int]
        Lista degli stati candidati correnti.
    arr : List[int]
        Array ordinato di valori.
    idx_confronto : int
        Indice dell'elemento da usare per il confronto.

    Returns
    -------
    gruppi : Dict[str, List[int]]
        Dizionario che mappa ogni esito ("<", "=", ">") alla lista
        degli stati che produrrebbero quell'esito.
    distribuzione : Dict[str, int]
        Dizionario che mappa ogni esito al numero di stati in quel gruppo.
    entropia_media : float
        Entropia media pesata sui tre gruppi (guadagno informativo atteso).
    entropia_worst : float
        Entropia massima tra i tre gruppi (caso peggiore).

    Examples
    --------
    >>> arr = [10, 20, 30, 40, 50]
    >>> stati = [0, 1, 2, 3, 4]
    >>> gruppi, dist, h_media, h_max = analizza(stati, arr, 2)
    >>> gruppi
    {'<': [0, 1], '=': [2], '>': [3, 4]}
    >>> dist
    {'<': 2, '=': 1, '>': 2}
    """
    # Inizializza i gruppi per i tre possibili esiti
    gruppi: Dict[str, List[int]] = defaultdict(list)

    # Per ogni stato candidato, simula quale sarebbe l'esito del confronto
    for s in stati:
        # Simula il confronto tra arr[s] e arr[idx_confronto]
        esito = confronta(s, arr, idx_confronto)
        # Aggiungi questo stato al gruppo corrispondente
        gruppi[esito].append(s)

    # Converte defaultdict a dict normale per chiarezza
    gruppi = dict(gruppi)

    # Calcola la distribuzione: numero di stati per ciascun esito
    distribuzione = {k: len(v) for k, v in gruppi.items()}

    # Numero totale di stati candidati
    n_tot = len(stati)

    # Calcola l'entropia media pesata (guadagno informativo atteso)
    entropia_media = 0.0
    for esito, lista_stati in gruppi.items():
        # Numero di stati in questo gruppo
        n_gruppo = len(lista_stati)
        # Probabilità di questo esito (assumendo distribuzione uniforme)
        p = n_gruppo / n_tot
        # Entropia di questo gruppo
        h_gruppo = entropia(lista_stati)
        # Contributo pesato all'entropia media
        entropia_media += p * h_gruppo

    # Calcola l'entropia worst-case: il massimo tra tutti i gruppi
    if gruppi:
        # Trova l'entropia massima tra tutti i gruppi non vuoti
        entropia_worst = max(entropia(v) for v in gruppi.values())
    else:
        # Nessun gruppo: entropia zero
        entropia_worst = 0.0

    return gruppi, distribuzione, entropia_media, entropia_worst


def valuta(
    stati: List[int], 
    confronti: List[int],
    arr: List[int]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Valuta tutti i confronti possibili e li ordina per metriche di qualità.

    Per ogni confronto in ``confronti``, calcola:
    - Information gain medio (IG medio)
    - Entropia worst-case
    
    Restituisce due liste ordinate: una per IG medio (decrescente),
    una per worst-case (crescente).

    Parameters
    ----------
    stati : List[int]
        Stati candidati correnti.
    confronti : List[int]
        Indici disponibili per i confronti.

    Returns
    -------
    per_media : List[Dict]
        Lista di dizionari ordinata per IG medio decrescente.
        Ogni dizionario contiene: idx, valore, ig_medio, h_max, gruppi, dist.
    per_worst : List[Dict]
        Lista di dizionari ordinata per entropia worst-case crescente.
        Ogni dizionario contiene: idx, valore, h_max, ig_medio, gruppi, dist.

    Notes
    -----
    **Per array ordinati**, il confronto ottimale (quello centrale) apparirà
    in cima a entrambe le liste, dimostrando che la ricerca binaria è
    ottimale sia per IG medio che per worst-case.
    """
    # Entropia corrente prima del confronto
    h_corrente = entropia(stati)
    
    # Lista per raccogliere i risultati dell'analisi
    risultati = []

    # Analizza ogni possibile confronto
    for idx in confronti:
        # Ottieni i gruppi e le metriche per questo confronto
        gruppi, dist, h_media, h_max = analizza(stati, arr, idx)
        
        # Information gain medio: quanto ci si aspetta di ridurre l'entropia
        ig_medio = h_corrente - h_media
        
        # Crea un record con tutte le informazioni
        record = {
            "idx":      idx,              # Indice del confronto
            "valore":   arr[idx],         # Valore dell'elemento
            "ig_medio": ig_medio,         # Guadagno informativo atteso
            "h_max":    h_max,            # Entropia worst-case
            "gruppi":   gruppi,           # Raggruppamento stati per esito
            "dist":     dist,             # Distribuzione numerica
        }
        risultati.append(record)

    # Ordina per IG medio decrescente (più alto = meglio)
    # In caso di parità, ordina per worst-case crescente
    per_media = sorted(
        risultati, 
        key=lambda r: (-r["ig_medio"], r["h_max"])
    )

    # Ordina per worst-case crescente (più basso = meglio)
    # In caso di parità, ordina per IG medio decrescente
    per_worst = sorted(
        risultati, 
        key=lambda r: (r["h_max"], -r["ig_medio"])
    )

    return per_media, per_worst


# ===========================================================================
# 4. FUNZIONI DI FORMATTAZIONE OUTPUT
# ===========================================================================

def formatta_array(arr: List[int]) -> str:
    """
    Formatta l'array per la visualizzazione con indici.

    Mostra ogni elemento con il suo indice nella forma:
    [indice]=valore

    Parameters
    ----------
    arr : List[int]
        Array da formattare.

    Returns
    -------
    str
        Stringa formattata con elementi e indici.

    Examples
    --------
    >>> formatta_array([10, 20, 30])
    '[0]=10  [1]=20  [2]=30'
    """
    # Crea una lista di stringhe "[i]=valore" per ogni elemento
    pezzi = [f"[{i}]={v}" for i, v in enumerate(arr)]
    # Unisce le stringhe con due spazi
    return "  ".join(pezzi)


def formatta_stati(stati: List[int]) -> str:
    """
    Formatta la lista degli stati candidati come stringa compatta.

    Parameters
    ----------
    stati : List[int]
        Lista di indici candidati.

    Returns
    -------
    str
        Stringa con gli indici separati da virgole.

    Examples
    --------
    >>> formatta_stati([0, 2, 5, 7])
    '0, 2, 5, 7'
    >>> formatta_stati([3])
    '3'
    """
    # Converte ogni indice in stringa e li unisce con ", "
    return ", ".join(map(str, stati))


def stampa_lista(titolo: str, lista: List[Dict]) -> None:
    """
    Stampa una lista di confronti suggeriti con le loro metriche.

    Mostra in formato tabulare: indice, valore, IG medio, worst-case,
    e la distribuzione degli stati risultanti.

    Parameters
    ----------
    titolo : str
        Intestazione della lista (es. "Top 3 (media)").
    lista : List[Dict]
        Lista di dizionari con le metriche dei confronti.

    Returns
    -------
    None
        Stampa direttamente su stdout.
    """
    # Stampa il titolo
    print(f"\n{titolo}:")
    # Riga di intestazione della tabella
    print(f"  {'Idx':<4} {'Val':<6} {'IG_medio':<10} {'Worst':<8} {'Distribuzione'}")
    print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*8} {'-'*30}")
    
    # Stampa ogni record
    for r in lista:
        # Formatta la distribuzione come stringa leggibile
        dist_str = "  ".join(f"{k}:{v}" for k, v in r["dist"].items())
        # Stampa la riga
        print(f"  {r['idx']:<4} {r['valore']:<6} "
              f"{r['ig_medio']:<10.3f} {r['h_max']:<8.3f} {dist_str}")


def stampa_stato(
    step: int,
    candidati: List[int],
    valore_cercato: int,
    entropia_attuale: float,
    ig: float,
    ig_cumulativo: float,
    ultimo_confronto: Optional[int],
    esito: Optional[str],
    modo: str,
    visualizzazione: Optional[str],
    arr: List[int]
) -> None:
    """
    Stampa lo stato corrente della ricerca con metriche informative.

    Mostra il numero di step, candidati rimanenti, entropia corrente,
    e informazione acquisita. Il livello di dettaglio dipende dalla
    modalità e visualizzazione scelta.

    Parameters
    ----------
    step : int
        Numero di confronti effettuati finora.
    candidati : List[int]
        Lista degli indici ancora candidati.
    valore_cercato : int
        Valore del bersaglio T.
    entropia_attuale : float
        Entropia corrente (incertezza residua).
    ig : float
        Information gain dell'ultimo confronto.
    ig_cumulativo : float
        Information gain totale dall'inizio.
    ultimo_confronto : Optional[int]
        Indice usato nell'ultimo confronto (None se step=0).
    esito : Optional[str]
        Esito dell'ultimo confronto (None se step=0).
    modo : str
        Modalità di esecuzione ('i' o 'a').
    visualizzazione : Optional[str]
        Livello di dettaglio ('b' o 'e'), usato solo in modalità interattiva.

    Returns
    -------
    None
        Stampa direttamente su stdout.
    """
    # Separatore visivo
    print("\n" + "="*70)
    # Intestazione dello step
    print(f"STEP {step}")
    print("="*70)

    # Mostra l'esito del confronto precedente (se esiste)
    if ultimo_confronto is not None and esito is not None:
        print(f"Ultimo confronto: arr[{ultimo_confronto}]={arr[ultimo_confronto]}  "
              f"→  Esito: {esito}")

    # Mostra sempre il numero di candidati e l'entropia
    print(f"Candidati rimanenti: {len(candidati)}  "
          f"(Entropia: {entropia_attuale:.3f} bit)")

    # Dettaglio esteso in modalità interattiva con visualizzazione estesa
    if modo == "i" and visualizzazione == "e":
        # Mostra la lista completa dei candidati
        print(f"Indici candidati: {formatta_stati(candidati)}")
        # Mostra le metriche di information gain
        if step > 0:
            print(f"IG ultimo step: {ig:.3f} bit  |  IG cumulativo: {ig_cumulativo:.3f} bit")


def parse_indice(s: str, n: int) -> Optional[int]:
    """
    Converte l'input utente in un indice valido.

    Verifica che la stringa sia un intero nell'intervallo [0, n-1].

    Parameters
    ----------
    s : str
        Stringa inserita dall'utente.
    n : int
        Dimensione dell'array (indice massimo = n-1).

    Returns
    -------
    Optional[int]
        Indice valido se l'input è corretto, None altrimenti.

    Examples
    --------
    >>> parse_indice("3", 10)
    3
    >>> parse_indice("15", 10)
    None                      # fuori range
    >>> parse_indice("abc", 10)
    None                      # non numerico
    """
    try:
        # Tenta di convertire la stringa in intero
        idx = int(s.strip())
        # Verifica che sia nell'intervallo valido
        if 0 <= idx < n:
            return idx
        else:
            return None  # Fuori range
    except ValueError:
        # La conversione è fallita: input non numerico
        return None


# ===========================================================================
# 5. FUNZIONE PRINCIPALE
# ===========================================================================

def risolvi() -> None:
    """
    Funzione principale: ricerca binaria ottimale guidata dalla teoria dell'informazione.

    **Workflow**:
    
    1. **Configurazione**: Chiede dimensione dell'array, genera array ordinato
       e sceglie casualmente la posizione del bersaglio.
    
    2. **Modalità di esecuzione**:
       - **Interattiva**: L'utente sceglie manualmente ogni confronto.
         Visualizzazione breve (b) o estesa (e) con suggerimenti.
       - **Automatica**: L'algoritmo sceglie automaticamente usando
         criterio medio (m) o worst-case (w).
    
    3. **Loop principale**: Ad ogni step:
       - Mostra lo stato corrente (candidati, entropia)
       - Valuta tutti i confronti possibili
       - Sceglie/chiede il confronto da eseguire
       - Simula l'esito e filtra i candidati
       - Aggiorna metriche (entropia, IG)
    
    4. **Terminazione**: Quando rimane un solo candidato, mostra il risultato
       e il confronto con l'ottimo teorico (⌈log₂ N⌉ per array ordinato).

    **Note sull'ottimalità**:
    Per array ordinati, la ricerca binaria (confronto con elemento centrale)
    è provabilmente ottimale. Questo script permette di verificare che:
    - Il confronto suggerito in modalità automatica è sempre quello centrale
    - Scegliere confronti diversi porta a performance sub-ottimali
    - Il numero di confronti nel caso peggiore è esattamente ⌈log₂ N⌉

    Returns
    -------
    None
        Funzione interattiva che stampa i risultati su stdout.

    Examples
    --------
    Esempio da terminale (array ordinato, modalità automatica, criterio worst)::

        $ python ricerca_array_ordinato.py
        Dimensione array (invio per default=16): 8
        Modalità (i=interattivo, a=automatico): a
        Criterio (m=media/w=worst): w
        
        Elemento cercato T = 55  (posizione ignota)
        
        ======================================================================
        STEP 0
        ======================================================================
        Candidati rimanenti: 8  (Entropia: 3.000 bit)
        ...
        
        Soluzione trovata:  5
        Soluzione reale:    5
        Confronti effettuati: 3    (ottimale teorico: ⌈log₂(8)⌉ = 3 per array ordinato)
    """
    # -----------------------------------------------------------------------
    # CONFIGURAZIONE INIZIALE
    # -----------------------------------------------------------------------

    # Chiede la dimensione dell'array; usa il default se l'utente preme Invio
    raw_n = input(f"Dimensione array (invio per default={DIM_DEFAULT}): ").strip()
    # Converte in intero o usa il default
    n = int(raw_n) if raw_n else DIM_DEFAULT

    # -----------------------------------------------------------------------
    # INIZIALIZZAZIONE DELL'ISTANZA
    # -----------------------------------------------------------------------

    # Genera l'array ordinato di n elementi distinti
    # NOTA: ordinato=True è hardcoded perché questo script tratta solo array ordinati
    arr = genera_array(n, ordinato=True)

    # Genera la posizione vera e nascosta del bersaglio T
    # Questa è la "soluzione" che l'algoritmo deve trovare
    indice_vero = genera_istanza(n)

    # Il valore del bersaglio è arr[indice_vero]
    # In uno scenario reale, l'utente conoscerebbe T ma non la sua posizione
    valore_cercato = arr[indice_vero]

    # Genera tutti gli N stati candidati iniziali (tutti gli indici possibili)
    candidati_iniziali = genera_stati(n)

    # Genera tutti i confronti disponibili (stessi indici)
    confronti = genera_confronti(n)

    # Calcola l'entropia iniziale: log₂(N) bit di incertezza
    entropia_iniziale = entropia(candidati_iniziali)

    # Mostra il valore da cercare (ma non la sua posizione)
    print(f"\nElemento cercato T = {valore_cercato}  (posizione ignota)")

    # -----------------------------------------------------------------------
    # CONFIGURAZIONE MODALITÀ DI ESECUZIONE
    # -----------------------------------------------------------------------

    # Chiede all'utente la modalità di esecuzione
    # 'i' = interattiva: l'utente sceglie ogni confronto
    # 'a' = automatica: l'algoritmo sceglie automaticamente
    modo = input("\nModalità (i=interattivo, a=automatico): ").strip()

    # Variabili di configurazione specifiche per modalità
    criterio      = None  # Usato solo in modalità automatica
    visualizzazione = None  # Usato solo in modalità interattiva

    if modo == "a":
        # In modalità automatica, chiede il criterio di selezione del confronto
        # 'm' = media: massimizza l'IG medio (guadagno informativo atteso)
        # 'w' = worst: minimizza l'entropia worst-case (migliore garanzia)
        criterio = input("Criterio (m=media/w=worst): ").strip()

    if modo == "i":
        # In modalità interattiva, chiede il livello di dettaglio
        # 'b' = breve: mostra solo le informazioni essenziali
        # 'e' = estesa: mostra candidati, metriche, suggerimenti
        visualizzazione = input("Visualizzazione (b=breve/e=estesa): ").strip()

    # -----------------------------------------------------------------------
    # VARIABILI DI STATO DEL LOOP
    # -----------------------------------------------------------------------

    step              = 0      # Contatore dei confronti effettuati
    ultimo_confronto  = None   # Indice usato nell'ultimo confronto
    esito_prec        = None   # Esito dell'ultimo confronto ("<", "=", ">")

    # Metriche di entropia e information gain
    entropia_attuale              = entropia_iniziale  # Entropia corrente
    informazione_acquisita        = 0.0  # IG dell'ultimo confronto
    informazione_acquisita_cumulativa = 0.0  # IG totale dall'inizio

    # -----------------------------------------------------------------------
    # LOOP PRINCIPALE DI RICERCA
    # -----------------------------------------------------------------------

    # Stati candidati correnti: inizia con tutti gli n indici
    candidati = candidati_iniziali
    
    # Continua finché non rimane esattamente un solo candidato
    while len(candidati) > 1:
        
        # -------------------------------------------------------------------
        # VISUALIZZAZIONE STATO CORRENTE
        # -------------------------------------------------------------------
        
        # Stampa il riepilogo dello stato prima di scegliere il prossimo confronto
        stampa_stato(
            step,                   # Numero di step corrente
            candidati,              # Candidati rimanenti
            valore_cercato,         # Valore del bersaglio
            entropia_attuale,       # Entropia corrente
            informazione_acquisita, # IG ultimo step
            informazione_acquisita_cumulativa,  # IG totale
            ultimo_confronto,       # Ultimo indice confrontato
            esito_prec,             # Ultimo esito
            modo,                   # Modalità esecuzione
            visualizzazione,        # Livello dettaglio
            arr                     # Array ordinato (per mostrare valori nei confronti)
        )

        # -------------------------------------------------------------------
        # VALUTAZIONE DI TUTTI I CONFRONTI POSSIBILI
        # -------------------------------------------------------------------
        
        # Valuta ogni confronto disponibile calcolando IG medio e worst-case
        # Restituisce due liste ordinate: per media (decrescente) e per worst (crescente)
        per_media, per_worst = valuta(candidati, confronti, arr)

        # -------------------------------------------------------------------
        # SELEZIONE DEL CONFRONTO
        # -------------------------------------------------------------------

        if modo == "i":
            # --- MODALITÀ INTERATTIVA ---
            # L'utente sceglie manualmente quale confronto eseguire

            if visualizzazione == "e":
                # Visualizzazione estesa: pausa e mostra suggerimenti
                _ = input("Premi il tasto Enter per continuare ")
                # Mostra i migliori confronti secondo IG medio
                stampa_lista(f"\nTop {TOP_K_MEDIA} (media)", per_media[:TOP_K_MEDIA])
                # Mostra i migliori confronti secondo worst-case
                stampa_lista(f"\nTop {TOP_M_WORST} (worst)", per_worst[:TOP_M_WORST])

            # Chiede all'utente quale indice confrontare con il bersaglio
            print(f"\n Indica l'indice da confrontare con il valore cercato (0–{n-1}):")
            idx_scelto = None

            # Ripete la richiesta finché l'utente non inserisce un indice valido
            while idx_scelto is None:
                idx_scelto = parse_indice(input("Indice: "), n)
                if idx_scelto is None:
                    # Input non valido: richiedi di nuovo
                    print(f"Inserisci un intero tra 0 e {n-1}.")

            # Cerca il record pre-calcolato per il confronto scelto
            r = None
            for cand in per_media:
                if cand["idx"] == idx_scelto:
                    r = cand  # Trovato
                    break

            if r is None:
                # Confronto non trovato tra quelli pre-valutati: lo calcola ora
                # (Può succedere se l'utente sceglie un indice inaspettato)
                gruppi, dist, h_media, h_max = analizza(stati, arr, idx_scelto)
                r = {
                    "idx":    idx_scelto,
                    "valore": arr[idx_scelto],
                    "gruppi": gruppi,
                }

        else:
            # --- MODALITÀ AUTOMATICA ---
            # L'algoritmo sceglie automaticamente il confronto migliore

            if criterio == "m":
                # Criterio medio: scegli il confronto con massimo IG medio
                r = per_media[0]
            else:
                # Criterio worst (w): scegli il confronto con minima entropia worst-case
                # Per array ordinato, questo coincide con la ricerca binaria
                r = per_worst[0]

        # -------------------------------------------------------------------
        # ESECUZIONE DEL CONFRONTO E AGGIORNAMENTO DEGLI STATI
        # -------------------------------------------------------------------

        # Simula il confronto usando la posizione vera (nascosta all'utente)
        # Confronta arr[indice_vero] con arr[r["idx"]]
        esito = confronta(indice_vero, arr, r["idx"])

        # Salva il confronto e l'esito per la visualizzazione al prossimo step
        ultimo_confronto = r["idx"]
        esito_prec       = esito

        # Filtra gli stati: conserva solo quelli compatibili con l'esito osservato
        # Il dizionario r["gruppi"] contiene gli stati raggruppati per esito
        candidati = r["gruppi"][esito]

        # Ricalcola l'entropia dopo il filtraggio
        entropia_ottenuta = entropia(candidati)

        # Calcola l'informazione acquisita con questo singolo confronto
        # IG = Entropia_prima - Entropia_dopo
        informazione_acquisita = entropia_attuale - entropia_ottenuta

        # Calcola l'informazione totale acquisita dall'inizio
        # IG_totale = Entropia_iniziale - Entropia_corrente
        informazione_acquisita_cumulativa = entropia_iniziale - entropia_ottenuta

        # Aggiorna l'entropia corrente per il prossimo step
        entropia_attuale = entropia_ottenuta

        # Incrementa il contatore dei confronti effettuati
        step += 1

    # -----------------------------------------------------------------------
    # SOLUZIONE TROVATA: rimane un solo candidato
    # -----------------------------------------------------------------------

    # Stampa il riepilogo dell'ultimo step con lo stato finale
    stampa_stato(
        step,                   # Numero finale di step
        candidati,              # Lista con un solo elemento
        valore_cercato,         # Valore cercato
        entropia_attuale,       # Dovrebbe essere 0.0
        informazione_acquisita, # IG ultimo step
        informazione_acquisita_cumulativa,  # IG totale
        ultimo_confronto,       # Ultimo confronto eseguito
        esito_prec,             # Ultimo esito
        modo,                   # Modalità esecuzione
        visualizzazione,        # Livello dettaglio
        arr
    )

    # Mostra il risultato della ricerca
    print(f"\nSoluzione trovata:  [{candidati[0]}]={arr[candidati[0]]}")
    print(f"Soluzione reale:    [{indice_vero}]={arr[indice_vero]}")
    
    # Confronta con l'ottimo teorico per array ordinati
    ottimo_teorico = math.ceil(math.log2(n))
    print(f"Confronti effettuati: {step}  "
          f"(ottimale teorico nel caso peggiore: ⌈log₂({n})⌉ = {ottimo_teorico} per array ordinato)")
    
    # Mostra il contenuto completo dell'array per riferimento
    print(f"\nContenuto array ordinato: {formatta_array(arr)}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Esegue la funzione principale quando lo script viene lanciato direttamente
    risolvi()

# %%