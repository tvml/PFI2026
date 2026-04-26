"""
Risolutore del problema delle 12 palline usando teoria dell'informazione.

Il problema: Ci sono 12 palline, di cui una ha peso diverso (più pesante o più leggera).
Obiettivo: Trovare quale pallina è diversa e se è più pesante o leggera usando una bilancia
a due piatti nel minor numero di pesate possibili.

Approccio: Usa l'entropia per selezionare la pesata ottimale che massimizza l'informazione
acquisita (information gain) ad ogni passo.
"""

import itertools
import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ============================================================================
# COSTANTI GLOBALI
# ============================================================================

# Numero di palline nel problema
NUM_PALLINE = 12

# Numero di top risultati da mostrare per ciascun criterio in modalità interattiva
TOP_K_MEDIA = 3  # Top pesate per entropia media
TOP_M_WORST = 3  # Top pesate per worst-case entropia

# Rappresentazione dei tipi di pallina
TIPO_PESANTE = +1  # Pallina più pesante del normale
TIPO_LEGGERA = -1  # Pallina più leggera del normale

# Esiti possibili di una pesata
ESITO_SINISTRA = "L"  # Il piatto sinistro pesa di più
ESITO_DESTRA = "R"    # Il piatto destro pesa di più
ESITO_PARI = "="      # I due piatti sono in equilibrio


# ============================================================================
# FUNZIONI DI GENERAZIONE
# ============================================================================

def genera_stati() -> List[Tuple[int, int]]:
    """
    Genera tutti i possibili stati iniziali del problema.
    
    Uno stato è una tupla (indice_pallina, tipo) dove:
    - indice_pallina: intero da 0 a 11 che identifica quale pallina è anomala
    - tipo: +1 se la pallina è più pesante, -1 se è più leggera
    
    Returns:
        Lista di 24 stati possibili (12 palline × 2 tipi)
    
    Esempio:
        [(0, -1), (0, +1), (1, -1), (1, +1), ..., (11, -1), (11, +1)]
    """
    # Genera tutte le combinazioni di indice pallina (0-11) e tipo (+1, -1)
    return [(i, t) for i in range(NUM_PALLINE) for t in (TIPO_LEGGERA, TIPO_PESANTE)]


def genera_istanza() -> Tuple[int, int]:
    """
    Genera un'istanza casuale del problema (la "soluzione vera").
    
    Returns:
        Tupla (indice_pallina, tipo) che rappresenta quale pallina è anomala
        e se è più pesante (+1) o più leggera (-1)
    
    Esempio:
        (7, -1) significa che la pallina 7 è più leggera
    """
    # Sceglie casualmente una pallina (0-11)
    indice = random.randint(0, NUM_PALLINE - 1)
    # Sceglie casualmente se è pesante o leggera
    tipo = random.choice([TIPO_LEGGERA, TIPO_PESANTE])
    return (indice, tipo)


def genera_pesate() -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Genera tutte le possibili pesate valide.
    
    Una pesata è una coppia (sx, dx) dove:
    - sx: tupla di indici delle palline sul piatto sinistro
    - dx: tupla di indici delle palline sul piatto destro
    
    Le pesate sono valide se:
    - Stesso numero di palline su entrambi i piatti (k palline per lato)
    - k varia da 1 a 4
    - Nessuna pallina appare su entrambi i piatti
    
    Returns:
        Lista di tuple ((sx), (dx)) con tutte le pesate possibili
    
    Esempio:
        [((0,), (1,)), ((0,), (2,)), ..., ((8,9,10,11), (0,1,2,3))]
    """
    pesate = []
    palline = list(range(NUM_PALLINE))
    
    # Itera su ogni possibile dimensione della pesata (1-4 palline per lato)
    for k in range(1, 5):
        # Genera tutte le combinazioni di k palline per il piatto sinistro
        for sx in itertools.combinations(palline, k):
            # Palline rimanenti (non sul piatto sinistro)
            rest = [p for p in palline if p not in sx]
            # Genera tutte le combinazioni di k palline per il piatto destro
            # scelte tra quelle rimanenti
            for dx in itertools.combinations(rest, k):
                # Aggiungi la pesata come coppia di tuple ordinate
                pesate.append((tuple(sorted(sx)), tuple(sorted(dx))))
    
    return pesate


# ============================================================================
# FUNZIONI DI CALCOLO E VALUTAZIONE
# ============================================================================

def entropia(stati: List[Tuple[int, int]]) -> float:
    """
    Calcola l'entropia di Shannon per un insieme di stati.
    
    L'entropia misura l'incertezza: quanta informazione ci manca per identificare
    lo stato vero. Assumendo equiprobabilità degli stati:
    H = log2(n) dove n è il numero di stati possibili
    
    Args:
        stati: Lista di stati ammissibili
    
    Returns:
        Entropia in bit. 0 se non ci sono stati, log2(len(stati)) altrimenti
    
    Esempio:
        entropia([(0,+1), (1,-1), (2,+1), (3,-1)]) = log2(4) = 2.0 bit
    """
    # Se non ci sono stati, l'entropia è 0 (nessuna incertezza)
    # Altrimenti, l'entropia è log2 del numero di stati
    return math.log2(len(stati)) if stati else 0


def pesa(stato: Tuple[int, int], sx: Tuple[int, ...], dx: Tuple[int, ...]) -> str:
    """
    Simula il risultato di una pesata dato uno stato specifico.
    
    Determina quale piatto peserà di più se lo stato dato fosse quello vero.
    
    Args:
        stato: Tupla (indice_pallina, tipo) che rappresenta lo stato da testare
        sx: Tuple di indici delle palline sul piatto sinistro
        dx: Tuple di indici delle palline sul piatto destro
    
    Returns:
        "L" se il piatto sinistro pesa di più
        "R" se il piatto destro pesa di più
        "=" se i piatti sono in equilibrio
    
    Logica:
        - Se la pallina anomala è sul piatto sx, quel piatto pesa tipo × 1
        - Se la pallina anomala è sul piatto dx, quel piatto pesa tipo × 1
        - Il tipo è +1 (pesante) o -1 (leggera)
    
    Esempio:
        stato = (3, +1)  # pallina 3 è pesante
        sx = (2, 3)      # palline 2 e 3 sul piatto sinistro
        dx = (0, 1)      # palline 0 e 1 sul piatto destro
        → ritorna "L" perché il piatto sinistro contiene la pallina pesante
    """
    idx, tipo = stato
    
    # Calcola il "peso extra" del piatto sinistro
    # Somma +tipo per ogni volta che la pallina anomala appare in sx
    sx_w = sum(tipo for p in sx if p == idx)
    
    # Calcola il "peso extra" del piatto destro
    # Somma +tipo per ogni volta che la pallina anomala appare in dx
    dx_w = sum(tipo for p in dx if p == idx)
    
    # Confronta i pesi
    if sx_w > dx_w:
        return ESITO_SINISTRA
    elif sx_w < dx_w:
        return ESITO_DESTRA
    return ESITO_PARI


def analizza(stati: List[Tuple[int, int]], 
             sx: Tuple[int, ...], 
             dx: Tuple[int, ...]) -> Tuple[Dict, Dict, float, float]:
    """
    Analizza come una pesata partiziona gli stati ammissibili.
    
    Per ogni possibile esito della pesata (L, R, =), calcola:
    - Quali stati sarebbero ancora ammissibili
    - Quanti stati rimangono
    - L'entropia residua per quel ramo
    
    Args:
        stati: Lista di stati attualmente ammissibili
        sx: Palline sul piatto sinistro
        dx: Palline sul piatto destro
    
    Returns:
        Tupla contenente:
        - gruppi: Dict[esito] -> lista di stati che producono quell'esito
        - dist: Dict[esito] -> {"n": num_stati, "h": entropia}
        - h_media: Entropia media pesata sui tre esiti
        - h_max: Entropia massima tra i tre esiti (worst case)
    
    Esempio:
        Se stati = [(0,+1), (1,+1), (2,+1), (3,+1)] e pesata sx=(0,1) dx=(2,3)
        → gruppi["L"] = [(0,+1), (1,+1)]  # stati che fanno pendere a sinistra
        → gruppi["R"] = [(2,+1), (3,+1)]  # stati che fanno pendere a destra
        → gruppi["="] = []                # nessuno stato produce equilibrio
    """
    # Raggruppa gli stati per esito
    gruppi = defaultdict(list)
    
    # Per ogni stato, simula la pesata e raggruppa per esito
    for s in stati:
        esito = pesa(s, sx, dx)
        gruppi[esito].append(s)
    
    totale = len(stati)
    dist = {}
    h_media = 0  # Entropia media (caso atteso)
    h_max = 0    # Entropia massima (worst case)
    
    # Calcola statistiche per ogni possibile esito
    for esito in [ESITO_SINISTRA, ESITO_PARI, ESITO_DESTRA]:
        g = gruppi[esito]
        n = len(g)  # Numero di stati in questo gruppo
        h = entropia(g) if n else 0  # Entropia del gruppo
        p = n / totale if totale else 0  # Probabilità di questo esito
        
        # Memorizza statistiche per questo esito
        dist[esito] = {"n": n, "h": h}
        
        # Accumula entropia media pesata per probabilità
        h_media += p * h
        
        # Aggiorna worst case
        h_max = max(h_max, h)
    
    return gruppi, dist, h_media, h_max


def valuta(stati: List[Tuple[int, int]], 
           pesate: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Valuta tutte le pesate possibili e le ordina per qualità.
    
    Per ogni pesata calcola:
    - Information Gain medio: H_iniziale - H_media
    - Information Gain minimo (worst case): H_iniziale - H_max
    
    Args:
        stati: Stati attualmente ammissibili
        pesate: Lista di tutte le pesate possibili
    
    Returns:
        Tupla contenente:
        - per_media: Pesate ordinate per IG medio (migliore caso atteso)
        - per_worst: Pesate ordinate per IG minimo (migliore worst case)
    
    Note:
        IG (Information Gain) alto = tanta informazione acquisita = buono
        H (entropia) bassa dopo la pesata = poca incertezza residua = buono
    """
    H = entropia(stati)  # Entropia iniziale
    risultati = []
    
    # Valuta ogni possibile pesata
    for sx, dx in pesate:
        # Analizza come questa pesata partiziona gli stati
        gruppi, dist, h_media, h_max = analizza(stati, sx, dx)
        
        # Memorizza tutte le metriche per questa pesata
        risultati.append({
            "sx": sx,
            "dx": dx,
            "h_media": h_media,           # Entropia media dopo la pesata
            "h_max": h_max,               # Entropia worst-case dopo la pesata
            "ig_medio": H - h_media,      # Information gain medio
            "ig_min": H - h_max,          # Information gain worst-case
            "dist": dist,                 # Distribuzione stati per esito
            "gruppi": gruppi              # Stati raggruppati per esito
        })
    
    # Ordina per information gain medio (caso atteso) - più alto è meglio
    per_media = sorted(risultati, key=lambda x: x["ig_medio"], reverse=True)
    
    # Ordina per entropia massima (worst case) - più bassa è meglio
    # (equivalente a ordinare per ig_min più alto)
    per_worst = sorted(risultati, key=lambda x: x["h_max"])
    
    return per_media, per_worst


# ============================================================================
# FUNZIONI DI FORMATTAZIONE OUTPUT
# ============================================================================

def formatta_stati(stati: List[Tuple[int, int]]) -> str:
    """
    Formatta una lista di stati in stringa leggibile.
    
    Args:
        stati: Lista di tuple (indice, tipo)
    
    Returns:
        Stringa formato "0+,1-,5+,..." dove:
        - il numero è l'indice della pallina
        - '+' indica pallina pesante, '-' indica pallina leggera
    
    Esempio:
        [(0, +1), (3, -1), (7, +1)] → "0+,3-,7+"
    """
    return ",".join(f"{i}{'+' if t == 1 else '-'}" for i, t in stati)


def formatta_tupla(t: Tuple[int, ...]) -> str:
    """
    Formatta una tupla di interi per visualizzazione.
    
    Args:
        t: Tupla di indici di palline
    
    Returns:
        Stringa formato "(1,2,3)" o "(x)" per singolo elemento
    
    Esempio:
        (1, 2, 3) → "(1,2,3)"
        (5,) → "(5)"
    """
    if len(t) == 1:
        return f"({t[0]})"
    return "(" + ",".join(str(x) for x in t) + ")"


def formatta_esito(e: str) -> str:
    """
    Formatta un esito di pesata in modo leggibile.
    
    Args:
        e: Carattere esito ('L', 'R', o '=')
    
    Returns:
        Stringa descrittiva dell'esito
        - "L" → "SX+" (piatto sinistro più pesante)
        - "R" → "DX+" (piatto destro più pesante)
        - "=" → "Pari" (equilibrio)
    """
    if e == ESITO_SINISTRA:
        return "SX+"
    elif e == ESITO_DESTRA:
        return "DX+"
    else:
        return "Pari"


# ============================================================================
# FUNZIONI DI I/O E INTERFACCIA UTENTE
# ============================================================================

def stampa_stato(step: int, 
                 stati: List[Tuple[int, int]], 
                 H0: float,
                 ultima_pesata: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]], 
                 esito_prec: Optional[str],
                 modo: str, 
                 visualizzazione: Optional[str]) -> None:
    """
    Stampa lo stato corrente del processo di risoluzione.
    
    Mostra:
    - Numero di pesate effettuate
    - Ultima pesata eseguita e suo esito
    - Stati ancora ammissibili
    - Entropia corrente e informazione acquisita (se modalità interattiva estesa)
    
    Args:
        step: Numero di pesate già effettuate
        stati: Stati attualmente ammissibili
        H0: Entropia iniziale (per calcolare informazione acquisita)
        ultima_pesata: Ultima pesata effettuata (sx, dx) o None
        esito_prec: Esito dell'ultima pesata o None
        modo: "i" per interattivo, "a" per automatico
        visualizzazione: "breve" o "estesa" (solo per modo interattivo)
    """
    print("\n==============================")
    print(f"Pesate effettuate: {step}")
    
    # Mostra ultima pesata se presente
    if ultima_pesata is not None:
        print(f"Ultima pesata: SX={formatta_tupla(ultima_pesata[0])} "
              f"DX={formatta_tupla(ultima_pesata[1])}")
        print(f"Esito: {formatta_esito(esito_prec)}")
    
    # Mostra stati ammissibili correnti
    print(f"Stati ammissibili: {formatta_stati(stati)}")
    
    # In modalità interattiva estesa, mostra anche metriche di entropia
    if modo == "i" and visualizzazione == 'estesa':
        H = entropia(stati)
        print(f"Entropia: {H:.4f}")
        print(f"Informazione totale acquisita: {H0 - H:.4f}")


def stampa_lista(titolo: str, lista: List[Dict]) -> None:
    """
    Stampa una lista di pesate con le loro metriche.
    
    Per ogni pesata mostra:
    - Quali palline vanno su sx e dx
    - Per ogni esito (L, =, R): numero di stati risultanti e loro entropia
    - Entropia media, entropia max, information gain medio e minimo
    
    Args:
        titolo: Titolo da stampare sopra la lista
        lista: Lista di dizionari con i risultati delle pesate
    """
    print(f"\n{titolo}")
    
    for r in lista:
        # Intestazione pesata
        print(f"\nSX={formatta_tupla(r['sx'])} DX={formatta_tupla(r['dx'])}")
        
        # Statistiche per ogni esito possibile
        s = ''
        for e in [ESITO_SINISTRA, ESITO_PARI, ESITO_DESTRA]:
            d = r["dist"][e]
            s += f"\t{formatta_esito(e)}: stati={d['n']} H={d['h']:.3f}"
        print(s)
        
        # Metriche aggregate
        s = (f"\tH_media={r['h_media']:.3f} (IG_medio={r['ig_medio']:.3f})"
             f"\tH_max={r['h_max']:.3f} (IG_min={r['ig_min']:.3f})")
        print(s)


def parse_input(s: str) -> Tuple[int, ...]:
    """
    Converte input utente in tupla di indici ordinata.
    
    Args:
        s: Stringa con numeri separati da spazi (es. "2 5 7")
    
    Returns:
        Tupla ordinata di interi. Tupla vuota se input vuoto.
    
    Esempio:
        "5 2 7" → (2, 5, 7)
        "" → ()
    """
    # Se l'input è vuoto o solo spazi, ritorna tupla vuota
    if not s.strip():
        return tuple()
    # Altrimenti, splitta per spazi, converte in int e ordina
    return tuple(sorted(int(x) for x in s.split()))


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def risolvi() -> None:
    """
    Funzione principale che esegue il processo di risoluzione.
    
    Flusso:
    1. Genera tutti gli stati possibili (24 stati)
    2. Genera l'istanza "vera" casuale (soluzione da trovare)
    3. Genera tutte le pesate possibili
    4. Chiede modalità di esecuzione (interattiva o automatica)
    5. Loop principale:
       - Valuta tutte le pesate ammissibili
       - Seleziona la pesata (automaticamente o tramite input utente)
       - Simula il risultato della pesata
       - Filtra gli stati ammissibili
       - Ripete fino a trovare la soluzione
    6. Mostra la soluzione trovata
    
    Modalità:
    - Interattiva: L'utente sceglie le pesate vedendo i suggerimenti
      - Breve: mostra solo prompt per input
      - Estesa: mostra top pesate e metriche dettagliate
    - Automatica: Il programma sceglie automaticamente usando un criterio
      - Media: sceglie pesata con miglior information gain medio
      - Worst: sceglie pesata con miglior worst-case (entropia max minima)
    """
    # ========== INIZIALIZZAZIONE ==========
    
    # Genera tutti i 24 stati possibili (12 palline × 2 tipi)
    stati = genera_stati()
    
    # Genera la soluzione "vera" casuale
    vero = genera_istanza()
    
    # (Opzionale) Decommentare per vedere la soluzione durante il debug
    # print("Istanza reale (debug):", vero)
    
    # Genera tutte le possibili pesate
    pesate = genera_pesate()
    
    # Calcola entropia iniziale (log2(24) = ~4.58 bit)
    H0 = entropia(stati)
    
    # ========== CONFIGURAZIONE MODALITÀ ==========
    
    modo = input("Modalità (i=interattivo, a=automatico): ").strip()
    
    criterio = None
    visualizzazione = None
    
    if modo == "a":
        # In modalità automatica, chiedi il criterio di scelta
        criterio = input("Criterio (media/worst): ").strip()
    
    if modo == "i":
        # In modalità interattiva, chiedi livello di dettaglio
        visualizzazione = input("Visualizzazione (breve/estesa): ").strip()
        print(modo, visualizzazione)
    
    # ========== VARIABILI DI STATO ==========
    
    step = 0  # Contatore pesate effettuate
    esito_prec = None  # Esito dell'ultima pesata
    ultima_pesata = None  # Ultima pesata effettuata
    
    # ========== LOOP PRINCIPALE ==========
    
    # Continua fino a quando rimane un solo stato ammissibile
    while len(stati) > 1:
        # Stampa stato corrente
        stampa_stato(step, stati, H0, ultima_pesata, esito_prec, modo, visualizzazione)
        
        # Valuta tutte le pesate possibili con gli stati correnti
        per_media, per_worst = valuta(stati, pesate)
        
        # ========== SELEZIONE PESATA ==========
        
        if modo == "i":
            # ===== MODALITÀ INTERATTIVA =====
            
            if visualizzazione == 'estesa':
                # Mostra top pesate per information gain medio
                stampa_lista(f"\nTop {TOP_K_MEDIA} (media)", per_media[:TOP_K_MEDIA])
                # Mostra top pesate per worst case
                stampa_lista(f"\nTop {TOP_M_WORST} (worst)", per_worst[:TOP_M_WORST])
            
            # Chiedi all'utente quale pesata effettuare
            print('\n Indica la pesata da effettuare\n')
            sx = parse_input(input("SX: "))
            dx = parse_input(input("DX: "))
            
            # Cerca la pesata scelta tra quelle valutate
            for cand in per_media:
                if cand["sx"] == sx and cand["dx"] == dx:
                    r = cand
                    break
            else:
                # Se la pesata non è tra quelle pre-valutate, calcolala al volo
                gruppi, dist, h_media, h_max = analizza(stati, sx, dx)
                r = {"sx": sx, "dx": dx, "gruppi": gruppi}
        
        else:
            # ===== MODALITÀ AUTOMATICA =====
            
            # Scegli automaticamente la migliore pesata secondo il criterio
            if criterio == "media":
                # Usa la pesata con miglior information gain medio
                r = per_media[0]
            else:  # criterio == "worst"
                # Usa la pesata con miglior worst case
                r = per_worst[0]
        
        # ========== ESECUZIONE PESATA ==========
        
        # Simula la pesata con lo stato vero
        esito = pesa(vero, r["sx"], r["dx"])
        
        # Salva pesata corrente per visualizzazione prossimo step
        ultima_pesata = (r["sx"], r["dx"])
        esito_prec = esito
        
        # Filtra gli stati: mantieni solo quelli compatibili con l'esito osservato
        stati = r["gruppi"][esito]
        
        # Incrementa contatore pesate
        step += 1
    
    # ========== SOLUZIONE TROVATA ==========
    
    # Mostra stato finale
    stampa_stato(step, stati, H0, ultima_pesata, esito_prec, modo, visualizzazione)
    
    # Mostra soluzione trovata
    print("\nSoluzione trovata:", formatta_stati(stati))
    # Mostra soluzione reale (per verifica)
    print("\nSoluzione reale:", formatta_stati([vero]))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    risolvi()
