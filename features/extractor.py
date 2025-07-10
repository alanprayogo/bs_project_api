# features/extractor.py

import sys
import pandas as pd

def calculate_hcp(hand):
    """
    Hitung High Card Points (HCP) dari sebuah tangan.
    A = 4, K = 3, Q = 2, J = 1
    """
    hcp_map = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum(hcp_map.get(card[0], 0) for card in hand)

def count_suit_length(hand, suit):
    """
    Hitung jumlah kartu dalam suit tertentu di tangan.
    Contoh suit: 'S', 'H', 'D', 'C'
    """
    return sum(1 for card in hand if card.endswith(suit))

def get_distribution(hand):
    """
    Mengembalikan distribusi jumlah kartu per suit: [S, H, D, C]
    """
    spades = count_suit_length(hand, 'S')
    hearts = count_suit_length(hand, 'H')
    diamonds = count_suit_length(hand, 'D')
    clubs = count_suit_length(hand, 'C')
    return [spades, hearts, diamonds, clubs]

def is_balanced(distribution):
    """
    Tentukan apakah distribusi seimbang berdasarkan aturan bridge:
    - 4333, 4432, 5332 → Balanced
    - Lainnya → Tidak seimbang
    Output: 0=Balanced, 1=Likely Balanced, 2=Likely Unbalanced, 3=Unbalanced
    """
    sorted_dist = sorted(distribution, reverse=True)
    if sorted_dist == [4, 3, 3, 3]:
        return 0
    elif sorted_dist == [4, 4, 3, 2] or sorted_dist == [5, 3, 3, 2]:
        return 1
    elif sorted_dist[0] >= 6:
        return 3
    elif sorted_dist[0] == 5 and sorted_dist[1] <= 3:
        return 2
    else:
        return 3

def estimate_losing_tricks(hand):
    """
    Estimasi Losing Trick Count (LTC) sederhana berdasarkan top 3 kartu per suit.
    """
    ltc = 0
    suits = ['S', 'H', 'D', 'C']
    for suit in suits:
        cards_in_suit = sorted([c[0] for c in hand if c.endswith(suit)], key=lambda x: 'AKQJT98765432'.index(x))
        length = len(cards_in_suit)
        if length == 0:
            continue
        elif length == 1:
            # Single card
            ltc += 2 if cards_in_suit[0] != 'A' else 1
        elif length == 2:
            # Two cards
            top_two = cards_in_suit[:2]
            if 'A' in top_two and 'K' in top_two:
                ltc += 0
            elif 'A' in top_two or 'K' in top_two:
                ltc += 1
            else:
                ltc += 2
        else:
            # Three or more cards
            top_three = cards_in_suit[:3]
            missing = 0
            if 'A' not in top_three:
                missing += 1
            if 'K' not in top_three:
                missing += 1
            if 'Q' not in top_three:
                missing += 1
            ltc += min(missing, 3)
    return round(ltc, 2)

def has_stopper(hand, suit):
    cards_in_suit = [c[0] for c in hand if c.endswith(suit)]
    if len(cards_in_suit) == 0:
        return 0  # void, no stopper
    elif 'A' in cards_in_suit:
        return 3
    elif 'K' in cards_in_suit:
        return 2 if len(cards_in_suit) > 1 else 1
    elif 'Q' in cards_in_suit:
        return 2 if len(cards_in_suit) > 2 else 1
    else:
        return 0

def extract_features_from_hand(hand1, hand2, as_dataframe=True):
    """
    Ekstrak fitur dari pasangan tangan.
    
    Parameters:
    - hand1 (list): list dari 13 kartu pemain 1
    - hand2 (list): list dari 13 kartu pemain 2
    - as_dataframe (bool): jika True, kembalikan pd.DataFrame
    
    Returns:
    - dict or pd.DataFrame
    """
    combined_hand = hand1 + hand2

    # Distribusi suit
    dist = get_distribution(combined_hand)
    spades, hearts, diamonds, clubs = dist

    # HCP per suit
    hcp_spades = calculate_hcp([c for c in combined_hand if c.endswith('S')])
    hcp_hearts = calculate_hcp([c for c in combined_hand if c.endswith('H')])
    hcp_diamonds = calculate_hcp([c for c in combined_hand if c.endswith('D')])
    hcp_clubs = calculate_hcp([c for c in combined_hand if c.endswith('C')])

    # Balanced distribution
    balanced_hand1 = is_balanced(get_distribution(hand1))
    balanced_hand2 = is_balanced(get_distribution(hand2))

    # Stopper
    stopper_spades = has_stopper(combined_hand, 'S')
    stopper_hearts = has_stopper(combined_hand, 'H')
    stopper_diamonds = has_stopper(combined_hand, 'D')
    stopper_clubs = has_stopper(combined_hand, 'C')

    # LTC
    ltc = estimate_losing_tricks(combined_hand)

    # Total HCP
    total_hcp = hcp_spades + hcp_hearts + hcp_diamonds + hcp_clubs

    # Rentang jumlah kartu per suit
    def get_cardinality_range(count):
        lower = max(0, count - 1)
        upper = min(13, count + 1)
        return lower, upper

    num_spades_low, num_spades_high = get_cardinality_range(spades)
    num_hearts_low, num_hearts_high = get_cardinality_range(hearts)
    num_diamonds_low, num_diamonds_high = get_cardinality_range(diamonds)
    num_clubs_low, num_clubs_high = get_cardinality_range(clubs)

    # Dictionary fitur
    features = {
        "hcp": total_hcp,
        "hcp_spades": hcp_spades,
        "hcp_hearts": hcp_hearts,
        "hcp_diamonds": hcp_diamonds,
        "hcp_clubs": hcp_clubs,

        "dist_spades": spades,
        "dist_hearts": hearts,
        "dist_diamonds": diamonds,
        "dist_clubs": clubs,

        "balanced_hand1": balanced_hand1,
        "balanced_hand2": balanced_hand2,

        "stopper_spades": stopper_spades,
        "stopper_hearts": stopper_hearts,
        "stopper_diamonds": stopper_diamonds,
        "stopper_clubs": stopper_clubs,

        "ltc": ltc,

        "num_spades_low": num_spades_low,
        "num_spades_high": num_spades_high,
        "num_hearts_low": num_hearts_low,
        "num_hearts_high": num_hearts_high,
        "num_diamonds_low": num_diamonds_low,
        "num_diamonds_high": num_diamonds_high,
        "num_clubs_low": num_clubs_low,
        "num_clubs_high": num_clubs_high,
    }

    # Return type
    if as_dataframe:
        return pd.DataFrame([features])
    else:
        return features