# features/extractor.py

import sys
import pandas as pd

def calculate_hcp(hand):
    # Hitung High Card Points (HCP) dari sebuah tangan.
    # A = 4, K = 3, Q = 2, J = 1
    hcp_map = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum(hcp_map.get(card[0], 0) for card in hand)

def count_suit_length(hand, suit):
    # Hitung jumlah kartu dalam suit tertentu di tangan.
    # Contoh suit: 'S', 'H', 'D', 'C'
    return sum(1 for card in hand if card.endswith(suit))

def get_distribution(hand):
    # Mengembalikan distribusi jumlah kartu per suit: [S, H, D, C]
    spades = count_suit_length(hand, 'S')
    hearts = count_suit_length(hand, 'H')
    diamonds = count_suit_length(hand, 'D')
    clubs = count_suit_length(hand, 'C')
    return [spades, hearts, diamonds, clubs]

def is_balanced(distribution):
    # Tentukan apakah distribusi seimbang berdasarkan aturan bridge:
    # 4333, 4432, 5332 → Balanced
    # Lainnya → Tidak seimbang
    # Output: 0=Balanced, 1=Likely Balanced, 2=Likely Unbalanced, 3=Unbalanced
    sorted_dist = sorted(distribution, reverse=True)
    if sorted_dist == [4, 3, 3, 3]:
        return 0
    elif sorted_dist == [4, 4, 3, 2] or sorted_dist == [5, 3, 3, 2]:
        return 1
    elif sorted_dist == [5, 4, 2, 2]:
        return 2
    elif sorted_dist[0] >= 6:
        return 3
    else:
        return 3
    
def calculate_honor_weight_per_suit(hand, suit):
    # Hitung bobot honor untuk suit tertentu dalam satu tangan.
    # Bobot honor:
    # - AKQ = 3.0
    # - AK = 2.0
    # - AQ = 1.5
    # - A = 1.0
    # - KQ = 1.0
    # - Kx = 0.5
    # - Qxx = 0.25

    # Ambil kartu dari suit tertentu
    suit_cards = [card for card in hand if card.endswith(suit)]
    
    # Ekstrak honor cards (A, K, Q, J)
    honors = []
    for card in suit_cards:
        if card[0] in ['A', 'K', 'Q', 'J']:
            honors.append(card[0])
    
    # Hitung bobot berdasarkan kombinasi honor
    if 'A' in honors and 'K' in honors and 'Q' in honors:
        return 3.0  # AKQ
    elif 'A' in honors and 'K' in honors:
        return 2.0  # AK
    elif 'A' in honors and 'Q' in honors:
        return 1.5  # AQ
    elif 'A' in honors:
        return 1.0  # A
    elif 'K' in honors and 'Q' in honors:
        return 1.0  # KQ
    elif 'K' in honors:
        return 0.5  # Kx
    elif 'Q' in honors:
        return 0.25  # Qxx
    else:
        return 0.0  # Tidak ada honor
    
def calculate_honor_weight_all_suits(hand):
    # Hitung bobot honor untuk semua suit dalam satu tangan.
    honor_s = calculate_honor_weight_per_suit(hand, 'S')
    honor_h = calculate_honor_weight_per_suit(hand, 'H')
    honor_d = calculate_honor_weight_per_suit(hand, 'D')
    honor_c = calculate_honor_weight_per_suit(hand, 'C')
    
    return honor_s, honor_h, honor_d, honor_c

def calculate_partnership_honor_weight_per_suit(hand1, hand2):
    # Hitung bobot honor per suit untuk partnership (2 tangan).
    # Hitung honor untuk hand1
    honor_s1, honor_h1, honor_d1, honor_c1 = calculate_honor_weight_all_suits(hand1)
    # Hitung honor untuk hand2  
    honor_s2, honor_h2, honor_d2, honor_c2 = calculate_honor_weight_all_suits(hand2)
    
    # Jumlahkan per suit
    sum_honor_s = honor_s1 + honor_s2
    sum_honor_h = honor_h1 + honor_h2
    sum_honor_d = honor_d1 + honor_d2
    sum_honor_c = honor_c1 + honor_c2
    
    # Total honor power
    honor_power = sum_honor_s + sum_honor_h + sum_honor_d + sum_honor_c
    
    return sum_honor_s, sum_honor_h, sum_honor_d, sum_honor_c, honor_power

def extract_features_from_hand(hand1, hand2, as_dataframe=True):
    # Ekstrak fitur dari pasangan tangan.

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

    # Total HCP
    total_hcp = hcp_spades + hcp_hearts + hcp_diamonds + hcp_clubs

    # Honor weight calculations
    sum_honor_s, sum_honor_h, sum_honor_d, sum_honor_c, honor_power = calculate_partnership_honor_weight_per_suit(hand1, hand2)

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

        "sum_honor_s": sum_honor_s,
        "sum_honor_h": sum_honor_h,
        "sum_honor_d": sum_honor_d,
        "sum_honor_c": sum_honor_c,
        "honor_power": honor_power,

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
    
# hand1 = ['AS', 'KS', 'QS', 'JS', 'TS', '9H', '8H', '7H', '6D', '5D', '4D', '3C', '2C']
# hand2 = ['AH', 'KH', 'QH', 'JH', 'TH', '9S', '8S', '7S', '6C', '5C', '4C', '3D', '2D']
# features = extract_features_from_hand(hand1, hand2, as_dataframe=True)
# print(features)