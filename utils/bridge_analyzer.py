# bid_snapper_backend/utils/bridge_analyzer.py

from collections import Counter

class BridgeHandAnalyzer:
    def __init__(self):
        self.hcp_values = {"A": 4, "K": 3, "Q": 2, "J": 1}

        self.balance_patterns = [
            [4, 4, 3, 2],
            [4, 3, 3, 3],
            [5, 3, 3, 2]
        ]

    def is_balanced_distribution(self, dist):
        dist_counter = Counter(sorted(dist, reverse=True))
        for pattern in self.balance_patterns:
            if Counter(pattern) == dist_counter:
                return True
        return False

    def calculate_hcp_and_distribution(self, hand):
        total_hcp = 0
        suit_counts = {"S": 0, "H": 0, "D": 0, "C": 0}

        for card in hand:
            if len(card) == 2:
                rank, suit = card[0], card[1]
            else:
                rank, suit = card[:2], card[2]

            hcp = self.hcp_values.get(rank, 0)
            total_hcp += hcp

            if suit in suit_counts:
                suit_counts[suit] += 1

        shdc = [suit_counts["S"], suit_counts["H"], suit_counts["D"], suit_counts["C"]]
        dist = sorted(shdc, reverse=True)

        return {
            'total_hcp': total_hcp,
            'suit_counts': suit_counts,
            'dist': dist,
            'shdc': shdc,
            'distribusi': ''.join(map(str, shdc))
        }

    def has_five_card_major(self, suit_counts):
        return suit_counts["S"] >= 5 or suit_counts["H"] >= 5