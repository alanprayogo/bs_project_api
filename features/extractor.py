import sys
import pandas as pd
import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class BridgeHandAnalyzer:
    """
    A comprehensive bridge hand analyzer for calculating HCP, distributions,
    honor power, and other bridge-related metrics.
    """
    
    def __init__(self):
        self.hcp_map = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
        self.suits = ['S', 'H', 'D', 'C']
        self.honor_weights = {
            frozenset(['A', 'K', 'Q']): 3.0,  # AKQ
            frozenset(['A', 'K']): 2.0,       # AK
            frozenset(['A', 'Q']): 1.5,       # AQ  
            frozenset(['A']): 1.0,            # A
            frozenset(['K', 'Q']): 1.0,       # KQ
            frozenset(['K']): 0.5,            # K alone
            frozenset(['Q']): 0.25,           # Q alone
            frozenset([]): 0.0                # No honors
        }
    
    def calculate_hcp(self, hand: List[str]) -> int:
        """Calculate High Card Points (HCP) from a hand."""
        return sum(self.hcp_map.get(card[0], 0) for card in hand)
    
    def count_suit_length(self, hand: List[str], suit: str) -> int:
        """Count number of cards in a specific suit."""
        return sum(1 for card in hand if card.endswith(suit))
    
    def get_distribution(self, hand: List[str]) -> List[int]:
        """Return distribution of cards per suit: [S, H, D, C]."""
        return [self.count_suit_length(hand, suit) for suit in self.suits]
    
    def classify_distribution(self, distribution: List[int]) -> Tuple[str, float]:
        """
        Classify hand distribution with both category and numeric balance score.
        Returns: (category_name, balance_score)
        """
        sorted_dist = sorted(distribution, reverse=True)
        
        if sorted_dist == [4, 3, 3, 3]:
            return "Balanced (4333)", 0.0
        elif sorted_dist == [4, 4, 3, 2]:
            return "Semi-Balanced (4432)", 0.25
        elif sorted_dist == [5, 3, 3, 2]:
            return "Semi-Balanced (5332)", 0.25
        elif sorted_dist == [5, 4, 2, 2]:
            return "Semi-Balanced (5422)", 0.5
        elif sorted_dist[0] == 6:
            if sorted_dist == [6, 3, 2, 2]:
                return "Unbalanced (6322)", 0.75
            else:
                return "Unbalanced (6+ suit)", 0.75
        elif sorted_dist[0] >= 7:
            return "Very Unbalanced (7+ suit)", 1.0
        elif 1 in sorted_dist:
            return "Unbalanced (singleton)", 0.8
        elif 0 in sorted_dist:
            return "Unbalanced (void)", 1.0
        else:
            return "Unbalanced (other)", 0.6
    
    def calculate_honor_weight_per_suit(self, hand: List[str], suit: str) -> float:
        """Calculate honor weight for a specific suit."""
        suit_cards = [card for card in hand if card.endswith(suit)]
        honors = frozenset(card[0] for card in suit_cards if card[0] in ['A', 'K', 'Q', 'J'])
        
        for honor_combo, weight in self.honor_weights.items():
            if honor_combo.issubset(honors):
                return weight
        
        return 0.0
    
    def calculate_partnership_honor_power(self, hand1: List[str], hand2: List[str]) -> Dict[str, float]:
        """Calculate comprehensive honor power metrics for partnership."""
        combined_hand = hand1 + hand2
        
        suit_honors = {}
        total_honor_power = 0
        
        for suit in self.suits:
            h1_weight = self.calculate_honor_weight_per_suit(hand1, suit)
            h2_weight = self.calculate_honor_weight_per_suit(hand2, suit)
            combined_weight = h1_weight + h2_weight
            
            suit_honors[f'honor_{suit.lower()}'] = combined_weight
            total_honor_power += combined_weight
        
        suit_honors['total_honor_power'] = total_honor_power
        return suit_honors
    
    def calculate_controls(self, hand: List[str]) -> Tuple[int, int]:
        """
        Calculate controls (Aces and Kings).
        Returns: (total_controls, aces_count)
        """
        aces = sum(1 for card in hand if card[0] == 'A')
        kings = sum(1 for card in hand if card[0] == 'K')
        total_controls = aces * 2 + kings
        return total_controls, aces
    
    def calculate_quick_tricks(self, hand: List[str]) -> float:
        """Calculate quick tricks for each suit and total."""
        quick_tricks = 0
        
        for suit in self.suits:
            suit_cards = [card for card in hand if card.endswith(suit)]
            honors = [card[0] for card in suit_cards if card[0] in ['A', 'K', 'Q']]
            
            if 'A' in honors and 'K' in honors:
                quick_tricks += 2.0
            elif 'A' in honors:
                quick_tricks += 1.0
            elif 'K' in honors and 'Q' in honors:
                quick_tricks += 1.0
            elif 'K' in honors:
                quick_tricks += 0.5
        
        return quick_tricks
    
    def extract_comprehensive_features(self, hand1: List[str], hand2: List[str]) -> Dict:
        """Extract comprehensive features from partnership hands."""
        combined_hand = hand1 + hand2
        
        total_hcp = self.calculate_hcp(combined_hand)
        hcp_hand1 = self.calculate_hcp(hand1)
        hcp_hand2 = self.calculate_hcp(hand2)
        
        dist_combined = self.get_distribution(combined_hand)
        dist_hand1 = self.get_distribution(hand1)
        dist_hand2 = self.get_distribution(hand2)
        
        balance_cat1, balance_score1 = self.classify_distribution(dist_hand1)
        balance_cat2, balance_score2 = self.classify_distribution(dist_hand2)
        
        honor_metrics = self.calculate_partnership_honor_power(hand1, hand2)
        
        controls1, aces1 = self.calculate_controls(hand1)
        controls2, aces2 = self.calculate_controls(hand2)
        qt1 = self.calculate_quick_tricks(hand1)
        qt2 = self.calculate_quick_tricks(hand2)
        
        hcp_difference = abs(hcp_hand1 - hcp_hand2)
        longest_suit = max(dist_combined)
        shortest_suit = min(dist_combined)
        suit_range = longest_suit - shortest_suit
        
        suits_8plus = sum(1 for length in dist_combined if length >= 8)
        suits_9plus = sum(1 for length in dist_combined if length >= 9)
        
        features = {
            'total_hcp': total_hcp,
            'hcp_hand1': hcp_hand1,
            'hcp_hand2': hcp_hand2,
            'hcp_difference': hcp_difference,
            'dist_spades': dist_combined[0],
            'dist_hearts': dist_combined[1], 
            'dist_diamonds': dist_combined[2],
            'dist_clubs': dist_combined[3],
            'longest_suit': longest_suit,
            'shortest_suit': shortest_suit,
            'suit_range': suit_range,
            'suits_8plus': suits_8plus,
            'suits_9plus': suits_9plus,
            'balance_score1': balance_score1,
            'balance_score2': balance_score2,
            'avg_balance_score': (balance_score1 + balance_score2) / 2,
            'total_controls': controls1 + controls2,
            'total_aces': aces1 + aces2,
            'total_quick_tricks': qt1 + qt2,
            **honor_metrics
        }
        
        return features