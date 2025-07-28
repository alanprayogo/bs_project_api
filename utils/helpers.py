def parse_contract(contract):
    """Parse contract string into suit and level."""
    level = int(contract[0])
    suit_char = contract[1] if len(contract) > 1 else 'N'
    suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'N': 4}
    return suit_map[suit_char], level

def map_level_to_category(level, suit):
    """Map contract level and suit to category."""
    if level == 7:
        return 3  # Grand slam
    elif level == 6:
        return 2  # Slam
    elif (suit in [0, 1] and level >= 4) or (suit == 4 and level >= 3) or (suit in [2, 3] and level >= 5):
        return 1  # Game
    else:
        return 0  # Partial game

def estimate_score_corrected(suit, level):
    """
    Scoring bridge yang benar:
    - Minor suits (♣♦): N × 20
    - Major suits (♠♥): N × 30  
    - NT: N × 30 + 10
    - Bonus part score: +50
    - Bonus game: +300
    - Bonus slam: +500
    - Bonus grand slam: +1000
    """
    if suit in [2, 3]:  # Minor suits (♦♣)
        trick_score = level * 20
    elif suit in [0, 1]:  # Major suits (♠♥)
        trick_score = level * 30
    else:  # NT
        trick_score = level * 30 + 10
    
    bonus = 0
    category = map_level_to_category(level, suit)
    if category == 0:  # Part score
        bonus = 50
    elif category == 1:  # Game
        bonus = 300
    elif category == 2:  # Small slam
        bonus = 500
    elif category == 3:  # Grand slam
        bonus = 1000
    
    return trick_score + bonus