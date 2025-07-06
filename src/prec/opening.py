from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_opening(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1: "Opening 1C",
        2: "Opening 1D",
        3: "Opening 1H",
        4: "Opening 1S",
        5: "Opening 1NT",
        6: "Opening 2C",
        7: "Opening 2D",
        8: "Opening 2H",
        9: "Opening 2S",
        10: "Opening 2NT",
        11: "Opening 3C",
        12: "Opening 3D",
        13: "Opening 3H",
        14: "Opening 3S",
        0: "Pass"
    }

    # Implementasi logika bidding opening
    if 15 <= total_hcp <= 17 and analyzer.is_balanced_distribution(dist) and not analyzer.has_five_card_major(suit_counts):
        possibleBids.append(5)
    if total_hcp >= 16:
        if 22 <= total_hcp <= 23 and analyzer.is_balanced_distribution(dist):
            possibleBids.append(7)
        else:
            possibleBids.append(1)
    if 11 <= total_hcp <= 15:
        if suit_counts["S"] >= 5:
            possibleBids.append(4)
        elif suit_counts["H"] >= 5:
            possibleBids.append(3)
        elif suit_counts["C"] >= 6 or (suit_counts["C"] == 5 and (suit_counts["S"] == 4 or suit_counts["H"] == 4)):
            possibleBids.append(6)
        else:
            possibleBids.append(2)
    if 6 <= total_hcp <= 10:
        if suit_counts["S"] == 6 or suit_counts["H"] == 6:
            possibleBids.append(7)
        elif suit_counts["H"] >= 5 and (suit_counts["D"] >= 5 or suit_counts["C"] >= 5):
            possibleBids.append(8)
        elif suit_counts["S"] >= 5 and (suit_counts["H"] >= 5 or suit_counts["D"] >= 5 or suit_counts["C"] >= 5):
            possibleBids.append(9)
        elif suit_counts["D"] >= 5 and suit_counts["C"] >= 5:
            possibleBids.append(10)
        elif suit_counts["C"] >= 7:
            possibleBids.append(11)
        elif suit_counts["D"] >= 7:
            possibleBids.append(12)
        elif suit_counts["H"] >= 7:
            possibleBids.append(13)
        elif suit_counts["S"] >= 7:
            possibleBids.append(14)

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]

    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }