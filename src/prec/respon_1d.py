from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_1d(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 1H",
        2:"Bid 1S",
        3:"Bid 1NT",
        4:"Bid 2C",
        5:"Bid 2D",
        6:"Bid 2NT",
        7:"Bid 3NT",
        8:"Bid 4C",
        0:"Pass"
    }

    # Determine the contract based on HCP and suit counts
    # 1H
    if suit_counts["H"] >= 4:
        possibleBids.append(1)
    if suit_counts["H"] >=4 and suit_counts["S"] >= 4:
        possibleBids.append(1)
    # 1S
    if suit_counts["S"] >= 4:
        possibleBids.append(2)
    # 2C, 2D
    if total_hcp >= 12 and (suit_counts["H"] < 4 or suit_counts["S"] < 4):
        if suit_counts["C"] >= 5:
            possibleBids.append(4)
        if suit_counts["D"] >= 5:
            possibleBids.append(5)
    # Pass
    if 0 <= total_hcp <=5 and (suit_counts["H"] < 4 or suit_counts["S"] < 4):
        possibleBids.append(0)
    # 1NT, 2NT, 3NT
    if not possibleBids:
        if 6 <= total_hcp <= 11:
            possibleBids.append(3)
        if 12 <= total_hcp <= 13:
            possibleBids.append(6)
        if 14 <= total_hcp <= 15:
            possibleBids.append(7)

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]
    
    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }