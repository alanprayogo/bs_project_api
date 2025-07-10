from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_1h(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 1S",
        2:"Bid 1NT",
        3:"Bid 2C",
        4:"Bid 2D",
        5:"Bid 2H",
        6:"Bid 2NT",
        7:"Bid 3H",
        8:"Bid 3NT",
        9:"Bid 4H",
        10:"Bid 4NT",
        0:"Pass"
    }

    # Determine the contract based on HCP and suit counts
    # 1S
    if total_hcp >= 6 and suit_counts["S"] >= 4:
        possibleBids.append(1)
    # 1NT
    if analyzer.is_balanced_distribution(dist) and suit_counts["H"] <= 2:
        if 6 <= total_hcp <= 10:
            possibleBids.append(2)
        if 11 <= total_hcp <= 13:
            possibleBids.append(6)
        if 14 <= total_hcp <= 15:
            possibleBids.append(8)
        if total_hcp >= 16:
            possibleBids.append(10)
    # 2C, 2D
    if total_hcp >= 12:
        if (suit_counts["C"] >= 5 and suit_counts["H"] <= 2):
            possibleBids.append(3)
        if (suit_counts["D"]>= 5 and suit_counts["H"] <=2):
            possibleBids.append(4)
    # 2H, 3H, 4H
    if suit_counts["H"] >= 3:
        if 6<= total_hcp <= 9:
            possibleBids.append(5)
        if 10<= total_hcp <= 11:
            possibleBids.append(7)
        if 12<= total_hcp <= 15:
            possibleBids.append(9)
    #Pass
    if not possibleBids:
        possibleBids.append(0)

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]

    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }
