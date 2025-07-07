from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_1nt(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 2C",
        2:"Bid 2D",
        3:"Bid 2H",
        4:"Bid 2S",
        5:"Bid 2NT",
        6:"Bid 3C",
        7:"Bid 3NT",
        8:"Bid 4C",
        0:"Pass"
    }

    # Determine the contract based on HCP and suit counts
    # 2D
    if suit_counts["H"] >= 5:
        possibleBids.append(2)
    # 2H
    if suit_counts["S"] >= 5:
        possibleBids.append(3)
    # 2C, 2S, 3C
    if total_hcp >= 8:
        if suit_counts["H"] == 4 or suit_counts["S"] == 4:
            possibleBids.append(1)
        elif suit_counts["H"] == 4 and suit_counts["S"] == 4:
            possibleBids.append(1)
        elif suit_counts["C"] >= 5:
            possibleBids.append(4)
        elif suit_counts["D"] >= 5:
            possibleBids.append(6)
    
    #Pass
    if not possibleBids:
        if 8 <= total_hcp <= 9:
            possibleBids.append(4)
        if 10 <= total_hcp <= 15:
            possibleBids.append(7)
        if total_hcp >= 16:
            possibleBids.append(8)

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]

    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }