from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_1s(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 1NT",
        2:"Bid 2C",
        3:"Bid 2D",
        4:"Bid 2H",
        5:"Bid 2S",
        6:"Bid 2NT",
        7:"Bid 3S",
        8:"Bid 3NT",
        9:"Bid 4S",
        10:"Bid 4NT",
        0:"Pass"
    }

    # Determine the contract based on HCP and suit counts
    # 1NT
    if analyzer.is_balanced_distribution(dist) and suit_counts["S"] <= 2:
        if 6<=total_hcp <=10:
            possibleBids.append(1)
        if 11<=total_hcp <=13:
            possibleBids.append(6)
        if 14<=total_hcp <=15:
            possibleBids.append(8)
        if total_hcp >=16:
            possibleBids.append(10)
    # 2C, 2D
    if total_hcp >= 12:
        # 2C
        if (suit_counts["C"] >= 5 and suit_counts["S"] <= 2):
            possibleBids.append(2)
        # 2D
        if (suit_counts["D"] >= 5 and suit_counts["S"] <= 2):
            possibleBids.append(3)
    # 2H
    if total_hcp >= 12 and (suit_counts["H"] >= 5 and suit_counts["S"] <=2):
        possibleBids.append(4)
    # 2S, 3S, 4S
    if suit_counts["S"] >= 3:
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
