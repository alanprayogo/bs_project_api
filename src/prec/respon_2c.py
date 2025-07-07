from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_2c(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 2D",
        2:"Bid 2H",
        3:"Bid 2S",
        4:"Bid 2NT",
        5:"Bid 3D",
        6:"Bid 3NT",
        7:"Bid 4NT",
        0:"Pass"
    }

    # Determine the contract based on HCP and suit counts
    #2D, 2H, 2S
    if total_hcp >= 8:
        if suit_counts["H"] == 4 or suit_counts["S"] == 4:
            possibleBids.append(1)
        elif suit_counts["H"] == 4 and suit_counts["S"] == 4:
            possibleBids.append(1)
        elif suit_counts["H"] >= 5:
            possibleBids.append(2)
        elif suit_counts["S"] >= 5:
            possibleBids.append(3)
    
    #2NT, 3C, 3NT, 4NT
    if (suit_counts["H"] <= 3 or suit_counts["S"] <= 3):
        if total_hcp >=11 and suit_counts["D"] >= 6:
            possibleBids.append(5)
        elif 11 <= total_hcp <= 13:
            possibleBids.append(4)
        elif 14 <= total_hcp <= 15:
            possibleBids.append(6)
        elif total_hcp >= 16:
            possibleBids.append(7)
    #Pass
    if not possibleBids:
        possibleBids.append(0)

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]

    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }