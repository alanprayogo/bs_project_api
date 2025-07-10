from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_1c(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    # Define the possible bids and their meanings
    bid_meanings = {
        1:"Bid 1D",
        2:"Bid 1H",
        3:"Bid 1S",
        4:"Bid 1NT",
        5:"Bid 2C",
        6:"Bid 2D",
        7:"Bid 2H",
        8:"Bid 2S",
        9:"Bid 2NT",
        10:"Bid 3NT",
        0:"Bid 4C",
    }

    # Determine the contract based on HCP and suit counts
    # 1D
    if 0 <= total_hcp <=7:
        possibleBids.append(1)
    # 1H, 1S, 2C, 2D, 2H, 2S
    if total_hcp >= 8:
        if suit_counts["H"] >= 5:
            possibleBids.append(2)
        if suit_counts["S"] >= 5:
            possibleBids.append(3)
        if suit_counts["C"] >= 5:
            possibleBids.append(5)
        if suit_counts["D"] >= 5:
            possibleBids.append(6)
        if shdc == [4, 4, 4, 1] or shdc == [1, 4, 4, 4]:
            possibleBids.append(7)
        if shdc == [4, 4, 1, 4] or shdc == [4, 1, 4, 4]:
            possibleBids.append(8)
    # 1NT, 2NT, 3NT
    if analyzer.is_balanced_distribution and not analyzer.has_five_card_major(suit_counts):
        if 8 <= total_hcp <=10:
            possibleBids.append(4)        
        if 11 <= total_hcp <=13:
            possibleBids.append(9)        
        if 14 <= total_hcp <=15:
            possibleBids.append(10)        

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]

    # Output
    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }

    