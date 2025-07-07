from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_2s(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 2NT",
        0:"Pass"
    }

    # Determine the contract based on HCP and suit counts
    #2S
    if total_hcp >= 8:
        possibleBids.append(1)
    else:
        possibleBids.append(0)

    output = bid_meanings[max(possibleBids)] if possibleBids else bid_meanings[0]

    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }