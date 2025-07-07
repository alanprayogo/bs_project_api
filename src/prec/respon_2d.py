from utils.bridge_analyzer import BridgeHandAnalyzer

def prec_respon_2d(hand):
    analyzer = BridgeHandAnalyzer()
    analysis = analyzer.calculate_hcp_and_distribution(hand)
    
    total_hcp = analysis['total_hcp']
    suit_counts = analysis['suit_counts']
    dist = analysis['dist']
    shdc = analysis['shdc']
    distribusi = analysis['distribusi']

    possibleBids = []

    bid_meanings = {
        1:"Bid 2H",
        2:"Bid 2S",
        3:"Bid 2NT",
        4:"Bid 3C",
        5:"Bid 3D",
        6:"Bid 3H",
        7:"Bid 3S",
        8:"Bid 4C",
        9:"Bid 4D",
        10:"Bid 4H",
        11:"Bid 4S",
        0:"Coming Soon Coy, Aku yo bingung og"
    }

    # Determine the contract based on HCP and suit counts

    # set shortMajor
    if suit_counts["H"] > suit_counts["S"]:
        shortMajor = "H"
    else:
        shortMajor = "S"
    
    # set shortSuit
    shortSuit = min(suit_counts, key=suit_counts.get)

    # Rule of Seventeen -> 2NT 
    if total_hcp + suit_counts[shortMajor] >= 17:
        possibleBids.append(3)
    else:
        possibleBids.append(0)
    
    output = bid_meanings[max(possibleBids)] 

    return {
        'result': output,
        'hcp': total_hcp,
        'distribusi': distribusi,
    }