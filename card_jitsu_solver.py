"""
Card-Jitsu Game Theory Optimal (GTO) Solver
-------------------------------------------
This engine reverse-engineers the Club Penguin minigame 'Card-Jitsu' into a rigorous mathematical model.
It features two AI agents:
1. An Exploitative Bot: This bot uses Bayesian updating to track the opponent's tendencies and exploits them for maximum gain.
2. A CFR-Trained Bot: This bot employs Counterfactual Regret Minimization (CFR) to learn an approximate Nash equilibrium strategy over time.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple
import itertools
from functools import lru_cache
import copy

# enums and core data structures

class Element(Enum):
    FIRE = 0
    WATER = 1
    SNOW = 2

class Colour(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    ORANGE = 4
    PURPLE = 5

class PowerEffect(Enum):
    """ Comprehensive list of the specific Power Effects in Card-Jitsu, categorized by their impact type."""
    NONE = 0
    
    # effects that impact next round
    BLOCK_FIRE = 1
    BLOCK_WATER = 2
    BLOCK_SNOW = 3
    PLUS_TWO_NEXT_TURN = 4
    MINUS_TWO_NEXT_TURN = 5
    
    # effects that impact current round
    LOW_WINS = 6
    SHIFT_FIRE_TO_SNOW = 7
    SHIFT_SNOW_TO_WATER = 8
    SHIFT_WATER_TO_FIRE = 9
    
    # tableau discards
    DISCARD_FIRE = 10
    DISCARD_WATER = 11
    DISCARD_SNOW = 12
    DISCARD_RED = 13
    DISCARD_BLUE = 14
    DISCARD_GREEN = 15
    DISCARD_YELLOW = 16
    DISCARD_ORANGE = 17
    DISCARD_PURPLE = 18
    
    # hand destructors
    DESTROY_ALL_RED = 19
    DESTROY_ALL_BLUE = 20
    DESTROY_ALL_GREEN = 21
    DESTROY_ALL_YELLOW = 22
    DESTROY_ALL_ORANGE = 23
    DESTROY_ALL_PURPLE = 24

@dataclass(frozen=True)
class Card:
    element: Element
    value: int # 1-12
    colour: Colour
    effect: PowerEffect = PowerEffect.NONE

    def __repr__(self):
        effect_str = f" [{self.effect.name}]" if self.effect != PowerEffect.NONE else ""
        return f"{self.value} {self.element.name} {self.colour.name}{effect_str}"

# CFR memory node

class Node:
    """
    Stores information for a specific InfoSet.
    Stores accumulated regrets over the many training iterations to calculate the Nash Equilibrium via Regret Matching.
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = {i: 0.0 for i in range(num_actions)}
        self.strategy_sum = {i: 0.0 for i in range(num_actions)}

    def get_strategy(self, realization_weight: float) -> dict:
        """Calculates current strategy proportionally to positive regrets."""
        strategy = {}
        normalizing_sum = 0.0
        for a in range(self.num_actions):
            strategy[a] = max(self.regret_sum[a], 0.0)
            normalizing_sum += strategy[a]
            
        for a in range(self.num_actions):
            if normalizing_sum > 0: strategy[a] /= normalizing_sum
            else: strategy[a] = 1.0 / self.num_actions
            self.strategy_sum[a] += strategy[a] * realization_weight
        return strategy

    def get_average_strategy(self) -> dict:
        """Returns the final, mathematically unexploitable percentage distribution over actions after training."""
        avg_strategy = {}
        normalizing_sum = sum(self.strategy_sum.values())
        for a in range(self.num_actions):
            if normalizing_sum > 0: avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else: avg_strategy[a] = 1.0 / self.num_actions
        return avg_strategy

# Global Memory Bank mapping InfoSet strings to Nodes for CFR training
NODE_MAP: Dict[str, Node] = {}

# global deck distributions

MASTER_DECK_DISTRIBUTION = {
    Element.FIRE: {1:0, 2:11, 3:19, 4:26, 5:25, 6:25, 7:21, 8:6, 9:10, 10:8, 11:11, 12:9},
    Element.WATER: {1:0, 2:23, 3:20, 4:25, 5:26, 6:21, 7:16, 8:17, 9:9, 10:8, 11:11, 12:7},
    Element.SNOW: {1:0, 2:11, 3:17, 4:26, 5:22, 6:23, 7:12, 8:10, 9:10, 10:9, 11:7, 12:10}
}   

COLOUR_DISTRIBUTION = {
    Element.FIRE: {Colour.RED: 39, Colour.BLUE: 30, Colour.GREEN: 18, Colour.YELLOW: 25, Colour.ORANGE: 30, Colour.PURPLE: 29},
    Element.WATER: {Colour.RED: 25, Colour.BLUE: 44, Colour.GREEN: 37, Colour.YELLOW: 23, Colour.ORANGE: 26, Colour.PURPLE: 28},
    Element.SNOW: {Colour.RED: 16, Colour.BLUE: 26, Colour.GREEN: 32, Colour.YELLOW: 25, Colour.ORANGE: 28, Colour.PURPLE: 30}
}

POWER_DISTRIBUTION = {
    Element.FIRE: {PowerEffect.NONE: 137, PowerEffect.BLOCK_FIRE: 0, PowerEffect.BLOCK_WATER: 0, PowerEffect.BLOCK_SNOW: 6, PowerEffect.PLUS_TWO_NEXT_TURN: 2, PowerEffect.MINUS_TWO_NEXT_TURN: 2, PowerEffect.LOW_WINS: 1, PowerEffect.SHIFT_FIRE_TO_SNOW: 0, PowerEffect.SHIFT_SNOW_TO_WATER: 0, PowerEffect.SHIFT_WATER_TO_FIRE: 5, PowerEffect.DISCARD_FIRE: 0, PowerEffect.DISCARD_WATER: 0, PowerEffect.DISCARD_SNOW: 4, PowerEffect.DISCARD_RED: 5, PowerEffect.DISCARD_BLUE: 0, PowerEffect.DISCARD_GREEN: 0, PowerEffect.DISCARD_YELLOW: 1, PowerEffect.DISCARD_ORANGE: 5, PowerEffect.DISCARD_PURPLE: 0, PowerEffect.DESTROY_ALL_RED: 1, PowerEffect.DESTROY_ALL_BLUE: 1, PowerEffect.DESTROY_ALL_GREEN: 1, PowerEffect.DESTROY_ALL_YELLOW: 0, PowerEffect.DESTROY_ALL_ORANGE: 0, PowerEffect.DESTROY_ALL_PURPLE: 0},
    Element.WATER: {PowerEffect.NONE: 148, PowerEffect.BLOCK_FIRE: 7, PowerEffect.BLOCK_WATER: 0, PowerEffect.BLOCK_SNOW: 0, PowerEffect.PLUS_TWO_NEXT_TURN: 3, PowerEffect.MINUS_TWO_NEXT_TURN: 2, PowerEffect.LOW_WINS: 1, PowerEffect.SHIFT_FIRE_TO_SNOW: 0, PowerEffect.SHIFT_SNOW_TO_WATER: 5, PowerEffect.SHIFT_WATER_TO_FIRE: 0, PowerEffect.DISCARD_FIRE: 6, PowerEffect.DISCARD_WATER: 0, PowerEffect.DISCARD_SNOW: 0, PowerEffect.DISCARD_RED: 1, PowerEffect.DISCARD_BLUE: 6, PowerEffect.DISCARD_GREEN: 2, PowerEffect.DISCARD_YELLOW: 0, PowerEffect.DISCARD_ORANGE: 1, PowerEffect.DISCARD_PURPLE: 0, PowerEffect.DESTROY_ALL_RED: 0, PowerEffect.DESTROY_ALL_BLUE: 0, PowerEffect.DESTROY_ALL_GREEN: 0, PowerEffect.DESTROY_ALL_YELLOW: 1, PowerEffect.DESTROY_ALL_ORANGE: 0, PowerEffect.DESTROY_ALL_PURPLE: 0},
    Element.SNOW: {PowerEffect.NONE: 122, PowerEffect.BLOCK_FIRE: 0, PowerEffect.BLOCK_WATER: 6, PowerEffect.BLOCK_SNOW: 0, PowerEffect.PLUS_TWO_NEXT_TURN: 2, PowerEffect.MINUS_TWO_NEXT_TURN: 2, PowerEffect.LOW_WINS: 1, PowerEffect.SHIFT_FIRE_TO_SNOW: 5, PowerEffect.SHIFT_SNOW_TO_WATER: 1, PowerEffect.SHIFT_WATER_TO_FIRE: 0, PowerEffect.DISCARD_FIRE: 0, PowerEffect.DISCARD_WATER: 7, PowerEffect.DISCARD_SNOW: 0, PowerEffect.DISCARD_RED: 0, PowerEffect.DISCARD_BLUE: 0, PowerEffect.DISCARD_GREEN: 1, PowerEffect.DISCARD_YELLOW: 2, PowerEffect.DISCARD_ORANGE: 0, PowerEffect.DISCARD_PURPLE: 6, PowerEffect.DESTROY_ALL_RED: 0, PowerEffect.DESTROY_ALL_BLUE: 0, PowerEffect.DESTROY_ALL_GREEN: 0, PowerEffect.DESTROY_ALL_YELLOW: 0, PowerEffect.DESTROY_ALL_ORANGE: 1, PowerEffect.DESTROY_ALL_PURPLE: 1}
    }

class DeckTracker:
    """Tracks dynamic deck depletion to calculate mathematically accurate Blockers and True Outs."""
    def __init__(self, starting_deck_dist: dict, starting_colour_dist: dict, starting_power_dist: dict):
        self.deck_dist = copy.deepcopy(starting_deck_dist)
        self.colour_dist = copy.deepcopy(starting_colour_dist)
        self.power_dist = copy.deepcopy(starting_power_dist)
        self.total_cards = sum(sum(vals.values()) for vals in self.deck_dist.values())

    def remove_card(self, card: Card):
        if self.deck_dist[card.element].get(card.value, 0) > 0:
            self.deck_dist[card.element][card.value] -= 1
        if self.colour_dist[card.element].get(card.colour, 0) > 0:
            self.colour_dist[card.element][card.colour] -= 1
        if self.power_dist[card.element].get(card.effect, 0) > 0:
            self.power_dist[card.element][card.effect] -= 1
        self.total_cards -= 1

    def destroy_all_of_colour(self, colour: Colour):
        """Mechanic for DESTROY_ALL power cards."""
        for element in Element:
            if colour in self.colour_dist[element]:
                self.total_cards -= self.colour_dist[element][colour]
                self.colour_dist[element][colour] = 0

    def get_colour_count(self, element: Element, colour: Colour) -> int:
        return self.colour_dist[element].get(colour, 0)

class OpponentTracker:
    """
    Uses Bayesian updating to model the opponent's psychology across three logical levels:
    Level 0: Random Plays.
    Level 1: Self-focused (plays to satisfy own outs).
    Level 2: Defensive (plays to block our outs).
    """
    def __init__(self):
        self.beliefs = {0: 0.2, 1: 0.4, 2: 0.4}

    def update_beliefs(self, played_element: Element, my_tableau: List[Card], opp_tableau: List[Card], my_deck: DeckTracker, opp_deck: DeckTracker, blocked_element: Element = None):
        my_outs = calculate_true_outs(my_tableau, my_deck)
        opp_outs = calculate_true_outs(opp_tableau, opp_deck)

        # calculate likelihood of observed play under each logic model
        p_moves = {
            0: get_opp_probabilities(0, my_outs, opp_outs, blocked_element)[played_element],
            1: get_opp_probabilities(1, my_outs, opp_outs, blocked_element)[played_element],
            2: get_opp_probabilities(2, my_outs, opp_outs, blocked_element)[played_element]
        }

        # Bayesian update
        total_prob = sum(p_moves[k] * self.beliefs[k] for k in range(3))
        
        if total_prob > 0:
            floor = 0.01 
            for k in range(3):
                self.beliefs[k] = max(floor, (p_moves[k] * self.beliefs[k]) / total_prob)
                
            new_total = sum(self.beliefs.values())
            for k in range(3):
                self.beliefs[k] /= new_total

    def get_blended_probabilities(self, my_outs: Dict[Element, int], opp_outs: Dict[Element, int], blocked_element: Element = None) -> Dict[Element, float]:
        """Returns the final predictive probability vector for the opponent's next play."""
        blended_probs = {Element.FIRE: 0.0, Element.WATER: 0.0, Element.SNOW: 0.0}
        for level, weight in self.beliefs.items():
            level_probs = get_opp_probabilities(level, my_outs, opp_outs, blocked_element)
            for e in Element: blended_probs[e] += weight * level_probs[e]
        return blended_probs

# game logic engine

def get_opp_probabilities(level: int, my_outs: Dict[Element, int], opp_outs: Dict[Element, int], blocked_element: Element = None) -> Dict[Element, float]:
    probs = {}
    if level == 0:
        for e in Element: probs[e] = 1.0 / 3.0
    elif level == 1:
        total_opp_outs = sum(opp_outs.values())
        denominator = 3.0 + total_opp_outs
        for e in Element: probs[e] = (1.0 + opp_outs[e]) / denominator
    elif level == 2:
        total_my_outs = sum(my_outs.values())
        denominator = 3.0 + total_my_outs
        probs[Element.FIRE] = (1.0 + my_outs[Element.SNOW]) / denominator
        probs[Element.WATER] = (1.0 + my_outs[Element.FIRE]) / denominator
        probs[Element.SNOW] = (1.0 + my_outs[Element.WATER]) / denominator

    if blocked_element is not None: probs[blocked_element] = 0.0
    total_probs = sum(probs.values())

    if total_probs == 0:
        valid_elements = [e for e in Element if e != blocked_element]
        for e in valid_elements: probs[e] = 1.0 / max(1, len(valid_elements))
    else:
        for e in Element: probs[e] /= total_probs
    return probs

def resolve_round(my_card: Card, opp_card: Card) -> int:
    """Resolves standard RPS, applying immediate rule-breaking Element Shifts."""
    my_elem = my_card.element
    opp_elem = opp_card.element
    effects = [my_card.effect, opp_card.effect]
    
    if PowerEffect.SHIFT_FIRE_TO_SNOW in effects:
        if my_elem == Element.FIRE: my_elem = Element.SNOW
        if opp_elem == Element.FIRE: opp_elem = Element.SNOW
            
    if PowerEffect.SHIFT_SNOW_TO_WATER in effects:
        if my_elem == Element.SNOW: my_elem = Element.WATER
        if opp_elem == Element.SNOW: opp_elem = Element.WATER
            
    if PowerEffect.SHIFT_WATER_TO_FIRE in effects:
        if my_elem == Element.WATER: my_elem = Element.FIRE
        if opp_elem == Element.WATER: opp_elem = Element.FIRE

    if my_elem == opp_elem: return 0
        
    win_matrix = {(Element.FIRE, Element.SNOW): 1, (Element.SNOW, Element.WATER): 1, (Element.WATER, Element.FIRE): 1}
    return win_matrix.get((my_elem, opp_elem), -1)

def get_tie_probabilities(element: Element, our_final_val: int, opp_deck: DeckTracker, is_low_wins: bool = False) -> Tuple[float, float, float]:
    element_deck = opp_deck.deck_dist[element]
    total_cards = sum(element_deck.values())

    if total_cards == 0: return (1.0, 0.0, 0.0) 

    if is_low_wins:
        wins = sum(count for val, count in element_deck.items() if val > our_final_val)
        losses = sum(count for val, count in element_deck.items() if val < our_final_val)
    else:
        wins = sum(count for val, count in element_deck.items() if val < our_final_val)
        losses = sum(count for val, count in element_deck.items() if val > our_final_val)
        
    washes = element_deck.get(our_final_val, 0)
    return (wins / total_cards, losses / total_cards, washes / total_cards)

def check_win(tableau: List[Card]) -> bool:
    """Evaluates the board state for each win condition: 3 cards of unique colour, either all of the same element or all of different elements."""
    if len(tableau) < 3: return False
    for combo in itertools.combinations(tableau, 3):
        if len({c.colour for c in combo}) != 3: continue
        unique_elements = {c.element for c in combo}   
        if len(unique_elements) == 1 or len(unique_elements) == 3: return True
    return False        

def calculate_true_outs(tableau: List[Card], deck: DeckTracker) -> Dict[Element, int]:
    outs = {Element.FIRE: 0, Element.WATER: 0, Element.SNOW: 0} 
    if check_win(tableau): return outs
    for element in Element:
        for colour in Colour:
            cards_remaining = deck.get_colour_count(element, colour)
            if cards_remaining == 0: continue 
            test_card = Card(element, 0, colour) 
            if check_win(list(tuple(tableau) + (test_card,))): outs[element] += cards_remaining
    return outs

@lru_cache(maxsize=10000)
def get_player_equity(tableau: Tuple[Card, ...], deck: DeckTracker) -> float:
    """Calculates Phase 1 (immediate) and Phase 2 (discounted future) combinatorial equity."""
    if check_win(list(tableau)): return 1.0
    phase1_outs_dict = calculate_true_outs(list(tableau), deck)
    phase1_equity = sum(phase1_outs_dict.values()) / max(1, deck.total_cards)

    future_outs_sum, non_winning_cards_count = 0, 0
    for element in Element:
        for colour in Colour:
            test_card = Card(element, 0, colour) 
            hypothetical_tableau = tableau + (test_card,)
            if not check_win(list(hypothetical_tableau)):
                future_outs_dict = calculate_true_outs(list(hypothetical_tableau), deck)
                future_outs_sum += sum(future_outs_dict.values())
                non_winning_cards_count += 1

    phase2_equity = 0.0
    if non_winning_cards_count > 0:
        phase2_equity = ((future_outs_sum / non_winning_cards_count) / max(1, deck.total_cards)) * 0.25 
    return min(phase1_equity + phase2_equity, 0.95)

def get_board_value(my_tableau: List[Card], opp_tableau: List[Card], my_deck: DeckTracker, opp_deck: DeckTracker) -> float:
    """Returns the state value V(S) of the current board."""
    if check_win(my_tableau): return 1.0
    if check_win(opp_tableau): return -1.0
    return get_player_equity(tuple(my_tableau), my_deck) - get_player_equity(tuple(opp_tableau), opp_deck)

# EV lookahead and solver

def calculate_best_move(hand: List[Card], my_tableau: List[Card], opp_tableau: List[Card], tracker: OpponentTracker, my_deck: DeckTracker, opp_deck: DeckTracker, active_block: Element = None, active_my_mod: int = 0, active_opp_mod: int = 0) -> Card:
    """
    Exploitative Solver: Evaluates EV of every card in hand against the opponent's predictive probability vector,
    accounting for dynamic Power Effect lookaheads.
    """
    my_outs = calculate_true_outs(my_tableau, my_deck)
    opp_outs = calculate_true_outs(opp_tableau, opp_deck)
    p_opp = tracker.get_blended_probabilities(my_outs, opp_outs, active_block)
    
    best_card, max_ev = None, float('-inf')

    for card in hand:
        expected_value = 0.0
        my_final_val = card.value + active_my_mod

        for opp_element in Element:
            prob_they_play_elem = p_opp[opp_element]
            if prob_they_play_elem == 0.0: continue
            
            total_effect_cards = sum(POWER_DISTRIBUTION[opp_element].values())
            
            for opp_effect, effect_count in POWER_DISTRIBUTION[opp_element].items():
                if effect_count == 0: continue
                prob_effect = effect_count / max(1, total_effect_cards)
                joint_prob = prob_they_play_elem * prob_effect

                mock_opp_card = Card(opp_element, 0, Colour.RED, opp_effect)
                result = resolve_round(card, mock_opp_card)

                if result == 1:
                    simulated_opp_tableau = list(opp_tableau)
                    
                    # true tableau discard effects
                    if card.effect.name.startswith("DISCARD_"):
                        target = card.effect.name.split("_")[1]
                        for c in simulated_opp_tableau:
                            if c.element.name == target or c.colour.name == target:
                                simulated_opp_tableau.remove(c)
                                break
                                
                    new_state_value = get_board_value(my_tableau + [card], simulated_opp_tableau, my_deck, opp_deck)
                    if card.effect.name.startswith("BLOCK_"):
                        new_state_value += 0.2 # Heuristic bonus for future constraint
                    expected_value += joint_prob * min(new_state_value, 0.95)

                elif result == -1:
                    lose_ev_sum = 0.0
                    total_colours = sum(COLOUR_DISTRIBUTION[opp_element].values())
                    for possible_colour in Colour:
                        prob_colour = COLOUR_DISTRIBUTION[opp_element].get(possible_colour, 0) / max(1, total_colours)
                        hypothetical_opp_card = Card(opp_element, 0, possible_colour, opp_effect)
                        
                        simulated_my_tableau = list(my_tableau)
                        
                        if opp_effect.name.startswith("DISCARD_"):
                            target = opp_effect.name.split("_")[1]
                            for c in simulated_my_tableau:
                                if c.element.name == target or c.colour.name == target:
                                    simulated_my_tableau.remove(c)
                                    break
                        
                        lose_ev_sum += prob_colour * get_board_value(simulated_my_tableau, opp_tableau + [hypothetical_opp_card], my_deck, opp_deck)
                    expected_value += joint_prob * lose_ev_sum

                elif result == 0:
                    is_low_wins = (card.effect == PowerEffect.LOW_WINS) or (opp_effect == PowerEffect.LOW_WINS)
                    prob_win_tie, prob_lose_tie, prob_wash = get_tie_probabilities(card.element, my_final_val, opp_deck, is_low_wins)

                    win_val = get_board_value(my_tableau + [card], opp_tableau, my_deck, opp_deck)
                    wash_val = get_board_value(my_tableau, opp_tableau, my_deck, opp_deck)

                    lose_ev_sum = 0.0
                    total_colours = sum(COLOUR_DISTRIBUTION[opp_element].values())
                    for possible_colour in Colour:
                        prob_colour = COLOUR_DISTRIBUTION[opp_element].get(possible_colour, 0) / max(1, total_colours)
                        lose_ev_sum += prob_colour * get_board_value(my_tableau, opp_tableau + [Card(opp_element, 0, possible_colour, opp_effect)], my_deck, opp_deck)

                    expected_value += joint_prob * ((prob_win_tie * win_val) + (prob_lose_tie * lose_ev_sum) + (prob_wash * wash_val))

        print(f"Card: {card}, EV: {expected_value:.4f}")
        if expected_value > max_ev:
            max_ev = expected_value
            best_card = card

    return best_card

# cfr training

def get_infoset_string(hand: List[Card], my_tableau: List[Card], opp_tableau: List[Card], active_block: Element = None, my_mod: int=0, opp_mod: int=0) -> str:
    """Compresses the observable game state into a unique, hashed string for the CFR algorithm's memory bank."""
    hand_str = ",".join(sorted([repr(c) for c in hand]))
    my_tab_str = ",".join(sorted([repr(c) for c in my_tableau]))
    opp_tab_str = ",".join(sorted([repr(c) for c in opp_tableau]))
    block_str = active_block.name if active_block else "NONE"
    return f"H:[{hand_str}]|M:[{my_tab_str}]|O:[{opp_tab_str}]|B:{block_str}|Mod:{my_mod}/{opp_mod}"

def cfr_train(p1_hand: List[Card], p2_hand: List[Card], p1_tableau: List[Card], p2_tableau: List[Card], p1_deck: DeckTracker, p2_deck: DeckTracker, p1_block: Element = None, p2_block: Element = None, p1_mod: int = 0, p2_mod: int = 0, reach_p1: float = 1.0, reach_p2: float = 1.0) -> float:
    """
    Recursively simulates simultaneous decision trees.
    Computes counterfactual values and updates Regret to discover Nash Equilibria.
    """
    if check_win(p1_tableau): return 1.0
    if check_win(p2_tableau): return -1.0
    if not p1_hand or not p2_hand: return get_board_value(p1_tableau, p2_tableau, p1_deck, p2_deck)

    p1_infoset = get_infoset_string(p1_hand, p1_tableau, p2_tableau, p1_block, p1_mod, p2_mod)
    p2_infoset = get_infoset_string(p2_hand, p2_tableau, p1_tableau, p2_block, p2_mod, p1_mod)

    if p1_infoset not in NODE_MAP: NODE_MAP[p1_infoset] = Node(len(p1_hand))
    if p2_infoset not in NODE_MAP: NODE_MAP[p2_infoset] = Node(len(p2_hand))

    node_p1, node_p2 = NODE_MAP[p1_infoset], NODE_MAP[p2_infoset]
    strategy_p1, strategy_p2 = node_p1.get_strategy(reach_p1), node_p2.get_strategy(reach_p2)

    action_values_p1 = {i: 0.0 for i in range(len(p1_hand))}
    action_values_p2 = {j: 0.0 for j in range(len(p2_hand))}
    node_ev_p1 = 0.0

    for i, card_p1 in enumerate(p1_hand):
        for j, card_p2 in enumerate(p2_hand):
            matchup_prob = strategy_p1[i] * strategy_p2[j]
            result = resolve_round(card_p1, card_p2)
            
            p1_final_val = card_p1.value + p1_mod
            p2_final_val = card_p2.value + p2_mod
            
            if result == 0:
                is_low_wins = (card_p1.effect == PowerEffect.LOW_WINS) or (card_p2.effect == PowerEffect.LOW_WINS)
                if is_low_wins: result = 1 if p1_final_val < p2_final_val else -1 if p1_final_val > p2_final_val else 0
                else: result = 1 if p1_final_val > p2_final_val else -1 if p1_final_val < p2_final_val else 0

            next_p1_tab, next_p2_tab = list(p1_tableau), list(p2_tableau)
            next_p1_hand, next_p2_hand = list(p1_hand), list(p2_hand)
            next_p1_hand.pop(i)
            next_p2_hand.pop(j)

            next_p1_block, next_p2_block = None, None
            next_p1_mod, next_p2_mod = 0, 0
            
            # saves cpu memory
            next_p1_deck = p1_deck
            next_p2_deck = p2_deck

            # logic isolated to parallel timelines
            if result == 1: 
                next_p1_tab.append(card_p1)
                
                if card_p1.effect == PowerEffect.BLOCK_FIRE: next_p2_block = Element.FIRE
                elif card_p1.effect == PowerEffect.BLOCK_WATER: next_p2_block = Element.WATER
                elif card_p1.effect == PowerEffect.BLOCK_SNOW: next_p2_block = Element.SNOW
                
                elif card_p1.effect == PowerEffect.PLUS_TWO_NEXT_TURN: next_p1_mod = 2
                elif card_p1.effect == PowerEffect.MINUS_TWO_NEXT_TURN: next_p2_mod = -2
                
                elif card_p1.effect.name.startswith("DESTROY_ALL_"):
                    next_p2_deck = copy.deepcopy(p2_deck) 
                    target_colour = Colour[card_p1.effect.name.split("_")[2]]
                    next_p2_deck.destroy_all_of_colour(target_colour)
                    
                elif card_p1.effect.name.startswith("DISCARD_"):
                    target = card_p1.effect.name.split("_")[1]
                    for c in next_p2_tab:
                        if c.element.name == target or c.colour.name == target:
                            next_p2_tab.remove(c)
                            break
                            
            elif result == -1: 
                next_p2_tab.append(card_p2)
                
                if card_p2.effect == PowerEffect.BLOCK_FIRE: next_p1_block = Element.FIRE
                elif card_p2.effect == PowerEffect.BLOCK_WATER: next_p1_block = Element.WATER
                elif card_p2.effect == PowerEffect.BLOCK_SNOW: next_p1_block = Element.SNOW
                
                elif card_p2.effect == PowerEffect.PLUS_TWO_NEXT_TURN: next_p2_mod = 2
                elif card_p2.effect == PowerEffect.MINUS_TWO_NEXT_TURN: next_p1_mod = -2
                
                elif card_p2.effect.name.startswith("DESTROY_ALL_"):
                    next_p1_deck = copy.deepcopy(p1_deck) 
                    target_colour = Colour[card_p2.effect.name.split("_")[2]]
                    next_p1_deck.destroy_all_of_colour(target_colour)
                    
                elif card_p2.effect.name.startswith("DISCARD_"):
                    target = card_p2.effect.name.split("_")[1]
                    for c in next_p1_tab:
                        if c.element.name == target or c.colour.name == target:
                            next_p1_tab.remove(c)
                            break

            payoff_p1 = cfr_train(next_p1_hand, next_p2_hand, next_p1_tab, next_p2_tab, 
                                  next_p1_deck, next_p2_deck, next_p1_block, next_p2_block, 
                                  next_p1_mod, next_p2_mod, 
                                  reach_p1 * strategy_p1[i], reach_p2 * strategy_p2[j])
            
            action_values_p1[i] += strategy_p2[j] * payoff_p1
            action_values_p2[j] += strategy_p1[i] * (-payoff_p1) 
            node_ev_p1 += matchup_prob * payoff_p1

    for i in range(len(p1_hand)): node_p1.regret_sum[i] += reach_p2 * (action_values_p1[i] - node_ev_p1)
    for j in range(len(p2_hand)): node_p2.regret_sum[j] += reach_p1 * (action_values_p2[j] - (-node_ev_p1))

    return node_ev_p1

# interface functions

def parse_card_input(prompt_text: str) -> Card:
    elem_map = {'F': Element.FIRE, 'W': Element.WATER, 'S': Element.SNOW}
    colour_map = {'R': Colour.RED, 'B': Colour.BLUE, 'G': Colour.GREEN, 'Y': Colour.YELLOW, 'O': Colour.ORANGE, 'P': Colour.PURPLE}
    
    effect_map = {
        'N': PowerEffect.NONE, 'BF': PowerEffect.BLOCK_FIRE, 'BW': PowerEffect.BLOCK_WATER, 'BS': PowerEffect.BLOCK_SNOW,
        '+2': PowerEffect.PLUS_TWO_NEXT_TURN, '-2': PowerEffect.MINUS_TWO_NEXT_TURN, 'LW': PowerEffect.LOW_WINS,
        'SFS': PowerEffect.SHIFT_FIRE_TO_SNOW, 'SSW': PowerEffect.SHIFT_SNOW_TO_WATER, 'SWF': PowerEffect.SHIFT_WATER_TO_FIRE,
        'DF': PowerEffect.DISCARD_FIRE, 'DW': PowerEffect.DISCARD_WATER, 'DS': PowerEffect.DISCARD_SNOW,
        'DR': PowerEffect.DISCARD_RED, 'DB': PowerEffect.DISCARD_BLUE, 'DG': PowerEffect.DISCARD_GREEN,
        'DY': PowerEffect.DISCARD_YELLOW, 'DO': PowerEffect.DISCARD_ORANGE, 'DP': PowerEffect.DISCARD_PURPLE,
        'DALLR': PowerEffect.DESTROY_ALL_RED, 'DALLB': PowerEffect.DESTROY_ALL_BLUE, 'DALLG': PowerEffect.DESTROY_ALL_GREEN,
        'DALLY': PowerEffect.DESTROY_ALL_YELLOW, 'DALLO': PowerEffect.DESTROY_ALL_ORANGE, 'DALLP': PowerEffect.DESTROY_ALL_PURPLE 
    }

    while True:
        user_in = input(prompt_text).strip().upper()
        parts = user_in.split()

        if len(parts) not in [3, 4]:
            print("Format: [Value] [Element] [Colour] [Optional: Effect] (e.g., 10 F R LW)")
            continue

        val_str, elem_str, colour_str = parts[:3]
        effect_str = parts[3] if len(parts) == 4 else 'N'

        try: val = int(val_str)
        except ValueError: continue

        if elem_str not in elem_map or colour_str not in colour_map or effect_str not in effect_map:
            print("Invalid code.")
            continue

        return Card(element=elem_map[elem_str], value=val, colour=colour_map[colour_str], effect=effect_map[effect_str])

def play_game():
    my_deck = DeckTracker(MASTER_DECK_DISTRIBUTION, COLOUR_DISTRIBUTION, POWER_DISTRIBUTION)
    opp_deck = DeckTracker(MASTER_DECK_DISTRIBUTION, COLOUR_DISTRIBUTION, POWER_DISTRIBUTION)

    print("=========================================")
    print("🔥💧❄️ CARD-JITSU EXPLOITATIVE SOLVER 🔥💧❄️")
    print("=========================================")
    
    tracker = OpponentTracker()
    my_tableau, opp_tableau, my_hand = [], [], []
    current_block, my_mod, opp_mod = None, 0, 0

    for i in range(5): my_hand.append(parse_card_input(f"Draw Card {i+1}/5: "))
    round_num = 1

    while True:
        get_player_equity.cache_clear() 
        print(f"\n=== ROUND {round_num} ===")
        print(f"My Tableau:  {my_tableau} | Modifier: {my_mod}")
        print(f"Opp Tableau: {opp_tableau} | Modifier: {opp_mod}")
        
        if current_block: print(f"⚠️ OPPONENT BLOCKED: {current_block.name}")

        print("\nCalculating best move...")
        best_card = calculate_best_move(my_hand, my_tableau, opp_tableau, tracker, my_deck, opp_deck, current_block, my_mod, opp_mod)
        print(f"\n>> GTO RECOMMENDED PLAY: {best_card} <<\n")

        played_card = parse_card_input("Enter the card YOU played: ")
        opp_card = parse_card_input("Enter the card OPPONENT played: ")

        pre_round_my_tab, pre_round_opp_tab = list(my_tableau), list(opp_tableau)
        next_block, next_my_mod, next_opp_mod = None, 0, 0

        # --- IMMEDIATE DIALOGUE: SHIFTS ---
        active_effects = [played_card.effect, opp_card.effect]
        if PowerEffect.SHIFT_FIRE_TO_SNOW in active_effects: print("🌀 POWER EFFECT: All Fire becomes Snow this round!")
        if PowerEffect.SHIFT_SNOW_TO_WATER in active_effects: print("🌀 POWER EFFECT: All Snow becomes Water this round!")
        if PowerEffect.SHIFT_WATER_TO_FIRE in active_effects: print("🌀 POWER EFFECT: All Water becomes Fire this round!")

        result = resolve_round(played_card, opp_card)
        
        # --- IMMEDIATE DIALOGUE: TIE BREAKERS ---
        if result == 0:
            is_low_wins = (played_card.effect == PowerEffect.LOW_WINS) or (opp_card.effect == PowerEffect.LOW_WINS)
            my_final, opp_final = played_card.value + my_mod, opp_card.value + opp_mod
            print(f"TIE! Values: You({my_final}) vs Opp({opp_final})")
            
            if is_low_wins: 
                print("⚠️ LOW WINS RULE IN EFFECT!")
                result = 1 if my_final < opp_final else -1 if my_final > opp_final else 0
            else: 
                result = 1 if my_final > opp_final else -1 if my_final < opp_final else 0

        # --- POST-WIN DIALOGUE & MODIFIERS ---
        if result == 1:
            print("Result: YOU WON!")
            my_tableau.append(played_card)
            
            if played_card.effect.name.startswith("BLOCK_"):
                target = played_card.effect.name.split("_")[1]
                next_block = Element[target]
                print(f"🚫 POWER EFFECT: Opponent cannot play {target} next turn!")
                
            elif played_card.effect == PowerEffect.PLUS_TWO_NEXT_TURN: 
                next_my_mod = 2
                print("⬆️ POWER EFFECT: Your next card gets +2!")
                
            elif played_card.effect == PowerEffect.MINUS_TWO_NEXT_TURN: 
                next_opp_mod = -2
                print("⬇️ POWER EFFECT: Opponent's next card gets -2!")
                
            elif played_card.effect.name.startswith("DESTROY_ALL_"): 
                target = played_card.effect.name.split("_")[2]
                opp_deck.destroy_all_of_colour(Colour[target])
                print(f"💥 POWER EFFECT: All opponent's {target} cards destroyed from deck!")
                
            elif played_card.effect.name.startswith("DISCARD_"):
                target = played_card.effect.name.split("_")[1] 
                for c in opp_tableau:
                    if c.element.name == target or c.colour.name == target:
                        opp_tableau.remove(c)
                        print(f"💥 POWER EFFECT: Destroyed opponent's {c} from tableau!")
                        break

        elif result == -1:
            print("Result: OPPONENT WON!")
            opp_tableau.append(opp_card)
            
            if opp_card.effect.name.startswith("BLOCK_"):
                target = opp_card.effect.name.split("_")[1]
                next_block = Element[target]
                print(f"🚫 POWER EFFECT: Opponent blocked you from playing {target} next turn!")
                
            elif opp_card.effect == PowerEffect.PLUS_TWO_NEXT_TURN: 
                next_opp_mod = 2
                print("⬆️ POWER EFFECT: Opponent's next card gets +2!")
                
            elif opp_card.effect == PowerEffect.MINUS_TWO_NEXT_TURN: 
                next_my_mod = -2
                print("⬇️ POWER EFFECT: Your next card gets -2!")
                
            elif opp_card.effect.name.startswith("DESTROY_ALL_"): 
                target = opp_card.effect.name.split("_")[2]
                my_deck.destroy_all_of_colour(Colour[target])
                print(f"💥 POWER EFFECT: All your {target} cards destroyed from deck!")
                
            elif opp_card.effect.name.startswith("DISCARD_"):
                target = opp_card.effect.name.split("_")[1] 
                for c in my_tableau:
                    if c.element.name == target or c.colour.name == target:
                        my_tableau.remove(c)
                        print(f"💥 POWER EFFECT: Opponent destroyed your {c} from tableau!")
                        break

        opp_deck.remove_card(opp_card)
        my_deck.remove_card(played_card)        

        if check_win(my_tableau):
            print("\n🏆🏆🏆 YOU WIN THE GAME! 🏆🏆🏆")
            break
        if check_win(opp_tableau):
            print("\n💀💀💀 OPPONENT WINS THE GAME! 💀💀💀")
            break

        tracker.update_beliefs(opp_card.element, pre_round_my_tab, pre_round_opp_tab, my_deck, opp_deck, current_block)
        current_block, my_mod, opp_mod = next_block, next_my_mod, next_opp_mod
        
        my_hand.remove(played_card)
        my_hand.append(parse_card_input("\nDraw new card: "))
        round_num += 1

def run_training_simulation(iterations: int = 10000):
    print("=========================================")
    print(f"🧠 CFR MATRIX TRAINING ({iterations} ITERATIONS) 🧠")
    print("=========================================")

    p1_hand = [Card(Element.FIRE, 10, Colour.RED,  PowerEffect.LOW_WINS), Card(Element.WATER, 4, Colour.BLUE), Card(Element.SNOW, 7, Colour.GREEN)]
    p2_hand = [Card(Element.FIRE, 3, Colour.YELLOW), Card(Element.WATER, 11, Colour.ORANGE), Card(Element.SNOW, 12, Colour.PURPLE)]
    p1_tableau, p2_tableau = [], []

    mock_deck_p1 = DeckTracker(MASTER_DECK_DISTRIBUTION, COLOUR_DISTRIBUTION, POWER_DISTRIBUTION)
    mock_deck_p2 = DeckTracker(MASTER_DECK_DISTRIBUTION, COLOUR_DISTRIBUTION, POWER_DISTRIBUTION)

    root_infoset_p1 = get_infoset_string(p1_hand, p1_tableau, p2_tableau)
    root_infoset_p2 = get_infoset_string(p2_hand, p2_tableau, p1_tableau)

    for i in range(iterations):
        if i % 2500 == 0 and i > 0: print(f"... {i} iterations ...")
        cfr_train(list(p1_hand), list(p2_hand), list(p1_tableau), list(p2_tableau), mock_deck_p1, mock_deck_p2)

    print("\n--- PLAYER 1 GTO STRATEGY ---")
    optimal_strategy_p1 = NODE_MAP[root_infoset_p1].get_average_strategy()
    for idx, prob in optimal_strategy_p1.items(): print(f"> {p1_hand[idx]}: {prob * 100:.2f}%")

    print("\n--- PLAYER 2 GTO STRATEGY ---")
    optimal_strategy_p2 = NODE_MAP[root_infoset_p2].get_average_strategy()
    for idx, prob in optimal_strategy_p2.items(): print(f"> {p2_hand[idx]}: {prob * 100:.2f}%")

if __name__ == "__main__":
    # to play the game interactively, uncomment the line below and comment out the training simulation.

    play_game() 
    run_training_simulation(10000)