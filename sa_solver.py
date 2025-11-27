import random
import math
from typing import List, Dict, Tuple, Generator


class GraphColoringSA:
    def __init__(self, graph: Dict[int, List[int]], num_colors: int):
        self.graph = graph
        self.num_colors = num_colors
        self.vertices = list(graph.keys())
        self.colors = list(range(num_colors))
        self.color_names = [
            'Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Cyan', 'Magenta', 'Pink', 'Brown',
            'Lime', 'Teal', 'Navy', 'Maroon', 'Olive', 'Silver', 'Gold', 'Coral', 'Lavender', 'Turquoise',
            'Indigo', 'Violet', 'Beige', 'Khaki', 'Salmon', 'Crimson', 'Aqua', 'Chartreuse', 'Fuchsia', 'Gray',
            'DarkGreen', 'DarkBlue', 'DarkRed', 'DarkOrange', 'DarkViolet', 'LightBlue', 'LightGreen', 'LightPink', 'LightYellow', 'LightGray'
        ]
    
    def initial_config(self, m: int) -> Dict[int, int]:
        return {vertex: random.choice(self.colors) for vertex in self.vertices}
    
    def calc_temp(self, iteration: int, initial_temp: float, cooling_rate: float = 0.95) -> float:
        return initial_temp * (cooling_rate ** iteration)
    
    def random_successor(self, current_coloring: Dict[int, int]) -> Dict[int, int]:
        new_coloring = current_coloring.copy()
        vertex = random.choice(self.vertices)
        available_colors = [c for c in self.colors if c != current_coloring[vertex]]
        if available_colors:
            new_coloring[vertex] = random.choice(available_colors)
        return new_coloring
    
    def count_conflicts(self, coloring: Dict[int, int]) -> int:
        conflicts = 0
        for vertex, neighbors in self.graph.items():
            for neighbor in neighbors:
                if coloring[vertex] == coloring[neighbor]:
                    conflicts += 1
        return conflicts // 2
    
    def _sim_anneal_core(self, m: int, iter_max: int, T: float, min_temp: float = 2.0, cooling_rate: float = 0.95):
        xcurr = self.initial_config(m)
        xbest = xcurr.copy()
        
        temperature_history = []
        conflict_history = []
        stop_reason = "max_iterations"
        
        try:
            for i in range(1, iter_max + 1):
                Tc = self.calc_temp(i, T, cooling_rate)
                temperature_history.append(Tc)
                
                if Tc <= min_temp:
                    stop_reason = "temperature_threshold"
                    conflicts_curr = self.count_conflicts(xcurr)
                    conflict_history.append(conflicts_curr)
                    break
                
                xnext = self.random_successor(xcurr)
                
                conflicts_curr = self.count_conflicts(xcurr)
                conflicts_next = self.count_conflicts(xnext)
                delta_E = conflicts_curr - conflicts_next
                
                conflict_history.append(conflicts_curr)
                
                if delta_E > 0:
                    xcurr = xnext
                    if self.count_conflicts(xbest) > self.count_conflicts(xcurr):
                        xbest = xcurr.copy()
                        if self.count_conflicts(xbest) == 0:
                            stop_reason = "solution_found"
                            conflict_history.append(conflicts_next)
                            break
                elif delta_E < 0:
                    prob = math.exp(delta_E / Tc) if Tc > 0 else 0
                    if random.random() < prob:
                        xcurr = xnext
                else:
                    if random.random() < 0.5:
                        xcurr = xnext
                
                if self.count_conflicts(xcurr) == 0:
                    xbest = xcurr.copy()
                    stop_reason = "solution_found"
                    break
        except Exception as e:
            import traceback
            print(f"Error in _sim_anneal_core: {e}")
            print(traceback.format_exc())
            stop_reason = "error"
        
        # Check if we found a solution after loop ends
        if self.count_conflicts(xbest) == 0 and stop_reason == "max_iterations":
            stop_reason = "solution_found"
        
        self._stop_reason = stop_reason
        return xbest, temperature_history, conflict_history, stop_reason
    
    def sim_anneal(self, m: int, iter_max: int, T: float, 
                   yield_progress: bool = False, min_temp: float = 2.0, cooling_rate: float = 0.95):
        if not yield_progress:
            return self._sim_anneal_core(m, iter_max, T, min_temp, cooling_rate)
        
        xcurr = self.initial_config(m)
        xbest = xcurr.copy()
        
        temperature_history = []
        conflict_history = []
        stop_reason = "max_iterations"
        
        for i in range(1, iter_max + 1):
            Tc = self.calc_temp(i, T, cooling_rate)
            temperature_history.append(Tc)
            
            if Tc <= min_temp:
                stop_reason = "temperature_threshold"
                conflicts_curr = self.count_conflicts(xcurr)
                conflict_history.append(conflicts_curr)
                
                yield (xcurr.copy(), i, Tc, conflicts_curr)
                break
            
            xnext = self.random_successor(xcurr)
            
            conflicts_curr = self.count_conflicts(xcurr)
            conflicts_next = self.count_conflicts(xnext)
            delta_E = conflicts_curr - conflicts_next
            
            conflict_history.append(conflicts_curr)
            
            yield (xcurr.copy(), i, Tc, conflicts_curr)
            
            if delta_E > 0:
                xcurr = xnext
                if self.count_conflicts(xbest) > self.count_conflicts(xcurr):
                    xbest = xcurr.copy()
                    if self.count_conflicts(xbest) == 0:
                        stop_reason = "solution_found"
                        conflict_history.append(conflicts_next)
                        yield (xbest.copy(), i, Tc, 0)
                        break
            elif delta_E < 0:
                prob = math.exp(delta_E / Tc) if Tc > 0 else 0
                if random.random() < prob:
                    xcurr = xnext
            else:
                if random.random() < 0.5:
                    xcurr = xnext
            
            if self.count_conflicts(xcurr) == 0:
                xbest = xcurr.copy()
                stop_reason = "solution_found"
                yield (xbest.copy(), i, Tc, 0)
                break
        
        # Check if we found a solution after loop ends
        if self.count_conflicts(xbest) == 0 and stop_reason == "max_iterations":
            stop_reason = "solution_found"
        
        self._stop_reason = stop_reason
        self._final_result = (xbest, temperature_history, conflict_history, stop_reason)
    
    def sim_anneal_simple(self, m: int, iter_max: int, T: float, min_temp: float = 2.0, cooling_rate: float = 0.95) -> Tuple[Dict[int, int], List[float], List[int], str]:
        return self._sim_anneal_core(m, iter_max, T, min_temp, cooling_rate)
