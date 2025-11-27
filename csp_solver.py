from typing import List, Dict, Tuple, Set, Generator


class GraphColoringCSP:
    def __init__(self, graph: Dict[int, List[int]], num_colors: int):
        self.graph = graph
        self.num_colors = num_colors
        self.vertices = sorted(graph.keys())
        self.colors = list(range(num_colors))
        self.color_names = [
            'Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Cyan', 'Magenta', 'Pink', 'Brown',
            'Lime', 'Teal', 'Navy', 'Maroon', 'Olive', 'Silver', 'Gold', 'Coral', 'Lavender', 'Turquoise',
            'Indigo', 'Violet', 'Beige', 'Khaki', 'Salmon', 'Crimson', 'Aqua', 'Chartreuse', 'Fuchsia', 'Gray',
            'DarkGreen', 'DarkBlue', 'DarkRed', 'DarkOrange', 'DarkViolet', 'LightBlue', 'LightGreen', 'LightPink', 'LightYellow', 'LightGray'
        ]
        self.assignments_count = 0
        self.backtracks_count = 0
    
    def is_consistent(self, vertex: int, color: int, assignment: Dict[int, int]) -> bool:
        for neighbor in self.graph[vertex]:
            if neighbor in assignment and assignment[neighbor] == color:
                return False
        return True
    
    def forward_check(self, vertex: int, color: int, domains: Dict[int, Set[int]]) -> bool:
        for neighbor in self.graph[vertex]:
            if neighbor not in domains:
                continue
            
            if color in domains[neighbor]:
                domains[neighbor].remove(color)
                
                if len(domains[neighbor]) == 0:
                    return False
        return True
    
    def select_unassigned_variable(self, assignment: Dict[int, int], domains: Dict[int, Set[int]]) -> int:
        unassigned = [v for v in self.vertices if v not in assignment]
        if not unassigned:
            return None
        
        return min(unassigned, key=lambda v: len(domains.get(v, set())))
    
    def order_domain_values(self, vertex: int, assignment: Dict[int, int], domains: Dict[int, Set[int]]) -> List[int]:
        available_colors = list(domains.get(vertex, set()))
        
        def count_eliminated_values(color):
            count = 0
            for neighbor in self.graph[vertex]:
                if neighbor not in assignment and color in domains.get(neighbor, set()):
                    count += 1
            return count
        
        return sorted(available_colors, key=count_eliminated_values)
    
    def _backtrack_core(self, assignment: Dict[int, int], domains: Dict[int, Set[int]]):
        if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
            return None
        
        if len(assignment) == len(self.vertices):
            conflicts = self.count_conflicts(assignment)
            if conflicts == 0:
                if hasattr(self, 'solution_found'):
                    self.solution_found = True
                return assignment
            else:
                return None
        
        vertex = self.select_unassigned_variable(assignment, domains)
        if vertex is None:
            return assignment
        
        ordered_colors = self.order_domain_values(vertex, assignment, domains)
        
        for color in ordered_colors:
            self.assignments_count += 1
            
            if not self.is_consistent(vertex, color, assignment):
                continue
            
            if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                return None
            
            assignment[vertex] = color
            
            new_domains = {v: domains[v].copy() for v in domains}
            
            if not self.forward_check(vertex, color, new_domains):
                del assignment[vertex]
                self.backtracks_count += 1
                continue
            
            if vertex in new_domains:
                del new_domains[vertex]
            
            if hasattr(self, 'solution_found') and self.solution_found:
                return assignment if len(assignment) == len(self.vertices) else None
            
            result = self._backtrack_core(assignment, new_domains)
            
            if result and isinstance(result, dict) and len(result) == len(self.vertices):
                if self.count_conflicts(result) == 0:
                    return result
            
            if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                return None
            
            del assignment[vertex]
            self.backtracks_count += 1
        
        return None
    
    def backtrack(self, assignment: Dict[int, int], domains: Dict[int, Set[int]], 
                  yield_progress: bool = False) -> Generator:
        if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
            return None
        
        if len(assignment) == len(self.vertices):
            conflicts = self.count_conflicts(assignment)
            if conflicts == 0:
                if hasattr(self, 'solution_found'):
                    self.solution_found = True
                if yield_progress:
                    yield (assignment.copy(), self.assignments_count, self.backtracks_count)
                return assignment
            else:
                return None
        
        vertex = self.select_unassigned_variable(assignment, domains)
        if vertex is None:
            return assignment
        
        ordered_colors = self.order_domain_values(vertex, assignment, domains)
        
        for color in ordered_colors:
            self.assignments_count += 1
            
            if not self.is_consistent(vertex, color, assignment):
                continue
            
            if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                return None
            
            assignment[vertex] = color
            
            new_domains = {v: domains[v].copy() for v in domains}
            
            if not self.forward_check(vertex, color, new_domains):
                del assignment[vertex]
                self.backtracks_count += 1
                continue
            
            if vertex in new_domains:
                del new_domains[vertex]
            
            if yield_progress:
                yield (assignment.copy(), self.assignments_count, self.backtracks_count)
            
            if hasattr(self, 'solution_found') and self.solution_found:
                return assignment if len(assignment) == len(self.vertices) else None
            
            if yield_progress:
                result = None
                for result in self.backtrack(assignment, new_domains, yield_progress):
                    if hasattr(self, 'solution_found') and self.solution_found:
                        if isinstance(result, dict) and len(result) == len(self.vertices):
                            if self.count_conflicts(result) == 0:
                                yield result
                                return result
                        elif isinstance(result, tuple) and len(result) == 3:
                            assignment_tuple, _, _ = result
                            if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                                if self.count_conflicts(assignment_tuple) == 0:
                                    yield result
                                    return assignment_tuple
                    
                    if isinstance(result, dict) and len(result) == len(self.vertices):
                        if self.count_conflicts(result) == 0:
                            yield result
                            return result
                    elif isinstance(result, tuple) and len(result) == 3:
                        assignment_tuple, _, _ = result
                        if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                            if self.count_conflicts(assignment_tuple) == 0:
                                yield result
                                return assignment_tuple
                            else:
                                yield result
                    else:
                        yield result
                    
                    if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                        return None
                
                if isinstance(result, dict) and len(result) == len(self.vertices):
                    if self.count_conflicts(result) == 0:
                        return result
                elif isinstance(result, tuple) and len(result) == 3:
                    assignment_tuple, _, _ = result
                    if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                        if self.count_conflicts(assignment_tuple) == 0:
                            return assignment_tuple
            else:
                result = None
                for result in self.backtrack(assignment, new_domains, yield_progress):
                    if hasattr(self, 'solution_found') and self.solution_found:
                        if isinstance(result, dict) and len(result) == len(self.vertices):
                            if self.count_conflicts(result) == 0:
                                return result
                    
                    if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                        return None
                    pass
                if isinstance(result, dict) and len(result) == len(self.vertices):
                    if self.count_conflicts(result) == 0:
                        return result
            
            del assignment[vertex]
            self.backtracks_count += 1
        
        return None
    
    def _solve_core(self, max_assignments: int = 10000):
        domains = {v: set(self.colors) for v in self.vertices}
        assignment = {}
        
        self.assignments_count = 0
        self.backtracks_count = 0
        self.max_assignments = max_assignments
        self.solution_found = False
        stop_reason = "solution_found"
        
        try:
            result = self._backtrack_core(assignment, domains)
        except Exception as e:
            import traceback
            print(f"Error in _backtrack_core: {e}")
            print(traceback.format_exc())
            result = None
            stop_reason = "error"
        
        if result and isinstance(result, dict) and len(result) == len(self.vertices):
            if self.count_conflicts(result) == 0:
                stop_reason = "solution_found"
            else:
                stop_reason = "no_solution"
        elif self.assignments_count >= self.max_assignments:
            stop_reason = "max_assignments"
        else:
            stop_reason = "no_solution"
        
        self._stop_reason = stop_reason
        return result, self.assignments_count, self.backtracks_count, stop_reason
    
    def _solve_with_yield(self, max_assignments: int = 10000):
        domains = {v: set(self.colors) for v in self.vertices}
        assignment = {}
        
        self.assignments_count = 0
        self.backtracks_count = 0
        self.max_assignments = max_assignments
        self.solution_found = False
        stop_reason = "solution_found"
        
        result = None
        solution_found = False
        for result in self.backtrack(assignment, domains, yield_progress=True):
            if isinstance(result, tuple):
                yield result
                if self.assignments_count >= self.max_assignments:
                    stop_reason = "max_assignments"
                    break
                if len(result) == 3:
                    assignment_tuple, _, _ = result
                    if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                        solution_found = True
                        stop_reason = "solution_found"
                        self._final_result = (assignment_tuple, self.assignments_count, self.backtracks_count, stop_reason)
                        self._stop_reason = stop_reason
                        yield (assignment_tuple, self.assignments_count, self.backtracks_count)
                        return
            elif isinstance(result, dict) and len(result) == len(self.vertices):
                solution_found = True
                stop_reason = "solution_found"
                yield (result, self.assignments_count, self.backtracks_count)
                self._final_result = (result, self.assignments_count, self.backtracks_count, stop_reason)
                self._stop_reason = stop_reason
                return
        
        if not solution_found:
            if self.assignments_count >= self.max_assignments:
                stop_reason = "max_assignments"
            else:
                stop_reason = "no_solution"
            final_solution = result
            if isinstance(result, tuple) and len(result) >= 1:
                if isinstance(result[0], dict):
                    final_solution = result[0]
                elif len(result) == 3 and isinstance(result[0], dict):
                    final_solution = result[0]
            elif not isinstance(result, dict):
                final_solution = None
            self._final_result = (final_solution, self.assignments_count, self.backtracks_count, stop_reason)
            self._stop_reason = stop_reason
    
    def solve(self, yield_progress: bool = False, max_assignments: int = 10000):
        if yield_progress:
            return self._solve_with_yield(max_assignments)
        else:
            return self._solve_core(max_assignments)
    
    def count_conflicts(self, coloring: Dict[int, int]) -> int:
        conflicts = 0
        for vertex, neighbors in self.graph.items():
            for neighbor in neighbors:
                if vertex in coloring and neighbor in coloring:
                    if coloring[vertex] == coloring[neighbor]:
                        conflicts += 1
        return conflicts // 2
