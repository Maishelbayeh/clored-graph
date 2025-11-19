"""
Constraint Satisfaction Problem (CSP) solver for Graph Coloring using Backtracking with Forward Checking.
"""
from typing import List, Dict, Tuple, Set, Generator


class GraphColoringCSP:
    """
    Constraint Satisfaction Problem solver for Graph Coloring using Backtracking with Forward Checking.
    """
    def __init__(self, graph: Dict[int, List[int]], num_colors: int):
        """
        Initialize the CSP solver.
        
        Args:
            graph: Dictionary representing adjacency list {vertex: [neighbors]}
            num_colors: Number of colors available
        """
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
        """
        Check if assigning color to vertex is consistent with current assignment.
        
        Args:
            vertex: Vertex to check
            color: Color to assign
            assignment: Current partial assignment
        
        Returns:
            True if assignment is consistent (no conflicts with neighbors)
        """
        for neighbor in self.graph[vertex]:
            if neighbor in assignment and assignment[neighbor] == color:
                return False
        return True
    
    def forward_check(self, vertex: int, color: int, domains: Dict[int, Set[int]]) -> bool:
        """
        Perform forward checking: remove inconsistent values from neighbors' domains.
        
        Args:
            vertex: Vertex that was just assigned
            color: Color assigned to vertex
            domains: Dictionary of remaining domains for each vertex
        
        Returns:
            True if no domain became empty, False otherwise
        """
        for neighbor in self.graph[vertex]:
            if neighbor not in domains:
                continue  # Already assigned
            
            # Remove the conflicting color from neighbor's domain
            if color in domains[neighbor]:
                domains[neighbor].remove(color)
                
                # If neighbor's domain becomes empty, assignment fails
                if len(domains[neighbor]) == 0:
                    return False
        return True
    
    def select_unassigned_variable(self, assignment: Dict[int, int], domains: Dict[int, Set[int]]) -> int:
        """
        Select next unassigned variable using MRV (Minimum Remaining Values) heuristic.
        
        Args:
            assignment: Current assignment
            domains: Current domains
        
        Returns:
            Next vertex to assign
        """
        unassigned = [v for v in self.vertices if v not in assignment]
        if not unassigned:
            return None
        
        # MRV: choose variable with fewest remaining values
        return min(unassigned, key=lambda v: len(domains.get(v, set())))
    
    def order_domain_values(self, vertex: int, assignment: Dict[int, int], domains: Dict[int, Set[int]]) -> List[int]:
        """
        Order domain values using LCV (Least Constraining Value) heuristic.
        
        Args:
            vertex: Vertex to assign
            assignment: Current assignment
            domains: Current domains
        
        Returns:
            Ordered list of colors to try
        """
        available_colors = list(domains.get(vertex, set()))
        
        def count_eliminated_values(color):
            """Count how many values would be eliminated from neighbors' domains."""
            count = 0
            for neighbor in self.graph[vertex]:
                if neighbor not in assignment and color in domains.get(neighbor, set()):
                    count += 1
            return count
        
        # Sort by least constraining (fewest eliminations)
        return sorted(available_colors, key=count_eliminated_values)
    
    def _backtrack_core(self, assignment: Dict[int, int], domains: Dict[int, Set[int]]):
        """
        Core backtracking search without yield (internal method).
        """
        # Check if we've exceeded max assignments
        if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
            return None
        
        # If all variables are assigned, we found a solution
        if len(assignment) == len(self.vertices):
            # Check if this is a valid solution (0 conflicts)
            conflicts = self.count_conflicts(assignment)
            if conflicts == 0:
                # Perfect solution found - stop immediately
                if hasattr(self, 'solution_found'):
                    self.solution_found = True
                return assignment
            else:
                # Solution has conflicts (shouldn't happen with forward checking, but handle it)
                return None
        
        # Select next variable using MRV
        vertex = self.select_unassigned_variable(assignment, domains)
        if vertex is None:
            return assignment
        
        # Order domain values using LCV
        ordered_colors = self.order_domain_values(vertex, assignment, domains)
        
        for color in ordered_colors:
            self.assignments_count += 1
            
            # Check consistency
            if not self.is_consistent(vertex, color, assignment):
                continue
            
            # Check max assignments limit
            if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                return None
            
            # Make assignment
            assignment[vertex] = color
            
            # Create copy of domains for forward checking
            new_domains = {v: domains[v].copy() for v in domains}
            
            # Forward checking
            if not self.forward_check(vertex, color, new_domains):
                # Assignment failed, backtrack
                del assignment[vertex]
                self.backtracks_count += 1
                continue
            
            # Remove assigned variable from domains
            if vertex in new_domains:
                del new_domains[vertex]
            
            # Check if solution already found (stop recursion)
            if hasattr(self, 'solution_found') and self.solution_found:
                return assignment if len(assignment) == len(self.vertices) else None
            
            # Recursive call
            result = self._backtrack_core(assignment, new_domains)
            
            # Check if solution found
            if result and isinstance(result, dict) and len(result) == len(self.vertices):
                # Verify it's a valid solution (0 conflicts)
                if self.count_conflicts(result) == 0:
                    return result  # Solution found, no backtrack needed
            
            # Check max assignments
            if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                return None
            
            # Backtrack: remove assignment (only if no solution found)
            # This means we tried this color and it didn't lead to a solution
            del assignment[vertex]
            self.backtracks_count += 1
        
        # No solution found with current path
        return None
    
    def backtrack(self, assignment: Dict[int, int], domains: Dict[int, Set[int]], 
                  yield_progress: bool = False) -> Generator:
        """
        Backtracking search with forward checking.
        
        Args:
            assignment: Current partial assignment
            domains: Current domains for each variable
            yield_progress: If True, yields intermediate states
        
        Yields:
            (assignment, assignments_count, backtracks_count) if yield_progress
        """
        # Check if we've exceeded max assignments
        if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
            return None
        
        # If all variables are assigned, we found a solution
        if len(assignment) == len(self.vertices):
            # Check if this is a valid solution (0 conflicts)
            conflicts = self.count_conflicts(assignment)
            if conflicts == 0:
                # Perfect solution found - stop immediately
                if hasattr(self, 'solution_found'):
                    self.solution_found = True
                if yield_progress:
                    yield (assignment.copy(), self.assignments_count, self.backtracks_count)
                return assignment
            else:
                # Solution has conflicts (shouldn't happen with forward checking, but handle it)
                # Continue searching - this means we need to backtrack
                return None
        
        # Select next variable using MRV
        vertex = self.select_unassigned_variable(assignment, domains)
        if vertex is None:
            return assignment
        
        # Order domain values using LCV
        ordered_colors = self.order_domain_values(vertex, assignment, domains)
        
        for color in ordered_colors:
            self.assignments_count += 1
            
            # Check consistency
            if not self.is_consistent(vertex, color, assignment):
                continue
            
            # Check max assignments limit
            if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                return None
            
            # Make assignment
            assignment[vertex] = color
            
            # Create copy of domains for forward checking
            new_domains = {v: domains[v].copy() for v in domains}
            
            # Forward checking
            if not self.forward_check(vertex, color, new_domains):
                # Assignment failed, backtrack
                del assignment[vertex]
                self.backtracks_count += 1
                continue
            
            # Remove assigned variable from domains
            if vertex in new_domains:
                del new_domains[vertex]
            
            # Yield progress if requested
            if yield_progress:
                yield (assignment.copy(), self.assignments_count, self.backtracks_count)
            
            # Check if solution already found (stop recursion)
            if hasattr(self, 'solution_found') and self.solution_found:
                return assignment if len(assignment) == len(self.vertices) else None
            
            # Recursive call
            if yield_progress:
                result = None
                for result in self.backtrack(assignment, new_domains, yield_progress):
                    # Check if solution found flag is set
                    if hasattr(self, 'solution_found') and self.solution_found:
                        # Solution found - return immediately
                        if isinstance(result, dict) and len(result) == len(self.vertices):
                            # Verify it's a valid solution (0 conflicts)
                            if self.count_conflicts(result) == 0:
                                yield result
                                return result
                        elif isinstance(result, tuple) and len(result) == 3:
                            assignment_tuple, _, _ = result
                            if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                                # Verify it's a valid solution (0 conflicts)
                                if self.count_conflicts(assignment_tuple) == 0:
                                    yield result
                                    return assignment_tuple
                    
                    # Check if this is a complete solution
                    if isinstance(result, dict) and len(result) == len(self.vertices):
                        # Verify it's a valid solution (0 conflicts)
                        if self.count_conflicts(result) == 0:
                            # Found perfect solution - stop immediately
                            yield result
                            return result
                    # Check if this is a progress tuple with complete assignment
                    elif isinstance(result, tuple) and len(result) == 3:
                        assignment_tuple, _, _ = result
                        if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                            # Verify it's a valid solution (0 conflicts)
                            if self.count_conflicts(assignment_tuple) == 0:
                                # Found perfect solution in tuple - stop immediately
                                yield result
                                return assignment_tuple
                            else:
                                # Regular progress update
                                yield result
                    else:
                        # Regular progress update
                        yield result
                    
                    # Check max assignments
                    if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                        return None
                
                # If we got a complete solution, verify and return it
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
                    # Check if solution found
                    if hasattr(self, 'solution_found') and self.solution_found:
                        if isinstance(result, dict) and len(result) == len(self.vertices):
                            # Verify it's a valid solution (0 conflicts)
                            if self.count_conflicts(result) == 0:
                                return result
                    
                    # Check max assignments
                    if hasattr(self, 'max_assignments') and self.assignments_count >= self.max_assignments:
                        return None
                    pass
                # If we got a complete solution, verify and return it
                if isinstance(result, dict) and len(result) == len(self.vertices):
                    if self.count_conflicts(result) == 0:
                        return result  # Solution found, no backtrack needed
            
            # Backtrack: remove assignment (only if no solution found)
            # This means we tried this color and it didn't lead to a solution
            del assignment[vertex]
            self.backtracks_count += 1
        
        # No solution found with current path
        return None
    
    def _solve_core(self, max_assignments: int = 10000):
        """
        Core solve method without yield (internal method).
        Returns: (solution, assignments_count, backtracks_count, stop_reason)
        """
        # Initialize domains: all vertices can have all colors
        domains = {v: set(self.colors) for v in self.vertices}
        assignment = {}
        
        self.assignments_count = 0
        self.backtracks_count = 0
        self.max_assignments = max_assignments
        self.solution_found = False  # Flag to stop when solution is found
        stop_reason = "solution_found"  # Default
        
        # Use core backtrack method without yield
        try:
            result = self._backtrack_core(assignment, domains)
        except Exception as e:
            # If any error occurs, return error state
            import traceback
            print(f"Error in _backtrack_core: {e}")
            print(traceback.format_exc())
            result = None
            stop_reason = "error"
        
        # Determine stop reason
        if result and isinstance(result, dict) and len(result) == len(self.vertices):
            # Verify it's a valid solution (0 conflicts)
            if self.count_conflicts(result) == 0:
                stop_reason = "solution_found"
            else:
                stop_reason = "no_solution"
        elif self.assignments_count >= self.max_assignments:
            stop_reason = "max_assignments"
        else:
            stop_reason = "no_solution"
        
        self._stop_reason = stop_reason
        # Always return 4 values
        return result, self.assignments_count, self.backtracks_count, stop_reason
    
    def _solve_with_yield(self, max_assignments: int = 10000):
        """
        Solve with yield (generator version, internal method).
        """
        # Initialize domains: all vertices can have all colors
        domains = {v: set(self.colors) for v in self.vertices}
        assignment = {}
        
        self.assignments_count = 0
        self.backtracks_count = 0
        self.max_assignments = max_assignments
        self.solution_found = False  # Flag to stop when solution is found
        stop_reason = "solution_found"  # Default
        
        result = None
        solution_found = False
        for result in self.backtrack(assignment, domains, yield_progress=True):
            if isinstance(result, tuple):
                # This is a progress update tuple
                yield result
                # Check if we've exceeded max assignments
                if self.assignments_count >= self.max_assignments:
                    stop_reason = "max_assignments"
                    break
                # Check if the assignment in the tuple is complete
                if len(result) == 3:
                    assignment_tuple, _, _ = result
                    if isinstance(assignment_tuple, dict) and len(assignment_tuple) == len(self.vertices):
                        # Complete solution found!
                        solution_found = True
                        stop_reason = "solution_found"
                        self._final_result = (assignment_tuple, self.assignments_count, self.backtracks_count, stop_reason)
                        self._stop_reason = stop_reason
                        # Yield final result and stop
                        yield (assignment_tuple, self.assignments_count, self.backtracks_count)
                        return
            elif isinstance(result, dict) and len(result) == len(self.vertices):
                # Complete solution found directly
                solution_found = True
                stop_reason = "solution_found"
                yield (result, self.assignments_count, self.backtracks_count)
                self._final_result = (result, self.assignments_count, self.backtracks_count, stop_reason)
                self._stop_reason = stop_reason
                return
        
        # Store final result if we didn't return earlier
        if not solution_found:
            if self.assignments_count >= self.max_assignments:
                stop_reason = "max_assignments"
            else:
                stop_reason = "no_solution"
            # Extract dict from result if it's a tuple
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
        """
        Solve the graph coloring problem using CSP.
        
        Args:
            yield_progress: If True, yields intermediate states
            max_assignments: Maximum number of assignments before stopping
        
        Yields (if yield_progress=True):
            (assignment, assignments_count, backtracks_count)
        
        Returns:
            If yield_progress=False: Tuple of (solution, assignments_count, backtracks_count, stop_reason)
            If yield_progress=True: Generator that yields progress, final result stored in instance
        """
        if yield_progress:
            # Use generator version
            return self._solve_with_yield(max_assignments)
        else:
            # Use core method that doesn't yield
            return self._solve_core(max_assignments)
    
    def count_conflicts(self, coloring: Dict[int, int]) -> int:
        """
        Count the number of conflicts (adjacent vertices with same color).
        
        Args:
            coloring: Coloring configuration
        
        Returns:
            Number of conflicts
        """
        conflicts = 0
        for vertex, neighbors in self.graph.items():
            for neighbor in neighbors:
                if vertex in coloring and neighbor in coloring:
                    if coloring[vertex] == coloring[neighbor]:
                        conflicts += 1
        return conflicts // 2

