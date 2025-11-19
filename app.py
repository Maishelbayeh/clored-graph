"""
Streamlit UI for Graph Coloring: Simulated Annealing vs CSP Comparison
"""
import streamlit as st
import matplotlib.pyplot as plt
import time

# Import solvers and utilities
from sa_solver import GraphColoringSA
from csp_solver import GraphColoringCSP
from graph_utils import create_graph_from_edges, generate_random_graph, visualize_graph

# Page configuration
st.set_page_config(page_title="Graph Coloring: SA vs CSP Comparison", layout="wide")


def _run_sa(graph, num_colors, initial_temp, min_temp, max_iterations, cooling_rate,
           show_animation, animation_speed, live_viz_placeholder, status_placeholder):
    """Run Simulated Annealing algorithm."""
    solver = GraphColoringSA(graph, num_colors)
    start_time = time.time()
    
    if show_animation:
        # Run with live visualization
        status_placeholder.info("🔄 Running Simulated Annealing with live visualization...")
        
        best_coloring = None
        temp_history = []
        conflict_history = []
        best_conflicts = float('inf')
        
        # Create initial visualization
        initial_coloring = solver.initial_config(len(graph))
        fig = visualize_graph(graph, initial_coloring, solver.color_names)
        live_viz_placeholder.pyplot(fig)
        
        # Run algorithm with progress updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generator = solver.sim_anneal(len(graph), max_iterations, initial_temp, yield_progress=True, min_temp=min_temp, cooling_rate=cooling_rate)
        
        try:
            iteration = 0
            last_yield = None
            stop_reason = "max_iterations"
            for current_coloring, iter_num, temp, conflicts in generator:
                iteration = iter_num
                last_yield = (current_coloring, iter_num, temp, conflicts)
                
                temp_history.append(temp)
                conflict_history.append(conflicts)
                
                # Check if temperature threshold reached
                if temp <= min_temp:
                    stop_reason = "temperature_threshold"
                
                # Update best solution
                if conflicts < best_conflicts:
                    best_coloring = current_coloring.copy()
                    best_conflicts = conflicts
                
                # Update visualization periodically
                if iteration % animation_speed == 0 or iteration == 1:
                    fig = visualize_graph(graph, current_coloring, solver.color_names)
                    live_viz_placeholder.pyplot(fig)
                    
                    # Update status with stop reason indicator
                    stop_indicator = "🌡️ Temp threshold" if temp <= min_temp else ""
                    status_text.text(f"🔄 Iteration: {iteration}/{max_iterations} | "
                                   f"Conflicts: {conflicts} | "
                                   f"Temperature: {temp:.2f} {stop_indicator}")
                    
                    # Update progress bar
                    progress_bar.progress(min(iteration / max_iterations, 1.0))
                    
                    # Small delay for visualization
                    time.sleep(0.01)
            
            # After generator completes, get final result from solver
            if hasattr(solver, '_final_result'):
                final_best, final_temp, final_conf, final_stop_reason = solver._final_result
                stop_reason = final_stop_reason
                # Use the tracked best from our loop, but ensure we have complete history
                if len(temp_history) == max_iterations and len(conflict_history) == max_iterations:
                    temp_history = final_temp
                    conflict_history = final_conf
                # best_coloring already tracks the best we found
                if best_coloring is None:
                    best_coloring = final_best
            elif hasattr(solver, '_stop_reason'):
                stop_reason = solver._stop_reason
        
        except Exception as e:
            st.error(f"Error during execution: {e}")
            import traceback
            st.code(traceback.format_exc())
            # Fallback to simple version
            best_coloring, temp_history, conflict_history, stop_reason = solver.sim_anneal_simple(
                len(graph), max_iterations, initial_temp, min_temp, cooling_rate
            )
        
        progress_bar.progress(1.0)
        
        # Show final best solution in visualization
        if best_coloring:
            fig = visualize_graph(graph, best_coloring, solver.color_names)
            live_viz_placeholder.pyplot(fig)
        
        # Show completion message with stop reason
        elapsed_time = time.time() - start_time
        if stop_reason == "temperature_threshold":
            status_placeholder.success(f"✅ Simulated Annealing completed! (Stopped at temperature ≤ {min_temp}) - Time: {elapsed_time:.2f}s")
        else:
            status_placeholder.success(f"✅ Simulated Annealing completed! (Reached max iterations) - Time: {elapsed_time:.2f}s")
        status_text.empty()
        
    else:
        # Run without animation (faster)
        with st.spinner("Running Simulated Annealing..."):
            best_coloring, temp_history, conflict_history, stop_reason = solver.sim_anneal_simple(
                len(graph), max_iterations, initial_temp, min_temp, cooling_rate
            )
            elapsed_time = time.time() - start_time
            if stop_reason == "temperature_threshold":
                st.success(f"✅ Simulated Annealing completed! (Stopped at temperature ≤ {min_temp}) - Time: {elapsed_time:.2f}s")
            else:
                st.success(f"✅ Simulated Annealing completed! (Reached max iterations) - Time: {elapsed_time:.2f}s")
    
    # Store results in session state
    st.session_state['best_coloring'] = best_coloring
    st.session_state['temp_history'] = temp_history
    st.session_state['conflict_history'] = conflict_history
    st.session_state['solver'] = solver
    st.session_state['graph'] = graph
    st.session_state['algorithm'] = 'SA'
    st.session_state['execution_time'] = elapsed_time
    st.session_state['stop_reason'] = stop_reason if 'stop_reason' in locals() else "max_iterations"


def _run_csp(graph, num_colors, max_assignments, show_animation, live_viz_placeholder, status_placeholder):
    """Run CSP (Backtracking) algorithm."""
    solver = GraphColoringCSP(graph, num_colors)
    start_time = time.time()
    
    if show_animation:
        status_placeholder.info("🔄 Running CSP with live visualization...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Empty initial visualization
        fig = visualize_graph(graph, None, None)
        live_viz_placeholder.pyplot(fig)
        
        try:
            generator = solver.solve(yield_progress=True, max_assignments=max_assignments)
            last_assignment = None
            
            for result in generator:
                if isinstance(result, tuple) and len(result) == 3:
                    assignment, assignments_count, backtracks_count = result
                    last_assignment = assignment
                    
                    # Check if solution is complete
                    if assignment and len(assignment) == len(graph):
                        # Complete solution found - stop immediately
                        fig = visualize_graph(graph, assignment, solver.color_names)
                        live_viz_placeholder.pyplot(fig)
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ Solution found! Assigned: {len(assignment)}/{len(graph)} | "
                                       f"Assignments: {assignments_count} | "
                                       f"Backtracks: {backtracks_count}")
                        break  # Stop the loop immediately
                    
                    # Update visualization
                    if assignment and len(assignment) > 0:
                        fig = visualize_graph(graph, assignment, solver.color_names)
                        live_viz_placeholder.pyplot(fig)
                        
                        # Update status
                        progress = len(assignment) / len(graph) if graph else 0
                        stop_indicator = "⚠️ Max assignments" if assignments_count >= max_assignments else ""
                        status_text.text(f"🔄 Assigned: {len(assignment)}/{len(graph)} | "
                                       f"Assignments: {assignments_count}/{max_assignments} | "
                                       f"Backtracks: {backtracks_count} {stop_indicator}")
                        progress_bar.progress(progress)
                        time.sleep(0.05)
            
            # Get final result
            if hasattr(solver, '_final_result'):
                if len(solver._final_result) == 4:
                    solution, assignments_count, backtracks_count, stop_reason = solver._final_result
                else:
                    solution, assignments_count, backtracks_count = solver._final_result
                    stop_reason = solver._stop_reason if hasattr(solver, '_stop_reason') else "unknown"
            elif last_assignment and len(last_assignment) == len(graph):
                solution = last_assignment
                assignments_count = solver.assignments_count
                backtracks_count = solver.backtracks_count
                stop_reason = "solution_found"
            else:
                # Run again without animation to get final result
                solution, assignments_count, backtracks_count, stop_reason = solver.solve(yield_progress=False, max_assignments=max_assignments)
            
            progress_bar.progress(1.0)
            
            # Show final solution in visualization
            # Ensure solution is a dict, not a tuple
            if solution:
                if isinstance(solution, tuple):
                    # If solution is a tuple, extract the dict from it
                    if len(solution) >= 1 and isinstance(solution[0], dict):
                        solution = solution[0]
                    else:
                        # Use last_assignment if available
                        solution = last_assignment if last_assignment else None
                if solution and isinstance(solution, dict):
                    fig = visualize_graph(graph, solution, solver.color_names)
                    live_viz_placeholder.pyplot(fig)
            
            elapsed_time = time.time() - start_time
            
            # Show completion message with stop reason
            if stop_reason == "solution_found":
                conflicts = solver.count_conflicts(solution) if solution else 0
                status_placeholder.success(f"✅ CSP completed! Solution found - Conflicts: {conflicts} - Time: {elapsed_time:.2f}s")
            elif stop_reason == "max_assignments":
                status_placeholder.warning(f"⚠️ CSP stopped! Reached max assignments ({max_assignments}) - Time: {elapsed_time:.2f}s")
            else:
                status_placeholder.warning(f"⚠️ CSP completed! No solution found - Time: {elapsed_time:.2f}s")
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error during execution: {e}")
            import traceback
            st.code(traceback.format_exc())
            solution, assignments_count, backtracks_count, stop_reason = solver.solve(yield_progress=False, max_assignments=max_assignments)
            elapsed_time = time.time() - start_time
    else:
        with st.spinner("Running CSP..."):
            solution, assignments_count, backtracks_count, stop_reason = solver.solve(yield_progress=False, max_assignments=max_assignments)
            elapsed_time = time.time() - start_time
            
            if stop_reason == "solution_found":
                conflicts = solver.count_conflicts(solution) if solution else 0
                st.success(f"✅ CSP completed! Solution found - Conflicts: {conflicts} - Time: {elapsed_time:.2f}s")
            elif stop_reason == "max_assignments":
                st.warning(f"⚠️ CSP stopped! Reached max assignments ({max_assignments}) - Time: {elapsed_time:.2f}s")
            else:
                st.warning(f"⚠️ CSP completed! No solution found - Time: {elapsed_time:.2f}s")
    
    # Store results in session state
    # Ensure solution is a dict, not a tuple
    final_solution = solution
    if solution and isinstance(solution, tuple):
        if len(solution) >= 1 and isinstance(solution[0], dict):
            final_solution = solution[0]
        else:
            final_solution = None
    elif solution and not isinstance(solution, dict):
        final_solution = None
    
    st.session_state['best_coloring'] = final_solution
    st.session_state['solver'] = solver
    st.session_state['graph'] = graph
    st.session_state['algorithm'] = 'CSP'
    st.session_state['execution_time'] = elapsed_time
    st.session_state['assignments_count'] = assignments_count
    st.session_state['backtracks_count'] = backtracks_count
    st.session_state['stop_reason'] = stop_reason if 'stop_reason' in locals() else "unknown"


def _compare_algorithms(graph, num_colors, initial_temp, min_temp, max_iterations, max_assignments, cooling_rate,
                       show_animation, animation_speed, live_viz_placeholder, status_placeholder):
    """Compare both SA and CSP algorithms."""
    status_placeholder.info("🔄 Running both algorithms for comparison...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulated Annealing")
        try:
            sa_start = time.time()
            sa_solver = GraphColoringSA(graph, num_colors)
            result = sa_solver.sim_anneal_simple(
                len(graph), max_iterations, initial_temp, min_temp, cooling_rate
            )
            # Ensure we got 4 values
            if result is None:
                raise ValueError("sim_anneal_simple returned None")
            if not isinstance(result, tuple):
                raise ValueError(f"sim_anneal_simple returned {type(result)}, expected tuple")
            if len(result) != 4:
                raise ValueError(f"Expected 4 values from sim_anneal_simple, got {len(result)}")
            sa_result, sa_temp, sa_conflicts, sa_stop = result
            sa_time = time.time() - sa_start
            sa_conflicts_final = sa_solver.count_conflicts(sa_result) if sa_result else 0
            
            st.metric("Execution Time", f"{sa_time:.3f}s")
            st.metric("Final Conflicts", sa_conflicts_final)
            st.metric("Iterations", len(sa_conflicts) if sa_conflicts else 0)
            if sa_stop == "temperature_threshold":
                st.metric("Stop Reason", "🌡️ Temperature")
            elif sa_stop == "solution_found":
                st.metric("Stop Reason", "✅ Solution Found")
            else:
                st.metric("Stop Reason", "🔄 Max Iterations")
        except Exception as e:
            st.error(f"Error in SA: {e}")
            import traceback
            st.code(traceback.format_exc())
            sa_result = None
            sa_time = 0
            sa_conflicts_final = 0
            sa_stop = "error"
    
    with col2:
        st.subheader("CSP (Backtracking)")
        csp_result = None
        csp_time = 0
        csp_conflicts = 0
        csp_assignments = 0
        csp_backtracks = 0
        csp_stop = "error"
        csp_solver = None
        
        try:
            csp_start = time.time()
            csp_solver = GraphColoringCSP(graph, num_colors)
            result = csp_solver.solve(yield_progress=False, max_assignments=max_assignments)
            # Ensure we got 4 values
            if result is None:
                raise ValueError("solve returned None")
            if not isinstance(result, tuple):
                raise ValueError(f"solve returned {type(result)}, expected tuple")
            if len(result) != 4:
                raise ValueError(f"Expected 4 values from solve, got {len(result)}")
            csp_result, csp_assignments, csp_backtracks, csp_stop = result
            csp_time = time.time() - csp_start
            
            if csp_result and isinstance(csp_result, dict):
                csp_conflicts = csp_solver.count_conflicts(csp_result)
                st.metric("Execution Time", f"{csp_time:.3f}s")
                st.metric("Final Conflicts", csp_conflicts)
                st.metric("Assignments", csp_assignments)
                st.metric("Backtracks", csp_backtracks)
            else:
                st.metric("Execution Time", f"{csp_time:.3f}s")
                st.metric("Solution", "❌ Not Found")
                st.metric("Assignments", csp_assignments)
                st.metric("Backtracks", csp_backtracks)
            
            # Show stop reason
            if csp_stop == "solution_found":
                st.metric("Stop Reason", "✅ Solution Found")
            elif csp_stop == "max_assignments":
                st.metric("Stop Reason", "⚠️ Max Assignments")
            else:
                st.metric("Stop Reason", "❌ No Solution")
        except Exception as e:
            st.error(f"Error in CSP: {e}")
            import traceback
            st.code(traceback.format_exc())
            csp_result = None
            csp_time = 0
            csp_conflicts = 0
            csp_assignments = 0
            csp_backtracks = 0
            csp_stop = "error"
    
    # Comparison summary
    st.markdown("---")
    st.subheader("📊 Comparison Summary")
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        if sa_result and csp_result:
            if sa_conflicts_final == 0 and csp_conflicts == 0:
                st.success("✅ Both found optimal solutions!")
            elif sa_conflicts_final < csp_conflicts:
                st.info("🏆 SA found better solution")
            elif csp_conflicts < sa_conflicts_final:
                st.info("🏆 CSP found better solution")
            else:
                st.info("🤝 Both found same quality solution")
        elif csp_result:
            st.warning("⚠️ SA had an error")
        elif sa_result:
            st.warning("⚠️ CSP found no solution")
        else:
            st.error("❌ Both algorithms had errors")
    
    with comp_col2:
        if sa_time > 0 and csp_time > 0:
            if sa_time < csp_time:
                st.info(f"⚡ SA was {csp_time/sa_time:.2f}x faster")
            else:
                st.info(f"⚡ CSP was {sa_time/csp_time:.2f}x faster")
        else:
            st.warning("⚠️ Cannot compare execution times")
    
    with comp_col3:
        sa_colors = len(set(sa_result.values())) if sa_result and isinstance(sa_result, dict) else 'N/A'
        csp_colors = len(set(csp_result.values())) if csp_result and isinstance(csp_result, dict) else 'N/A'
        st.info(f"🎨 Colors used: SA={sa_colors}, CSP={csp_colors}")
    
    # Store both results
    st.session_state['sa_result'] = sa_result if 'sa_result' in locals() else None
    st.session_state['csp_result'] = csp_result if 'csp_result' in locals() else None
    st.session_state['sa_solver'] = sa_solver if 'sa_solver' in locals() else None
    st.session_state['csp_solver'] = csp_solver if 'csp_solver' in locals() else None
    st.session_state['graph'] = graph
    st.session_state['algorithm'] = 'COMPARE'
    st.session_state['sa_time'] = sa_time if 'sa_time' in locals() else 0
    st.session_state['csp_time'] = csp_time if 'csp_time' in locals() else 0
    # Store CSP metrics from comparison (variables are defined before try block, so always available)
    st.session_state['assignments_count'] = csp_assignments
    st.session_state['backtracks_count'] = csp_backtracks
    
    status_placeholder.success("✅ Comparison completed!")


def main():
    st.title("🎨 Graph Coloring: Simulated Annealing vs CSP")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Algorithm selection
    st.sidebar.subheader("0. Algorithm Selection")
    algorithm_mode = st.sidebar.radio(
        "Choose Algorithm",
        ["Simulated Annealing (SA)", "CSP (Backtracking)", "Compare Both"]
    )
    
    # Graph creation section
    st.sidebar.subheader("1. Create Graph")
    graph_option = st.sidebar.radio(
        "Graph Input Method",
        ["Random Graph (Specify Nodes)", "Manual Edge Entry", "Example Graphs"]
    )
    
    graph = {}
    
    if graph_option == "Random Graph (Specify Nodes)":
        num_nodes = st.sidebar.slider("Number of Nodes", min_value=3, max_value=20, value=8, step=1)
        edge_prob = st.sidebar.slider("Edge Probability", min_value=0.1, max_value=0.9, value=0.4, step=0.1)
        
        if st.sidebar.button("Generate Random Graph", type="primary"):
            graph = generate_random_graph(num_nodes, edge_prob)
            st.session_state['current_graph'] = graph
            st.sidebar.success(f"Generated graph with {num_nodes} nodes")
        
        # Load graph from session state if available
        if 'current_graph' in st.session_state:
            graph = st.session_state['current_graph']
    
    elif graph_option == "Manual Edge Entry":
        st.sidebar.markdown("**Enter edges (format: vertex1,vertex2)**")
        edge_input = st.sidebar.text_area(
            "Edges (one per line)",
            value="0,1\n1,2\n2,3\n3,0\n0,2",
            height=150
        )
        
        try:
            edges = []
            for line in edge_input.strip().split('\n'):
                if line.strip():
                    v1, v2 = map(int, line.strip().split(','))
                    edges.append((v1, v2))
            graph = create_graph_from_edges(edges)
            st.session_state['current_graph'] = graph
            st.sidebar.success(f"Graph created with {len(graph)} vertices")
        except Exception as e:
            st.sidebar.error(f"Error parsing edges: {e}")
            graph = {}
    
    else:  # Example Graphs
        example = st.sidebar.selectbox(
            "Choose Example Graph",
            ["Complete Graph K4", "Cycle Graph C5", "Wheel Graph W5", "Bipartite Graph"]
        )
        
        if example == "Complete Graph K4":
            graph = create_graph_from_edges([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
        elif example == "Cycle Graph C5":
            graph = create_graph_from_edges([(0,1), (1,2), (2,3), (3,4), (4,0)])
        elif example == "Wheel Graph W5":
            graph = create_graph_from_edges([(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)])
        elif example == "Bipartite Graph":
            graph = create_graph_from_edges([(0,3), (0,4), (1,3), (1,4), (2,3), (2,4)])
        
        # Store in session state
        if graph:
            st.session_state['current_graph'] = graph
    
    # SA Parameters
    st.sidebar.subheader("2. Simulated Annealing Parameters")
    num_colors = st.sidebar.slider("Number of Colors", min_value=2, max_value=20, value=3)
    initial_temp = st.sidebar.slider("Initial Temperature", min_value=1.0, max_value=2000.0, value=1000.0, step=10.0)
    min_temp = st.sidebar.slider("Minimum Temperature (Stop Threshold)", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
                                  help="Algorithm stops when temperature reaches this value")
    cooling_rate = st.sidebar.slider("Cooling Rate", min_value=0.85, max_value=0.99, value=0.95, step=0.01,
                                      help="Rate at which temperature decreases. Lower values cool faster (0.85-0.99)")
    max_iterations = st.sidebar.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
    
    # CSP Parameters
    st.sidebar.subheader("3. CSP Parameters")
    max_assignments = st.sidebar.slider("Max Assignments (CSP Stop Threshold)", min_value=100, max_value=50000, value=10000, step=100,
                                         help="CSP stops when reaching this number of assignments")
    
    # Display options
    st.sidebar.subheader("4. Display Options")
    show_animation = st.sidebar.checkbox("Show Live Animation (Colors Changing)", value=True)
    animation_speed = st.sidebar.slider("Animation Update Frequency", min_value=1, max_value=100, value=10, 
                                         help="Update every N iterations (lower = faster updates)")
    show_progress = st.sidebar.checkbox("Show Progress Charts", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Graph Visualization")
        if graph:
            fig = visualize_graph(graph)
            st.pyplot(fig)
            st.info(f"**Graph Info:** {len(graph)} vertices, {sum(len(neighbors) for neighbors in graph.values()) // 2} edges")
        else:
            st.warning("Please create a graph first")
    
    with col2:
        # Show appropriate control based on algorithm mode
        if algorithm_mode == "Simulated Annealing (SA)":
            st.subheader("Simulated Annealing Control")
            button_text = "🚀 Run Simulated Annealing"
        elif algorithm_mode == "CSP (Backtracking)":
            st.subheader("CSP (Backtracking) Control")
            button_text = "🚀 Run CSP Solver"
        else:  # Compare Both
            st.subheader("Algorithm Comparison")
            button_text = "🚀 Run Both Algorithms"
        
        # Placeholder for live visualization
        live_viz_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if graph and st.button(button_text, type="primary"):
            # Run based on selected algorithm mode
            if algorithm_mode == "Simulated Annealing (SA)":
                _run_sa(graph, num_colors, initial_temp, min_temp, max_iterations, cooling_rate,
                       show_animation, animation_speed, live_viz_placeholder, status_placeholder)
            elif algorithm_mode == "CSP (Backtracking)":
                _run_csp(graph, num_colors, max_assignments, show_animation, live_viz_placeholder, status_placeholder)
            else:  # Compare Both
                _compare_algorithms(graph, num_colors, initial_temp, min_temp, max_iterations, max_assignments, cooling_rate,
                                  show_animation, animation_speed, live_viz_placeholder, status_placeholder)
    
    # Results section
    algorithm = st.session_state.get('algorithm', 'SA')
    
    if algorithm == 'COMPARE' and 'sa_result' in st.session_state:
        # Show comparison results
        st.markdown("---")
        st.subheader("📊 Detailed Results Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simulated Annealing Result")
            sa_result = st.session_state['sa_result']
            sa_solver = st.session_state['sa_solver']
            graph = st.session_state['graph']
            
            if sa_result and isinstance(sa_result, dict):
                sa_conflicts = sa_solver.count_conflicts(sa_result)
                st.metric("Conflicts", sa_conflicts)
                st.metric("Execution Time", f"{st.session_state.get('sa_time', 0):.3f}s")
                st.metric("Colors Used", len(set(sa_result.values())))
                
                # Visualize SA result
                fig_sa = visualize_graph(graph, sa_result, sa_solver.color_names)
                st.pyplot(fig_sa)
        
        with col2:
            st.subheader("CSP Result")
            csp_result = st.session_state.get('csp_result')
            csp_solver = st.session_state.get('csp_solver')
            
            if csp_result and csp_solver and isinstance(csp_result, dict):
                csp_conflicts = csp_solver.count_conflicts(csp_result)
                st.metric("Conflicts", csp_conflicts)
                st.metric("Execution Time", f"{st.session_state.get('csp_time', 0):.3f}s")
                st.metric("Colors Used", len(set(csp_result.values())))
                st.metric("Assignments", st.session_state.get('assignments_count', 0))
                st.metric("Backtracks", st.session_state.get('backtracks_count', 0))
                
                # Visualize CSP result
                fig_csp = visualize_graph(graph, csp_result, csp_solver.color_names)
                st.pyplot(fig_csp)
            else:
                st.warning("No solution found by CSP")
    
    elif 'best_coloring' in st.session_state and st.session_state['best_coloring']:
        st.markdown("---")
        st.subheader("📊 Results")
        
        best_coloring = st.session_state['best_coloring']
        solver = st.session_state['solver']
        graph = st.session_state['graph']
        
        # Ensure best_coloring is a dict
        if best_coloring and not isinstance(best_coloring, dict):
            st.error(f"Error: best_coloring is not a dict, it's {type(best_coloring)}")
            best_coloring = None
        
        # Calculate final conflicts
        final_conflicts = solver.count_conflicts(best_coloring) if best_coloring else 0
        
        # Display results in columns
        if algorithm == 'SA':
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Conflicts", final_conflicts)
            
            with col2:
                unique_colors = len(set(best_coloring.values())) if best_coloring and isinstance(best_coloring, dict) else 0
                st.metric("Colors Used", unique_colors)
            
            with col3:
                st.metric("Execution Time", f"{st.session_state.get('execution_time', 0):.2f}s")
            
            with col4:
                stop_reason = st.session_state.get('stop_reason', 'max_iterations')
                if stop_reason == "temperature_threshold":
                    st.metric("Stop Reason", "🌡️ Temperature")
                else:
                    st.metric("Stop Reason", "🔄 Max Iterations")
        else:  # CSP
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Conflicts", final_conflicts)
            
            with col2:
                unique_colors = len(set(best_coloring.values())) if best_coloring and isinstance(best_coloring, dict) else 0
                st.metric("Colors Used", unique_colors)
            
            with col3:
                st.metric("Execution Time", f"{st.session_state.get('execution_time', 0):.2f}s")
            
            with col4:
                st.metric("Assignments", st.session_state.get('assignments_count', 0))
            
            with col5:
                stop_reason = st.session_state.get('stop_reason', 'unknown')
                if stop_reason == "solution_found":
                    st.metric("Stop Reason", "✅ Solution")
                elif stop_reason == "max_assignments":
                    st.metric("Stop Reason", "⚠️ Max Assign")
                elif stop_reason == "no_solution":
                    st.metric("Stop Reason", "❌ No Solution")
                else:
                    st.metric("Stop Reason", "❓ Unknown")
        
        # Color assignment table
        if best_coloring and isinstance(best_coloring, dict):
            st.subheader("Color Assignment")
            color_data = []
            for vertex in sorted(best_coloring.keys()):
                color_idx = best_coloring[vertex]
                color_name = solver.color_names[color_idx % len(solver.color_names)]
                color_data.append({
                    "Vertex": vertex,
                    "Color": color_name,
                    "Color Index": color_idx
                })
            
            st.dataframe(color_data, use_container_width=True)
            
            # Visualize colored graph
            st.subheader("Colored Graph")
            fig_colored = visualize_graph(graph, best_coloring, solver.color_names)
            st.pyplot(fig_colored)
        elif best_coloring:
            st.error(f"Error: best_coloring is not a dict, it's {type(best_coloring)}")
        
        # Progress visualization (only for SA)
        if algorithm == 'SA' and show_progress and 'temp_history' in st.session_state:
            st.subheader("Algorithm Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Temperature Over Time**")
                fig_temp = plt.figure(figsize=(10, 4))
                plt.plot(st.session_state['temp_history'])
                plt.xlabel("Iteration")
                plt.ylabel("Temperature")
                plt.title("Temperature Cooling Schedule")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_temp)
            
            with col2:
                st.markdown("**Conflicts Over Time**")
                fig_conflicts = plt.figure(figsize=(10, 4))
                conflict_history = st.session_state.get('conflict_history', [])
                if conflict_history:
                    plt.plot(conflict_history)
                    plt.xlabel("Iteration")
                    plt.ylabel("Number of Conflicts")
                    plt.title("Solution Quality Improvement")
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig_conflicts)
                    
                    # Best solution indicator
                    best_iteration = min(range(len(conflict_history)), key=lambda i: conflict_history[i])
                    st.info(f"✨ Best solution found at iteration {best_iteration + 1} with {conflict_history[best_iteration]} conflicts")


if __name__ == "__main__":
    try:
        # Try to access streamlit runtime context
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            raise RuntimeError("Not running in Streamlit")
        main()
    except (ImportError, RuntimeError):
        import sys
        print("\n" + "="*60)
        print("ERROR: This is a Streamlit application!")
        print("="*60)
        print("\nTo run this app, use the following command:")
        print(f"\n    streamlit run app.py\n")
        print("Or from the current directory:")
        print(f"\n    streamlit run \"{__file__}\"\n")
        print("="*60 + "\n")
        sys.exit(1)
