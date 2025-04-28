# subgoal_check.py

def isSubgoalReached(env, goal):
    """
    Custom subgoal reaching logic.

    Args:
        env: The environment object (ALEEnvironment).
        goal: An integer representing the subgoal id.

    Returns:
        bool: True if subgoal is considered reached, False otherwise.
    """

    # Option 1 (simple fallback): use environment's own method
    return env.goalReached(goal)

    # Option 2 (for more control later): 
    # You could match specific symbolic fluents if you extract them like:
    # symbolic_state = env.getSymbolicState()
    # and then check specific conditions for goal