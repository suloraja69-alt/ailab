def find_unit_clause(clauses):
    """
    Finds a unit clause in the list of clauses.
    """
    for clause in clauses:
        if len(clause) == 1:
            return clause[0]
    return None

def simplify_clauses(clauses, literal):
    simplified = []
    for clause in clauses:
        if literal in clause:
            continue  
        new_clause = [l for l in clause if l != -literal]
        if not new_clause:
            return None 
        simplified.append(new_clause)
    return simplified

def dpll(clauses, assignments):
    
    unit = find_unit_clause(clauses)
    while unit is not None:
        if unit not in assignments:
            assignments.append(unit)
        clauses = simplify_clauses(clauses, unit)
        if clauses is None:
            return False
        unit = find_unit_clause(clauses)

    if not clauses:
        return True 

    literal = clauses[0][0]

    new_clauses = simplify_clauses(clauses, literal)
    if new_clauses is not None:
        assignments.append(literal)
        if dpll(new_clauses, assignments):
            return True
        assignments.pop()

    new_clauses = simplify_clauses(clauses, -literal)
    if new_clauses is not None:
        assignments.append(-literal)
        if dpll(new_clauses, assignments):
            return True
        assignments.pop()

    return False

def main():
    A, B, C = 1, 2, 3 
    clauses = [[A, B], [-A, B], [-B, -C]]
    assignments = []
    if dpll(clauses, assignments):
        print("SATISFIABLE with assignments:", assignments)
    else:
        print("UNSATISFIABLE")

if __name__ == "__main__":
    main()
