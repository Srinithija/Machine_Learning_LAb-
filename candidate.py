# ---------------------------------------------
# Candidate Elimination Algorithm (FULL VERSION)
# ---------------------------------------------

# Training data
# Each example: ([attributes], label)
data = [
    (["Technical", "Senior", "Excellent", "Good", "Urban"], "Yes"),
    (["Technical", "Junior", "Excellent", "Good", "Urban"], "Yes"),
    (["Non-Technical", "Junior", "Average", "Poor", "Rural"], "No"),
    (["Technical", "Senior", "Average", "Good", "Rural"], "No"),
    (["Technical", "Senior", "Excellent", "Good", "Rural"], "Yes")
]

num_attributes = 5

# Initialize boundaries
S = ["Ø"] * num_attributes
G = [["?"] * num_attributes]


# ---------------- HELPER FUNCTIONS ----------------

def is_consistent(hypothesis, example):
    """Check if hypothesis covers the example"""
    for h, e in zip(hypothesis, example):
        if h != "?" and h != e:
            return False
    return True


def more_general(h1, h2):
    """
    Returns True if h1 is more general than or equal to h2
    """
    for x, y in zip(h1, h2):
        if x != "?" and (y == "?" or x != y):
            return False
    return True


def prune_G(G):
    """
    Keep only maximally general hypotheses
    """
    pruned = []
    for g in G:
        if not any(more_general(other, g) and other != g for other in G):
            pruned.append(g)
    return pruned


# ---------------- MAIN ALGORITHM ----------------

for index, (example, label) in enumerate(data, start=1):
    print(f"\nExample {index}: {example} -> {label}")

    # ---------- POSITIVE EXAMPLE ----------
    if label == "Yes":
        # Generalize S minimally
        for i in range(num_attributes):
            if S[i] == "Ø":
                S[i] = example[i]
            elif S[i] != example[i]:
                S[i] = "?"

        # Remove inconsistent hypotheses from G
        G = [g for g in G if is_consistent(g, example)]

    # ---------- NEGATIVE EXAMPLE ----------
    else:
        new_G = []
        for g in G:
            if is_consistent(g, example):
                for i in range(num_attributes):
                    if g[i] == "?":
                        if S[i] != "?" and S[i] != example[i]:
                            new_hypothesis = g.copy()
                            new_hypothesis[i] = S[i]
                            new_G.append(new_hypothesis)
            else:
                new_G.append(g)

        G = new_G

    # Prune G to keep only maximally general hypotheses
    G = prune_G(G)

    print("Specific Boundary S:", S)
    print("General Boundary G:", G)

# ---------------- FINAL RESULT ----------------

print("\n✅ FINAL RESULT")
print("Final Specific Hypothesis (S):", S)
print("Final General Hypotheses (G):", G)
