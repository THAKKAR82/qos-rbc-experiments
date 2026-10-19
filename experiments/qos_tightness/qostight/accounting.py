"""Sample accounting: one block-encoding query of A/s uses O_row, O_col, O_ele, O_ele^dagger (Lemma D.22)."""


def samples_per_be_query(n_row_index, n_col_index, n_element):
    return int(n_row_index) + int(n_col_index) + 2 * int(n_element)


def amplified_be_samples(per_query, amplification_degree):
    """Theorem D.23: the A/2 block encoding queries the A/s block encoding O(d) times."""
    return int(amplification_degree) * int(per_query)
